"""A pragmatic IBIS (I/O Buffer Information Specification) model loader.

IBIS is how silicon vendors publish the *measured* electrical behaviour of an
I/O buffer without revealing the transistor netlist: pull-up / pull-down I-V
curves, the output edge rate (``[Ramp]``), and the package parasitics
(``[Package]``). Feeding a real IBIS model into the signal-integrity analysis
replaces our generic driver-impedance / rise-time guesses with the numbers from
the datasheet -- the single biggest fidelity jump available short of transistor
SPICE.

This parser is deliberately focused (not a full IBIS validator): it extracts the
fields the SI models consume and derives an effective output impedance and a
10-90 rise time. Extraction is documented and approximate; it is meant to be
sanity-checked against the datasheet, not to certify IBIS compliance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .signal_integrity import Net, PackageModel

# IBIS metric-suffix multipliers (case-sensitive per the spec).
_SUFFIX = {
    "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3,
    "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18,
}


def _num(token: str) -> float | None:
    """Parse an IBIS numeric token like ``3.0nH``, ``0.5pF``, ``50``, ``NA``."""
    token = token.strip()
    if not token or token.upper() == "NA":
        return None
    m = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([TGMkmunpfa]?)", token)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2):
        value *= _SUFFIX[m.group(2)]
    return value


@dataclass
class IbisModel:
    name: str = ""
    model_type: str = ""
    vdd: float = 3.3
    pullup: list[tuple[float, float]] = field(default_factory=list)     # (V, I_typ)
    pulldown: list[tuple[float, float]] = field(default_factory=list)
    ramp_rise_v_per_ns: float = 0.0
    ramp_fall_v_per_ns: float = 0.0
    r_pkg: float = 0.0
    l_pkg_nh: float = 0.0
    c_pkg_pf: float = 0.0

    def output_impedance(self) -> float:
        """Effective driver impedance from the I-V curves' active region.

        Uses the median incremental resistance dV/dI over the mid-swing portion
        of the pull-down (and pull-up) tables, where the buffer behaves most
        like a series resistor. Falls back to 40 ohm if the tables are absent.
        """
        rs = []
        for table in (self.pulldown, self.pullup):
            r = _incremental_resistance(table, self.vdd)
            if r is not None:
                rs.append(r)
        if not rs:
            return 40.0
        return sum(rs) / len(rs)

    def rise_time_ns(self) -> float:
        """Approximate 10-90 rise time from the ramp slew rate."""
        slew = self.ramp_rise_v_per_ns or self.ramp_fall_v_per_ns
        if slew <= 0:
            return 1.0
        return 0.8 * self.vdd / slew


def _incremental_resistance(table: list[tuple[float, float]], vdd: float) -> float | None:
    """Median dV/dI across the 20%-80% window of the swing (linear region)."""
    pts = [(v, i) for v, i in table if i is not None and abs(i) > 1e-9]
    if len(pts) < 2:
        return None
    lo, hi = 0.2 * vdd, 0.8 * vdd
    window = [(v, i) for v, i in pts if lo <= abs(v) <= hi] or pts
    window.sort()
    slopes = []
    for (v0, i0), (v1, i1) in zip(window, window[1:]):
        di = i1 - i0
        if abs(di) > 1e-9:
            slopes.append(abs((v1 - v0) / di))
    if not slopes:
        return None
    slopes.sort()
    return slopes[len(slopes) // 2]


# ---------------------------------------------------------------------- #
def parse_ibis(text: str) -> dict[str, IbisModel]:
    """Parse the ``[Model]`` blocks of an IBIS file into IbisModel objects.

    ``[Package]`` parasitics are component-level (they precede ``[Model]``), so
    they are captured once and applied to every model that lacks its own.
    """
    lines = [ln.split("|", 1)[0].rstrip() for ln in text.splitlines()]
    models: dict[str, IbisModel] = {}
    current: IbisModel | None = None
    section = ""
    comp_pkg: dict[str, float] = {}     # component-level [Package] parasitics

    for ln in lines:
        if not ln.strip():
            continue
        kw = re.match(r"\s*\[([^\]]+)\]\s*(.*)", ln)
        if kw:
            key = kw.group(1).strip().lower()
            arg = kw.group(2).strip()
            if key == "model":
                current = IbisModel(name=arg)
                models[arg] = current
                section = "model"
            elif key in ("pulldown", "pullup", "ramp", "package", "voltage range",
                         "model_spec", "gnd clamp", "power clamp", "gnd_clamp",
                         "power_clamp", "temperature range"):
                section = key.replace(" ", "_")
            else:
                section = ""    # a section we don't consume
            if key == "voltage range" and current is not None:
                v = _num(arg.split()[0]) if arg else None
                if v is not None:
                    current.vdd = v
            continue

        toks = ln.split()
        # [Package] is component-level and appears before any [Model].
        if section == "package" and len(toks) >= 2:
            val = _num(toks[1])     # typ column
            key = toks[0].lower()
            if val is not None:
                comp_pkg[key] = val
            continue

        if current is None:
            continue
        if section == "model" and len(toks) >= 3 and toks[0].lower() == "model_type":
            current.model_type = toks[2]
        elif section in ("pulldown", "pullup") and len(toks) >= 2:
            v, i = _num(toks[0]), _num(toks[1])
            if v is not None:
                (current.pulldown if section == "pulldown" else current.pullup).append((v, i))
        elif section == "ramp" and toks[0].lower() in ("dv/dt_r", "dv/dt_f"):
            slew = _ramp_slew(toks[1]) if len(toks) > 1 else 0.0
            if toks[0].lower() == "dv/dt_r":
                current.ramp_rise_v_per_ns = slew
            else:
                current.ramp_fall_v_per_ns = slew

    # Apply component package parasitics to models that did not set their own.
    for m in models.values():
        if not m.r_pkg and "r_pkg" in comp_pkg:
            m.r_pkg = comp_pkg["r_pkg"]
        if not m.l_pkg_nh and "l_pkg" in comp_pkg:
            m.l_pkg_nh = comp_pkg["l_pkg"] * 1e9
        if not m.c_pkg_pf and "c_pkg" in comp_pkg:
            m.c_pkg_pf = comp_pkg["c_pkg"] * 1e12
    return models


def _ramp_slew(field_str: str) -> float:
    """Turn an IBIS ramp ``dV/time`` field (e.g. ``2.20/1.20n``) into V/ns."""
    if "/" not in field_str:
        return 0.0
    dv_s, dt_s = field_str.split("/", 1)
    dv, dt = _num(dv_s), _num(dt_s)
    if not dv or not dt or dt <= 0:
        return 0.0
    return dv / (dt * 1e9)      # dt is in seconds -> V per ns


def load_ibis(path: str | Path, model: str | None = None) -> IbisModel:
    """Load a single IbisModel from a file (the named one, or the first)."""
    models = parse_ibis(Path(path).read_text())
    if not models:
        raise ValueError(f"no [Model] blocks found in {path}")
    if model is not None:
        if model not in models:
            raise KeyError(f"model {model!r} not in {path}; have {list(models)}")
        return models[model]
    return next(iter(models.values()))


# ---------------------------------------------------------------------- #
def net_from_ibis(
    m: IbisModel,
    name: str = "io",
    trace_z0_ohm: float = 50.0,
    trace_len_mm: float = 50.0,
    load_pf: float = 10.0,
    n_simultaneous: int = 1,
    velocity_mm_ns: float = 150.0,
) -> Net:
    """Build an SI :class:`Net` whose driver comes from the IBIS model.

    The output impedance and rise time are taken from the *measured* IBIS I-V
    curves and ramp instead of generic defaults; trace/load remain board inputs.
    """
    return Net(
        name=name, vdd=m.vdd,
        drive_impedance_ohm=m.output_impedance(),
        driver_rise_ns=m.rise_time_ns(),
        load_pf=load_pf, trace_z0_ohm=trace_z0_ohm, trace_len_mm=trace_len_mm,
        n_simultaneous=n_simultaneous, velocity_mm_ns=velocity_mm_ns,
    )


def package_from_ibis(
    m: IbisModel, n_power_pins: int = 4, power_pin_inductance_nh: float = 3.0
) -> PackageModel:
    """Build a :class:`PackageModel` from the IBIS ``[Package]`` parasitics."""
    return PackageModel(
        name="ibis",
        lead_inductance_nh=m.l_pkg_nh or 5.0,
        lead_capacitance_pf=m.c_pkg_pf or 1.0,
        power_pin_inductance_nh=power_pin_inductance_nh,
        n_power_pins=n_power_pins,
    )
