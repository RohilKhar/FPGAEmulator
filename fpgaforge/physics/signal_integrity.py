"""Signal-integrity models for FPGA I/O.

When a pin switches, it is no longer "just a logic value": it is a driver with
finite output impedance pushing an edge down a transmission line into a load,
through package lead inductance, sharing a return path with its neighbours. This
module models the four effects that most often break first-silicon I/O:

* **Edge / rise time** -- driver slew combined with the RC of its load.
* **Transmission-line reflections** -- overshoot/undershoot when the source is
  not back-terminated to the trace impedance (``Zout != Z0``).
* **Package LC ringing** -- lead inductance resonating with load capacitance.
* **Simultaneous-switching noise (SSN / ground bounce)** -- ``N * L * dI/dt``
  induced on the shared power/ground return when many outputs switch together.

Each is a closed-form, physically-grounded estimate. For a genuine circuit-level
answer, :func:`spice_deck` emits an ngspice netlist and :func:`analyze_net` will
run it automatically when ``ngspice`` is on PATH, folding the simulated
overshoot/settling back into the result.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PackageModel:
    """Package parasitics that matter for SI."""

    name: str = "generic"
    lead_inductance_nh: float = 5.0       # per-signal bond+lead inductance
    lead_capacitance_pf: float = 1.0
    power_pin_inductance_nh: float = 3.0  # per power/ground pin
    n_power_pins: int = 4                 # pins sharing the switching return


# iCE40 UltraPlus in the sg48 QFN (small package -> relatively few power pins).
ICE40_SG48 = PackageModel(
    name="sg48", lead_inductance_nh=4.5, lead_capacitance_pf=0.8,
    power_pin_inductance_nh=3.5, n_power_pins=4,
)


@dataclass
class Net:
    """A driven net: driver, board trace, and load."""

    name: str = "io"
    vdd: float = 3.3
    drive_impedance_ohm: float = 45.0     # FPGA output driver impedance
    driver_rise_ns: float = 1.0           # intrinsic driver 10-90% slew
    load_pf: float = 10.0                 # receiver + trace capacitance
    trace_z0_ohm: float = 50.0            # PCB trace characteristic impedance
    trace_len_mm: float = 50.0
    velocity_mm_ns: float = 150.0         # ~microstrip on FR-4
    n_simultaneous: int = 1               # outputs switching together on this bank
    overshoot_limit_frac: float = 0.20    # reliability limit as a fraction of Vdd


@dataclass
class SIResult:
    net: str
    rise_time_ns: float = 0.0
    flight_time_ns: float = 0.0
    electrically_long: bool = False
    reflection_overshoot_frac: float = 0.0
    ringing_overshoot_frac: float = 0.0
    overshoot_frac: float = 0.0           # worst of the two (or SPICE)
    peak_voltage: float = 0.0
    settle_time_ns: float = 0.0
    max_toggle_mhz: float = 0.0
    ssn_volts: float = 0.0
    ssn_margin_frac: float = 0.0          # SSN as a fraction of Vdd
    needs_termination: bool = False
    spice_ran: bool = False
    risks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.risks

    def summary(self) -> str:
        verdict = "OK" if self.ok else "AT RISK"
        lines = [
            f"signal integrity [{self.net}]: {verdict}"
            + ("  (ngspice-verified)" if self.spice_ran else ""),
            f"  rise time      : {self.rise_time_ns:.2f} ns",
            f"  flight time    : {self.flight_time_ns:.2f} ns "
            f"({'transmission line' if self.electrically_long else 'lumped'})",
            f"  overshoot      : {self.overshoot_frac*100:.0f}% of Vdd "
            f"(peak {self.peak_voltage:.2f} V)",
            f"  settling       : {self.settle_time_ns:.2f} ns "
            f"-> safe toggle <= {self.max_toggle_mhz:.0f} MHz",
            f"  SSN/gnd bounce : {self.ssn_volts:.2f} V "
            f"({self.ssn_margin_frac*100:.0f}% of Vdd)",
        ]
        if self.needs_termination:
            lines.append("  recommendation : add source series termination to match Z0")
        for r in self.risks:
            lines.append(f"  [risk] {r}")
        return "\n".join(lines)


def _rise_time_ns(net: Net) -> float:
    """Driver slew combined (root-sum-square) with the load RC 10-90% time."""
    rc_10_90 = 2.2 * net.drive_impedance_ohm * (net.load_pf * 1e-12) * 1e9  # ns
    return math.sqrt(net.driver_rise_ns ** 2 + rc_10_90 ** 2)


def _reflection_overshoot(net: Net) -> float:
    """Fractional overshoot from an under-terminated source on a long line.

    Launched step V0 = Vdd*Z0/(Z0+Zout); at a high-Z load it nearly doubles, so
    the peak overshoot fraction is (Z0-Zout)/(Z0+Zout) when Zout < Z0.
    """
    z0, zs = net.trace_z0_ohm, net.drive_impedance_ohm
    if zs >= z0:
        return 0.0
    return (z0 - zs) / (z0 + zs)


def _ringing_overshoot(net: Net, pkg: PackageModel) -> float:
    """Second-order LC ringing overshoot for a series-L / load-C / series-R net."""
    L = pkg.lead_inductance_nh * 1e-9
    C = net.load_pf * 1e-12
    R = net.drive_impedance_ohm
    if L <= 0 or C <= 0:
        return 0.0
    zeta = (R / 2.0) * math.sqrt(C / L)      # damping ratio
    if zeta >= 1.0:
        return 0.0                            # overdamped -> no overshoot
    return math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta ** 2))


def _settle_time_ns(net: Net, pkg: PackageModel, long_line: bool, tol: float = 0.05) -> float:
    if long_line:
        gs = abs((net.drive_impedance_ohm - net.trace_z0_ohm) /
                 (net.drive_impedance_ohm + net.trace_z0_ohm))
        tflight = net.trace_len_mm / net.velocity_mm_ns
        if gs <= 1e-6:
            return 2 * tflight
        n = max(1.0, math.log(tol) / math.log(gs))   # round trips to decay
        return n * 2 * tflight
    # Lumped LC envelope decay: t ~ -ln(tol)/(zeta*wn)
    L = pkg.lead_inductance_nh * 1e-9
    C = net.load_pf * 1e-12
    R = net.drive_impedance_ohm
    if L <= 0 or C <= 0:
        return _rise_time_ns(net) * 2
    wn = 1.0 / math.sqrt(L * C)              # rad/s
    zeta = (R / 2.0) * math.sqrt(C / L)
    zeta = max(zeta, 0.05)
    return (-math.log(tol) / (zeta * wn)) * 1e9


def _ssn_volts(net: Net, pkg: PackageModel, rise_ns: float) -> float:
    """Ground bounce ~ N * L_return * dI/dt on the shared power return."""
    l_return = (pkg.power_pin_inductance_nh * 1e-9) / max(1, pkg.n_power_pins)
    tr = max(rise_ns, 1e-3) * 1e-9
    # Peak transient current I ~ C*Vdd/tr; dI/dt ~ I/tr.
    di_dt = (net.load_pf * 1e-12) * net.vdd / (tr ** 2)
    return net.n_simultaneous * l_return * di_dt


def analyze_net(net: Net, pkg: PackageModel | None = None, use_spice: bool = True) -> SIResult:
    """Full SI analysis of one net (analytical, plus ngspice if available)."""
    pkg = pkg or PackageModel()
    res = SIResult(net=net.name)

    res.rise_time_ns = _rise_time_ns(net)
    tflight = net.trace_len_mm / net.velocity_mm_ns
    res.flight_time_ns = tflight
    # Electrically long if the round-trip delay is comparable to the edge.
    res.electrically_long = (2 * tflight) > (0.5 * res.rise_time_ns)

    res.reflection_overshoot_frac = (
        _reflection_overshoot(net) if res.electrically_long else 0.0
    )
    res.ringing_overshoot_frac = _ringing_overshoot(net, pkg)
    res.overshoot_frac = max(res.reflection_overshoot_frac, res.ringing_overshoot_frac)
    res.peak_voltage = net.vdd * (1.0 + res.overshoot_frac)

    res.settle_time_ns = _settle_time_ns(net, pkg, res.electrically_long)
    res.max_toggle_mhz = 1000.0 / (2.0 * res.settle_time_ns) if res.settle_time_ns > 0 else 0.0

    res.ssn_volts = _ssn_volts(net, pkg, res.rise_time_ns)
    res.ssn_margin_frac = res.ssn_volts / net.vdd if net.vdd else 0.0

    res.needs_termination = (
        res.electrically_long and net.drive_impedance_ohm < net.trace_z0_ohm * 0.8
    )

    # Optional: replace analytical overshoot/settling with a real simulation.
    if use_spice and shutil.which("ngspice"):
        sim = _run_spice(net)
        if sim is not None:
            res.spice_ran = True
            peak, settle = sim
            res.peak_voltage = peak
            res.overshoot_frac = max(0.0, peak / net.vdd - 1.0)
            if settle > 0:
                res.settle_time_ns = settle
                res.max_toggle_mhz = 1000.0 / (2.0 * settle)
            res.notes.append("overshoot/settling from ngspice transient simulation")

    # ---- risk flags ----
    if res.overshoot_frac > net.overshoot_limit_frac:
        res.risks.append(
            f"overshoot {res.overshoot_frac*100:.0f}% exceeds "
            f"{net.overshoot_limit_frac*100:.0f}% limit (peak {res.peak_voltage:.2f} V) "
            "-> reliability/latch-up risk; terminate or slow the edge"
        )
    if res.ssn_margin_frac > 0.30:
        res.risks.append(
            f"SSN {res.ssn_volts:.2f} V is {res.ssn_margin_frac*100:.0f}% of Vdd "
            "-> may corrupt quiet I/O; add power pins/decoupling or stagger outputs"
        )
    if res.needs_termination:
        res.notes.append(
            "under-terminated transmission line; series termination recommended"
        )
    return res


# ---------------------------------------------------------------------- #
_EDGE_START_NS = 1.0        # PULSE launch time in the deck


def spice_deck(net: Net, data_file: str = "si_out.data") -> str:
    """Emit an ngspice transient deck: Thevenin driver -> T-line -> C load.

    The rising edge launches at t=1 ns; the run captures one full settling
    window and writes ``time v(load)`` to ``data_file`` for post-processing.
    """
    tflight_ns = net.trace_len_mm / net.velocity_mm_ns
    tr_ns = max(net.driver_rise_ns, 0.05)
    period_ns = max(20.0, tflight_ns * 30, tr_ns * 30)
    cload_f = net.load_pf * 1e-12
    step_ns = min(tr_ns, tflight_ns if tflight_ns > 0 else tr_ns) / 40.0
    step_ns = max(step_ns, 1e-4)
    return f"""* fpgaforge signal-integrity transient for net {net.name}
Vsrc src 0 PULSE(0 {net.vdd} {_EDGE_START_NS}n {tr_ns}n {tr_ns}n {period_ns}n {period_ns*2}n)
Rout src drv {net.drive_impedance_ohm}
T1 drv 0 load 0 Z0={net.trace_z0_ohm} TD={tflight_ns}n
Cload load 0 {cload_f}
.tran {step_ns}n {period_ns}n
.control
run
wrdata {data_file} v(load)
.endc
.end
"""


def _run_spice(net: Net) -> tuple[float, float] | None:
    """Run ngspice on the net's deck; return (peak_voltage, settle_time_ns)."""
    try:
        with tempfile.TemporaryDirectory() as d:
            deck = Path(d) / "si.cir"
            deck.write_text(spice_deck(net))
            proc = subprocess.run(
                ["ngspice", "-b", str(deck)],
                cwd=d, capture_output=True, text=True, timeout=30,
            )
            data = Path(d) / "si_out.data"
            if not data.exists():
                return None
            samples = _parse_wrdata(data.read_text())
        if not samples:
            return None
        return _measure_transient(samples, net.vdd)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _parse_wrdata(text: str) -> list[tuple[float, float]]:
    """Parse ngspice ``wrdata`` output (columns: time value) into samples."""
    out: list[tuple[float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                out.append((float(parts[0]), float(parts[-1])))
            except ValueError:
                continue
    return out


def _measure_transient(samples: list[tuple[float, float]], vdd: float,
                       tol_frac: float = 0.05) -> tuple[float, float]:
    """From (t, v) samples return (peak_voltage, settle_time_ns_after_edge)."""
    peak = max(v for _t, v in samples)
    final = samples[-1][1]
    band = tol_frac * vdd
    settle_t = _EDGE_START_NS * 1e-9
    for t, v in samples:
        if abs(v - final) > band:
            settle_t = t
    settle_ns = max(0.0, (settle_t - _EDGE_START_NS * 1e-9) * 1e9)
    return peak, settle_ns
