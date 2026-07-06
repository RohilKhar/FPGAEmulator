"""Activity-based power and a thermal model that closes the PVT loop.

Timing derating (see :mod:`~fpgaforge.physics.pvt`) needs a junction temperature
-- but temperature is not a free parameter: it is set by how much power the
design burns and how well the package sheds heat. This module estimates power
from the *implemented* design and computes the resulting junction temperature,
which can then feed back into the worst-case Fmax so the timing sign-off is
self-consistent (and so a power/thermal budget gets checked at all).

Model:

* dynamic power  ``P = a * C_eff * V^2 * f`` summed over LUTs, FFs, BRAM, DSP
  (core voltage) and I/O (I/O voltage x load), where ``a`` is the switching
  activity factor;
* static power = device leakage, which rises roughly 2x per ~10 C;
* junction temp ``Tj = Tamb + P_total * theta_JA``, solved as a fixed point
  because leakage depends on ``Tj`` which depends on leakage.

Coefficients are order-of-magnitude typical values meant to be calibrated
against a vendor power estimator; the structure (and the feedback into PVT) is
the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PowerConfig:
    device: str = "ice40_up5k"
    vdd_core: float = 1.2
    vdd_io: float = 3.3
    # Effective switched capacitance per resource (pF), incl. local routing.
    c_lut_pf: float = 0.05
    c_ff_pf: float = 0.03
    c_bram_pf: float = 2.0
    c_dsp_pf: float = 3.0
    c_io_pf: float = 10.0
    activity: float = 0.15          # fraction of nodes toggling per clock
    io_activity: float = 0.25
    static_mw_nom: float = 0.30     # leakage at temp_nom_c
    static_tnom_c: float = 25.0
    leakage_double_per_c: float = 10.0   # leakage ~doubles per this many deg C
    theta_ja_c_per_w: float = 40.0  # junction-to-ambient thermal resistance
    ambient_c: float = 25.0
    tj_max_c: float = 85.0          # spec junction max (commercial)


ICE40_UP5K_POWER = PowerConfig(device="ice40_up5k")


@dataclass
class PowerResult:
    dynamic_core_mw: float = 0.0
    dynamic_io_mw: float = 0.0
    static_mw: float = 0.0
    junction_temp_c: float = 0.0
    ambient_c: float = 25.0
    tj_max_c: float = 85.0
    freq_mhz: float = 0.0
    activity: float = 0.15
    notes: list[str] = field(default_factory=list)

    @property
    def total_mw(self) -> float:
        return self.dynamic_core_mw + self.dynamic_io_mw + self.static_mw

    @property
    def within_thermal_limit(self) -> bool:
        return self.junction_temp_c <= self.tj_max_c

    def summary(self) -> str:
        lines = [
            "power & thermal",
            f"  frequency  : {self.freq_mhz:.1f} MHz, activity {self.activity:.0%}",
            f"  dynamic    : {self.dynamic_core_mw:.2f} mW core + "
            f"{self.dynamic_io_mw:.2f} mW I/O",
            f"  static     : {self.static_mw:.2f} mW (leakage @ Tj)",
            f"  total      : {self.total_mw:.2f} mW",
            f"  junction T : {self.junction_temp_c:.1f} C "
            f"(ambient {self.ambient_c:.0f} C, "
            f"{'OK' if self.within_thermal_limit else 'OVER'} vs {self.tj_max_c:.0f} C max)",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def estimate_power(
    resources: dict,
    freq_mhz: float,
    io_count: int = 0,
    cfg: PowerConfig | None = None,
    activity: float | None = None,
) -> PowerResult:
    """Estimate power and the resulting junction temperature.

    ``resources`` keys: ``luts, ffs, bram, dsp`` (missing -> 0). ``freq_mhz`` is
    the operating clock.
    """
    cfg = cfg or PowerConfig()
    a = cfg.activity if activity is None else activity
    f = freq_mhz * 1e6
    v2 = cfg.vdd_core ** 2

    luts = resources.get("luts", 0)
    ffs = resources.get("ffs", 0)
    bram = resources.get("bram", 0)
    dsp = resources.get("dsp", 0)

    def dyn(n, c_pf):
        return a * n * (c_pf * 1e-12) * v2 * f    # watts

    dyn_core_w = (dyn(luts, cfg.c_lut_pf) + dyn(ffs, cfg.c_ff_pf)
                  + dyn(bram, cfg.c_bram_pf) + dyn(dsp, cfg.c_dsp_pf))
    dyn_io_w = cfg.io_activity * io_count * (cfg.c_io_pf * 1e-12) * (cfg.vdd_io ** 2) * f

    dyn_core_mw = dyn_core_w * 1e3
    dyn_io_mw = dyn_io_w * 1e3

    # Fixed-point on leakage <-> junction temperature. Leakage rises with Tj,
    # which raises Tj -> if there is no stable solution the die is in thermal
    # runaway. Cap the exponent to avoid overflow and detect divergence.
    tj = cfg.ambient_c
    static_mw = cfg.static_mw_nom
    runaway = False
    for _ in range(100):
        exp = min((tj - cfg.static_tnom_c) / cfg.leakage_double_per_c, 40.0)
        static_mw = cfg.static_mw_nom * 2 ** exp
        total_w = (dyn_core_mw + dyn_io_mw + static_mw) / 1e3
        tj_new = cfg.ambient_c + total_w * cfg.theta_ja_c_per_w
        if tj_new > 1000.0:            # no stable fixed point in a sane range
            runaway = True
            tj = tj_new
            break
        if abs(tj_new - tj) < 1e-3:
            tj = tj_new
            break
        tj = tj_new

    res = PowerResult(
        dynamic_core_mw=dyn_core_mw, dynamic_io_mw=dyn_io_mw, static_mw=static_mw,
        junction_temp_c=min(tj, 1e4), ambient_c=cfg.ambient_c, tj_max_c=cfg.tj_max_c,
        freq_mhz=freq_mhz, activity=a,
    )
    if runaway:
        res.notes.append(
            "thermal runaway: leakage has no stable operating point -> the design "
            "cannot dissipate this power in this package; reduce power or cool it"
        )
    elif not res.within_thermal_limit:
        res.notes.append(
            f"junction temp {tj:.1f} C exceeds {cfg.tj_max_c:.0f} C -> add cooling, "
            "lower activity/frequency, or use a bigger package (lower theta_JA)"
        )
    return res
