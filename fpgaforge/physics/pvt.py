"""Process / Voltage / Temperature derating of timing.

Static timing analysis reports a delay at *one* operating corner. Silicon,
however, must work across the whole envelope: a slow-process part at minimum
supply voltage and maximum junction temperature is the slowest, and that is the
corner your clock must actually meet. This module derates a nominal STA Fmax to
every corner using a physically-grounded delay model:

    delay(P, V, T) = k_process(P) * (Vnom / V)^alpha * (1 + tempco * (T - Tnom))

* Process spread ``k_process`` bounds fab variation (slow > 1 > fast).
* The alpha-power law captures CMOS delay's rise as supply drops.
* A linear temperature coefficient captures delay's rise with junction temp.

The STA number is assumed to already correspond to some corner (for
``nextpnr-ice40`` that is the *slow* corner, i.e. already worst-case); we derate
*relative* to that corner so we never double-count. All coefficients are typical
values and are meant to be calibrated against a vendor datasheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PVTConfig:
    """Operating envelope + delay-sensitivity coefficients for a device."""

    device: str = "generic"
    vdd_nom: float = 1.2
    vdd_min: float = 1.14           # -5%
    vdd_max: float = 1.26           # +5%
    temp_min_c: float = 0.0
    temp_nom_c: float = 25.0
    temp_max_c: float = 85.0        # commercial junction max
    process_slow: float = 1.35      # slowest-process delay multiplier
    process_fast: float = 0.72      # fastest-process delay multiplier
    voltage_alpha: float = 1.3      # alpha-power-law exponent
    tempco_per_c: float = 0.0012    # fractional delay increase per deg C

    def delay_factor(self, process: str, voltage: float, temp_c: float) -> float:
        kp = {"slow": self.process_slow, "typ": 1.0, "fast": self.process_fast}[process]
        kv = (self.vdd_nom / voltage) ** self.voltage_alpha
        kt = 1.0 + self.tempco_per_c * (temp_c - self.temp_nom_c)
        return kp * kv * kt


# Lattice iCE40 UltraPlus: 1.2 V core, commercial grade.
ICE40_UP5K_PVT = PVTConfig(device="ice40_up5k")


@dataclass
class PVTCorner:
    name: str
    process: str
    voltage: float
    temp_c: float
    fmax_mhz: float
    delay_factor: float


@dataclass
class PVTResult:
    fmax_sta_mhz: float
    sta_corner: str
    target_mhz: float
    corners: list[PVTCorner] = field(default_factory=list)

    @property
    def guaranteed_fmax_mhz(self) -> float:
        """The slowest corner -- what you can promise across the envelope."""
        return min((c.fmax_mhz for c in self.corners), default=0.0)

    @property
    def best_fmax_mhz(self) -> float:
        return max((c.fmax_mhz for c in self.corners), default=0.0)

    @property
    def meets_worst_case(self) -> bool:
        return self.guaranteed_fmax_mhz >= self.target_mhz

    @property
    def worst_corner(self) -> PVTCorner | None:
        return min(self.corners, key=lambda c: c.fmax_mhz, default=None)

    def margin_pct(self) -> float:
        if self.target_mhz <= 0:
            return 0.0
        return 100.0 * (self.guaranteed_fmax_mhz / self.target_mhz - 1.0)

    def summary(self) -> str:
        lines = [
            "PVT timing sign-off",
            f"STA Fmax   : {self.fmax_sta_mhz:.1f} MHz (assumed {self.sta_corner} corner)",
            f"target     : {self.target_mhz:.1f} MHz",
            f"guaranteed : {self.guaranteed_fmax_mhz:.1f} MHz across P/V/T "
            f"({'MEETS' if self.meets_worst_case else 'FAILS'} worst case, "
            f"{self.margin_pct():+.0f}% margin)",
            "corners    :",
        ]
        for c in sorted(self.corners, key=lambda x: x.fmax_mhz):
            lines.append(
                f"  {c.name:14} P={c.process:4} V={c.voltage:.2f} T={c.temp_c:+.0f}C "
                f"-> {c.fmax_mhz:7.1f} MHz  (x{c.delay_factor:.3f})"
            )
        return "\n".join(lines)


# The corners worth reporting: the worst-case (SS/lowV/hotT), the typical, and
# the best-case (FF/highV/coldT), plus the two single-axis extremes engineers
# ask about most (hot and cold).
def _corner_defs(cfg: PVTConfig):
    return [
        ("slow_cold", "slow", cfg.vdd_min, cfg.temp_min_c),
        ("slow_hot",  "slow", cfg.vdd_min, cfg.temp_max_c),   # usually the worst
        ("typical",   "typ",  cfg.vdd_nom, cfg.temp_nom_c),
        ("fast_hot",  "fast", cfg.vdd_max, cfg.temp_max_c),
        ("fast_cold", "fast", cfg.vdd_max, cfg.temp_min_c),   # usually the best
    ]


def derate_fmax(
    fmax_sta_mhz: float,
    target_mhz: float,
    cfg: PVTConfig | None = None,
    sta_corner: str = "slow",
    sta_temp_c: float | None = None,
) -> PVTResult:
    """Derate a nominal STA Fmax across the P/V/T envelope.

    ``sta_corner`` states which corner the input Fmax already represents (for
    ``nextpnr-ice40`` this is ``"slow"`` -- already the slow-process worst case).
    Delays at each corner are computed *relative* to that reference so the STA
    number is neither optimistically inflated nor pessimistically re-penalized.

    ``sta_temp_c`` is the junction temperature the STA corner was characterized
    at (defaults to the corner's canonical temperature). Supplying the device
    *spec* temperature while ``cfg.temp_max_c`` carries the actual self-heated
    junction temp lets thermal feedback genuinely derate Fmax when the die runs
    hotter than the datasheet corner.
    """
    cfg = cfg or PVTConfig()
    # Temperature the STA corner represents (spec), decoupled from the actual
    # worst-case junction temp used for the hot corner.
    ref_temp = sta_temp_c
    if ref_temp is None:
        ref_temp = (cfg.temp_max_c if sta_corner == "slow"
                    else cfg.temp_nom_c if sta_corner == "typ" else cfg.temp_min_c)
    ref = cfg.delay_factor(
        sta_corner,
        cfg.vdd_min if sta_corner == "slow" else cfg.vdd_nom if sta_corner == "typ" else cfg.vdd_max,
        ref_temp,
    )
    corners: list[PVTCorner] = []
    for name, proc, v, t in _corner_defs(cfg):
        factor = cfg.delay_factor(proc, v, t)
        rel = factor / ref
        fmax = fmax_sta_mhz / rel if rel > 0 else 0.0
        corners.append(PVTCorner(name, proc, v, t, fmax, rel))
    return PVTResult(
        fmax_sta_mhz=fmax_sta_mhz, sta_corner=sta_corner,
        target_mhz=target_mhz, corners=corners,
    )
