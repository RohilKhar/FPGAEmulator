"""Power-distribution-network (PDN) impedance model.

Simultaneous-switching noise and rail collapse are ultimately a *power delivery*
problem: the network of bulk + decoupling capacitors, their parasitic ESR/ESL,
and the plane/mount inductance presents an impedance ``Z(f)`` to the die. As
long as ``|Z(f)|`` stays below the target impedance

    Z_target = (Vdd * ripple_fraction) / I_transient

across the band of interest, the supply holds up. The danger is *anti-resonance*
peaks between capacitor banks, where ``|Z|`` spikes above target.

This models each capacitor bank as a series RLC (C, ESL, ESR) in parallel,
optionally behind a plane/mount inductance, sweeps the impedance over frequency,
and flags the worst peak against the target. A first-order lumped model -- good
for choosing decoupling and spotting anti-resonance, not a substitute for a
full plane/cavity extraction.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field


@dataclass
class DecouplingCap:
    capacitance_uf: float
    esr_mohm: float = 10.0
    esl_nh: float = 0.5
    count: int = 1
    label: str = ""

    def impedance(self, f: float) -> complex:
        """Complex impedance of this bank (``count`` in parallel) at freq f."""
        w = 2 * math.pi * f
        c = self.capacitance_uf * 1e-6
        z = (self.esr_mohm * 1e-3) + 1j * (w * self.esl_nh * 1e-9 - 1.0 / (w * c))
        return z / max(1, self.count)

    def self_resonant_hz(self) -> float:
        return 1.0 / (2 * math.pi * math.sqrt(
            (self.esl_nh * 1e-9) * (self.capacitance_uf * 1e-6)))


@dataclass
class PDNConfig:
    vdd: float = 1.2
    ripple_fraction: float = 0.05         # allowed rail ripple (5%)
    transient_current_a: float = 1.0      # worst-case dynamic current step
    plane_capacitance_nf: float = 0.0     # inter-plane capacitance
    mount_inductance_nh: float = 0.3      # plane/via/mount L to the die
    # Voltage regulator: it actively holds the rail below its loop bandwidth, so
    # it sets the low-frequency impedance floor. Modelled as R + jwL.
    vrm_resistance_mohm: float = 5.0
    vrm_inductance_nh: float = 5.0
    f_min_hz: float = 1e3
    f_max_hz: float = 1e8
    caps: list[DecouplingCap] = field(default_factory=list)

    @property
    def z_target_ohm(self) -> float:
        if self.transient_current_a <= 0:
            return math.inf
        return (self.vdd * self.ripple_fraction) / self.transient_current_a


@dataclass
class PDNResult:
    z_target_ohm: float = 0.0
    worst_z_ohm: float = 0.0
    worst_freq_hz: float = 0.0
    curve: list[tuple[float, float]] = field(default_factory=list)  # (f, |Z|)
    resonances: list[tuple[str, float]] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    @property
    def meets_target(self) -> bool:
        return self.worst_z_ohm <= self.z_target_ohm

    def summary(self) -> str:
        verdict = "MEETS" if self.meets_target else "EXCEEDS"
        lines = [
            f"power delivery (PDN): {verdict} target",
            f"  Z_target   : {self.z_target_ohm*1000:.1f} mohm",
            f"  worst |Z|  : {self.worst_z_ohm*1000:.1f} mohm @ "
            f"{_fmt_hz(self.worst_freq_hz)}",
        ]
        for label, f in self.resonances:
            lines.append(f"  SRF {label:10}: {_fmt_hz(f)}")
        for r in self.risks:
            lines.append(f"  [risk] {r}")
        return "\n".join(lines)


def _fmt_hz(f: float) -> str:
    if f >= 1e6:
        return f"{f/1e6:.1f} MHz"
    if f >= 1e3:
        return f"{f/1e3:.1f} kHz"
    return f"{f:.0f} Hz"


def analyze_pdn(cfg: PDNConfig, points: int = 200) -> PDNResult:
    res = PDNResult(z_target_ohm=cfg.z_target_ohm)
    if not cfg.caps and cfg.plane_capacitance_nf <= 0:
        res.risks.append("no decoupling specified")
        return res

    worst_z, worst_f = 0.0, cfg.f_min_hz
    lo, hi = math.log10(cfg.f_min_hz), math.log10(cfg.f_max_hz)
    for i in range(points):
        f = 10 ** (lo + (hi - lo) * i / (points - 1))
        w = 2 * math.pi * f
        # Parallel admittance of the VRM + all capacitor banks + plane cap.
        y = 0j
        z_vrm = (cfg.vrm_resistance_mohm * 1e-3) + 1j * w * cfg.vrm_inductance_nh * 1e-9
        if z_vrm != 0:
            y += 1.0 / z_vrm
        for cap in cfg.caps:
            y += 1.0 / cap.impedance(f)
        if cfg.plane_capacitance_nf > 0:
            y += 1j * w * cfg.plane_capacitance_nf * 1e-9
        z_parallel = 1.0 / y if y != 0 else cmath.inf
        # Plane/mount inductance in series to the die.
        z = z_parallel + 1j * w * cfg.mount_inductance_nh * 1e-9
        mag = abs(z)
        res.curve.append((f, mag))
        if mag > worst_z:
            worst_z, worst_f = mag, f

    res.worst_z_ohm, res.worst_freq_hz = worst_z, worst_f
    for cap in cfg.caps:
        label = cap.label or f"{cap.capacitance_uf:g}uF"
        res.resonances.append((label, cap.self_resonant_hz()))

    if not res.meets_target:
        res.risks.append(
            f"PDN impedance {worst_z*1000:.1f} mohm exceeds target "
            f"{cfg.z_target_ohm*1000:.1f} mohm at {_fmt_hz(worst_f)} "
            "-> add decoupling near this frequency (anti-resonance); "
            "rail may droop under transient load"
        )
    return res
