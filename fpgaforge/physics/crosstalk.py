"""Coupled-line crosstalk between adjacent PCB traces.

When an aggressor net switches next to a quiet victim, mutual capacitance and
mutual inductance inject noise onto the victim: near-end (backward, NEXT) and
far-end (forward, FEXT) crosstalk. Using the standard weakly-coupled model with
per-unit-length coupling ratios ``k_c = Cm/C0`` and ``k_l = Lm/L0``:

    Kb   = (k_c + k_l) / 4                 (backward coupling coefficient)
    NEXT = Kb * Vdd * min(1, 2*Td / tr)    (saturates for long coupled runs)
    FEXT = 0.5 * |k_l - k_c| * (Td / tr) * Vdd

where ``Td`` is the coupled-section flight time and ``tr`` the aggressor edge.
Multiple synchronous aggressors add roughly linearly (worst case). These are
first-order estimates; extracted 2-D field-solver ratios make them exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrosstalkPair:
    victim: str = "victim"
    vdd: float = 3.3
    aggressor_rise_ns: float = 1.0
    coupling_len_mm: float = 25.0
    velocity_mm_ns: float = 150.0
    k_c: float = 0.05                 # capacitive coupling ratio Cm/C0
    k_l: float = 0.08                 # inductive coupling ratio  Lm/L0
    n_aggressors: int = 1
    noise_margin_v: float | None = None   # default: 0.3 * Vdd


@dataclass
class CrosstalkResult:
    victim: str
    next_v: float = 0.0               # near-end (backward) crosstalk
    fext_v: float = 0.0               # far-end (forward) crosstalk
    worst_v: float = 0.0
    noise_margin_v: float = 0.0
    margin_frac: float = 0.0          # worst noise as a fraction of the margin
    risks: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.risks

    def summary(self) -> str:
        verdict = "OK" if self.ok else "AT RISK"
        lines = [
            f"crosstalk [{self.victim}]: {verdict}",
            f"  near-end (NEXT): {self.next_v*1000:.0f} mV",
            f"  far-end  (FEXT): {self.fext_v*1000:.0f} mV",
            f"  worst / margin : {self.worst_v*1000:.0f} mV / {self.noise_margin_v*1000:.0f} mV "
            f"({self.margin_frac*100:.0f}%)",
        ]
        for r in self.risks:
            lines.append(f"  [risk] {r}")
        return "\n".join(lines)


def analyze_crosstalk(p: CrosstalkPair) -> CrosstalkResult:
    res = CrosstalkResult(victim=p.victim)
    td = p.coupling_len_mm / p.velocity_mm_ns          # coupled flight time (ns)
    tr = max(p.aggressor_rise_ns, 1e-3)

    kb = (p.k_c + p.k_l) / 4.0
    next_sat = kb * p.vdd
    next_scale = min(1.0, (2.0 * td) / tr)             # short runs don't saturate
    res.next_v = next_sat * next_scale * p.n_aggressors

    kf = 0.5 * abs(p.k_l - p.k_c)
    res.fext_v = kf * (td / tr) * p.vdd * p.n_aggressors

    res.worst_v = max(res.next_v, res.fext_v)
    res.noise_margin_v = p.noise_margin_v if p.noise_margin_v is not None else 0.3 * p.vdd
    res.margin_frac = res.worst_v / res.noise_margin_v if res.noise_margin_v else 0.0

    if res.worst_v > res.noise_margin_v:
        res.risks.append(
            f"crosstalk {res.worst_v*1000:.0f} mV exceeds the "
            f"{res.noise_margin_v*1000:.0f} mV noise margin -> may false-trigger the "
            "victim; increase spacing, add a guard trace, or slow the aggressor edge"
        )
    elif res.margin_frac > 0.6:
        res.risks.append(
            f"crosstalk uses {res.margin_frac*100:.0f}% of the noise margin -> tight; "
            "consider more spacing"
        )
    return res
