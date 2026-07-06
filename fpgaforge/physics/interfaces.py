"""Timing budgets for external synchronous interfaces (source-synchronous / DDR).

An FPGA that meets its *internal* timing can still fail at the board boundary:
data launched to (or captured from) an external device -- SDR/DDR memory, an
ADC, another FPGA -- has to satisfy the receiver's setup/hold within one unit
interval, after every source of uncertainty eats into the eye. This module does
that eye budget:

    eye        = UI - (tco_spread + board_skew + clock_jitter + dcd)
    required   = tSU + tH
    margin     = eye - required

A non-negative margin means the interface closes with that slack; the setup and
hold margins are reported assuming a centered sampling point. All inputs are
datasheet/board numbers the user supplies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InterfaceBudget:
    name: str = "interface"
    clock_mhz: float = 100.0
    ddr: bool = False                 # data on both clock edges -> UI = half period
    tco_max_ns: float = 3.0           # FPGA clock-to-out, slow
    tco_min_ns: float = 1.0           # FPGA clock-to-out, fast
    setup_req_ns: float = 0.5         # receiver tSU
    hold_req_ns: float = 0.5          # receiver tH
    board_skew_ns: float = 0.2        # data-vs-clock trace skew
    clock_jitter_ns: float = 0.1      # peak-to-peak clock jitter
    dcd_ns: float = 0.0               # duty-cycle distortion (matters for DDR)


@dataclass
class InterfaceResult:
    name: str
    ui_ns: float = 0.0
    eye_ns: float = 0.0
    required_ns: float = 0.0
    margin_ns: float = 0.0
    setup_margin_ns: float = 0.0
    hold_margin_ns: float = 0.0
    uncertainty_ns: float = 0.0
    risks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return self.margin_ns >= 0 and self.setup_margin_ns >= 0 and self.hold_margin_ns >= 0

    def summary(self) -> str:
        verdict = "CLOSES" if self.passes else "FAILS"
        kind = "DDR" if self.ui_ns and self.notes and "DDR" in " ".join(self.notes) else "SDR"
        lines = [
            f"interface [{self.name}]: {verdict}",
            f"  unit interval : {self.ui_ns:.3f} ns ({kind})",
            f"  eye opening   : {self.eye_ns:.3f} ns "
            f"(UI - {self.uncertainty_ns:.3f} ns uncertainty)",
            f"  required tSU+tH: {self.required_ns:.3f} ns",
            f"  total margin  : {self.margin_ns:+.3f} ns",
            f"  setup / hold  : {self.setup_margin_ns:+.3f} / {self.hold_margin_ns:+.3f} ns",
        ]
        for r in self.risks:
            lines.append(f"  [risk] {r}")
        return "\n".join(lines)


def analyze_interface(b: InterfaceBudget) -> InterfaceResult:
    period_ns = 1000.0 / b.clock_mhz if b.clock_mhz > 0 else 0.0
    ui = period_ns / 2.0 if b.ddr else period_ns
    res = InterfaceResult(name=b.name, ui_ns=ui)
    if b.ddr:
        res.notes.append("DDR: data on both edges")

    tco_spread = max(0.0, b.tco_max_ns - b.tco_min_ns)
    uncertainty = tco_spread + b.board_skew_ns + b.clock_jitter_ns + b.dcd_ns
    res.uncertainty_ns = uncertainty
    res.eye_ns = ui - uncertainty
    res.required_ns = b.setup_req_ns + b.hold_req_ns
    res.margin_ns = res.eye_ns - res.required_ns

    # Centered sampling: split the eye around the sample point.
    half = res.eye_ns / 2.0
    res.setup_margin_ns = half - b.setup_req_ns
    res.hold_margin_ns = half - b.hold_req_ns

    if res.eye_ns <= 0:
        res.risks.append(
            f"uncertainty ({uncertainty:.2f} ns) consumes the entire "
            f"{ui:.2f} ns unit interval -> no eye; lower the data rate"
        )
    elif res.margin_ns < 0:
        res.risks.append(
            f"setup+hold ({res.required_ns:.2f} ns) exceeds the "
            f"{res.eye_ns:.2f} ns eye -> interface will not close"
        )
    if res.setup_margin_ns < 0 and res.eye_ns > 0:
        res.risks.append(f"negative setup margin ({res.setup_margin_ns:.2f} ns)")
    if res.hold_margin_ns < 0 and res.eye_ns > 0:
        res.risks.append(f"negative hold margin ({res.hold_margin_ns:.2f} ns)")
    return res
