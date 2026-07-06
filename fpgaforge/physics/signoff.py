"""Physical sign-off: fuse PVT, signal integrity, and interface budgets.

Given the STA Fmax from the implementation flow plus a description of the board
(I/O nets and external interfaces), produce a single physical verdict that
captures the effects the netlist cannot: does the clock still close at the slow
P/V/T corner, do the I/O settle cleanly without damaging overshoot, and do the
external interfaces meet setup/hold across the eye budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .pvt import PVTConfig, PVTResult, derate_fmax
from .signal_integrity import Net, PackageModel, SIResult, analyze_net
from .interfaces import InterfaceBudget, InterfaceResult, analyze_interface
from .crosstalk import CrosstalkPair, CrosstalkResult, analyze_crosstalk
from .pdn import PDNConfig, PDNResult, analyze_pdn
from .power import PowerConfig, PowerResult, estimate_power


@dataclass
class PhysicalReport:
    design_id: str = ""
    pvt: PVTResult | None = None
    si: list[SIResult] = field(default_factory=list)
    crosstalk: list[CrosstalkResult] = field(default_factory=list)
    interfaces: list[InterfaceResult] = field(default_factory=list)
    pdn: PDNResult | None = None
    power: PowerResult | None = None

    @property
    def verdict(self) -> str:
        blocked = False
        risky = False
        if self.pvt is not None and not self.pvt.meets_worst_case:
            blocked = True
        for s in self.si:
            if s.risks:
                risky = True
        for x in self.crosstalk:
            if x.risks:
                risky = True
        for i in self.interfaces:
            if not i.passes:
                blocked = True
        if self.pdn is not None and not self.pdn.meets_target:
            risky = True
        if self.power is not None and not self.power.within_thermal_limit:
            blocked = True      # over junction-temp spec -> will not run reliably
        if blocked:
            return "BLOCKED"
        if risky:
            return "AT_RISK"
        return "PASS"

    def summary(self) -> str:
        lines = [
            f"physical sign-off: {self.verdict}",
            f"design : {self.design_id}",
        ]
        if self.power is not None:
            lines.append("")
            lines.append(self.power.summary())
        if self.pvt is not None:
            lines.append("")
            lines.append(self.pvt.summary())
        for s in self.si:
            lines.append("")
            lines.append(s.summary())
        for x in self.crosstalk:
            lines.append("")
            lines.append(x.summary())
        if self.pdn is not None:
            lines.append("")
            lines.append(self.pdn.summary())
        for i in self.interfaces:
            lines.append("")
            lines.append(i.summary())
        lines.append("")
        lines.append(
            "note   : physically-grounded models; accuracy is bounded by the board/"
            "package parameters supplied. Validate on hardware before volume."
        )
        return "\n".join(lines)


def physical_signoff(
    fmax_sta_mhz: float,
    target_mhz: float,
    design_id: str = "",
    pvt_cfg: PVTConfig | None = None,
    sta_corner: str = "slow",
    nets: list[Net] | None = None,
    package: PackageModel | None = None,
    interfaces: list[InterfaceBudget] | None = None,
    crosstalk: list[CrosstalkPair] | None = None,
    pdn: PDNConfig | None = None,
    resources: dict | None = None,
    io_count: int = 0,
    power_cfg: PowerConfig | None = None,
    activity: float | None = None,
    vcd: str | None = None,
) -> PhysicalReport:
    """Run every physical model and fuse the results.

    When ``resources`` are supplied, power and junction temperature are estimated
    first and the self-heated ``Tj`` is fed into the PVT envelope so the
    guaranteed Fmax is thermally self-consistent.

    If ``activity`` is not given but ``vcd`` is, the switching activity is
    *measured* from the waveform (per-bit toggle rate per cycle) instead of using
    the model default -- grounding power in the design's real behavior.
    """
    report = PhysicalReport(design_id=design_id)

    # Prefer a measured activity factor from a real simulation waveform.
    measured_note = None
    if activity is None and vcd:
        from ..vcd import measure_activity

        act = measure_activity(vcd)
        if act.cycles > 0 and act.n_bits > 0:
            activity = act.activity
            measured_note = (
                f"activity {act.activity:.1%} measured from {act.n_bits} bits over "
                f"{act.cycles} cycles ({vcd})"
            )

    # Power first: it sets the junction temperature the PVT worst case runs at.
    sta_temp = None
    if resources is not None:
        report.power = estimate_power(
            resources, freq_mhz=target_mhz or fmax_sta_mhz,
            io_count=io_count, cfg=power_cfg, activity=activity,
        )
        if measured_note:
            report.power.notes.insert(0, measured_note)
        if pvt_cfg is None:
            pvt_cfg = PVTConfig()
        # Use the actual self-heated junction temp as the hot corner, keeping the
        # spec temperature as the STA reference so hotter-than-spec truly derates.
        import dataclasses as _dc

        sta_temp = pvt_cfg.temp_max_c
        tj = max(report.power.junction_temp_c, sta_temp)
        pvt_cfg = _dc.replace(pvt_cfg, temp_max_c=tj)

    if fmax_sta_mhz > 0:
        report.pvt = derate_fmax(fmax_sta_mhz, target_mhz, pvt_cfg, sta_corner,
                                 sta_temp_c=sta_temp)
    for net in (nets or []):
        report.si.append(analyze_net(net, package))
    for x in (crosstalk or []):
        report.crosstalk.append(analyze_crosstalk(x))
    if pdn is not None:
        report.pdn = analyze_pdn(pdn)
    for b in (interfaces or []):
        report.interfaces.append(analyze_interface(b))
    return report
