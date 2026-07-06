"""Timing sign-off from real static timing analysis.

nextpnr's STA walks the actual per-LUT and per-net delays of the placed &
routed design. This module parses its critical-path report into structured
data (clk-to-q / logic / routing stages), computes slack against the target
clock, and reports whether the design will run at speed on the FPGA -- plus the
SDF artifact for delay-annotated simulation.

The parser is pure and unit-tested against captured nextpnr output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# "Info:   logic  0.28  4.10 Source cell.PIN"  /  "Info:  routing 1.76 3.15 Net ..."
_STAGE_RE = re.compile(
    r"^\s*(?:Info:\s*)?(?P<kind>clk-to-q|logic|routing|setup)\s+"
    r"(?P<curr>[\d.]+)\s+(?P<total>[\d.]+)\s+(?P<detail>.*\S)?\s*$"
)
_PATH_HDR_RE = re.compile(
    r"Critical path report for (?:clock '(?P<clk>[^']+)'|(?P<xdomain>cross-domain[^:]*))"
)


@dataclass
class TimingStage:
    kind: str            # "clk-to-q" | "logic" | "routing" | "setup"
    delay_ns: float
    cumulative_ns: float
    detail: str = ""


@dataclass
class CriticalPath:
    clock: str
    stages: list[TimingStage] = field(default_factory=list)

    @property
    def total_ns(self) -> float:
        return max((s.cumulative_ns for s in self.stages), default=0.0)

    @property
    def logic_ns(self) -> float:
        return sum(s.delay_ns for s in self.stages if s.kind in ("logic", "clk-to-q"))

    @property
    def routing_ns(self) -> float:
        return sum(s.delay_ns for s in self.stages if s.kind == "routing")

    @property
    def n_logic_stages(self) -> int:
        return sum(1 for s in self.stages if s.kind in ("logic", "clk-to-q"))


def parse_critical_paths(log: str) -> list[CriticalPath]:
    """Parse all critical-path reports from a nextpnr log."""
    paths: list[CriticalPath] = []
    current: CriticalPath | None = None

    for line in log.splitlines():
        hdr = _PATH_HDR_RE.search(line)
        if hdr:
            clk = hdr.group("clk") or (hdr.group("xdomain") or "path").strip()
            current = CriticalPath(clock=clk)
            paths.append(current)
            continue
        if current is None:
            continue
        m = _STAGE_RE.match(line)
        if m:
            current.stages.append(
                TimingStage(
                    kind=m.group("kind"),
                    delay_ns=float(m.group("curr")),
                    cumulative_ns=float(m.group("total")),
                    detail=(m.group("detail") or "").strip(),
                )
            )
    return [p for p in paths if p.stages]


@dataclass
class TimingReport:
    design_id: str
    meets_timing: bool = False
    fmax_mhz: float = 0.0
    target_mhz: float = 0.0
    slack_ns: float = 0.0
    routed_ok: bool = False
    worst_path: CriticalPath | None = None
    sdf_path: str | None = None
    routed_netlist_path: str | None = None
    error: str | None = None

    def summary(self) -> str:
        verdict = "MET" if self.meets_timing else "VIOLATED"
        lines = [
            f"timing sign-off: {verdict}",
            f"design : {self.design_id}",
            f"clock  : {self.fmax_mhz:.1f} MHz achievable vs {self.target_mhz:.1f} MHz target",
            f"slack  : {self.slack_ns:+.2f} ns",
        ]
        wp = self.worst_path
        if wp:
            lines.append(
                f"critical path ({wp.clock}): {wp.total_ns:.2f} ns "
                f"= {wp.logic_ns:.2f} ns logic ({wp.n_logic_stages} LUT/cell stages) "
                f"+ {wp.routing_ns:.2f} ns routing"
            )
            src = next((s for s in wp.stages if s.detail), None)
            snk = next((s for s in reversed(wp.stages) if s.detail), None)
            if src:
                lines.append(f"  from: {src.detail}")
            if snk and snk is not src:
                lines.append(f"  to  : {snk.detail}")
        if self.sdf_path:
            lines.append(f"sdf    : {self.sdf_path}")
        if self.error:
            lines.append(f"error  : {self.error}")
        return "\n".join(lines)


def signoff(
    rtl: str | Sequence[str],
    top: str,
    target_fpga: str = "ice40_up5k",
    clock_ns: float = 10.0,
    seed: int = 1,
    workdir: str | Path = ".runs/timing",
    backend=None,
) -> TimingReport:
    """Run place & route and produce a real timing sign-off with SDF artifact."""
    from .backends.base import Design, FlowOptions
    from .backends.ice40 import Ice40Backend

    rtl_files = [rtl] if isinstance(rtl, str) else list(rtl)
    design = Design(rtl_files=tuple(rtl_files), top=top, target=target_fpga, clock_ns=clock_ns)
    if backend is None:
        if target_fpga.startswith("ecp5"):
            from .backends.ecp5 import Ecp5Backend

            backend = Ecp5Backend(emit_timing_artifacts=True)
        else:
            backend = Ice40Backend(emit_timing_artifacts=True)

    report = TimingReport(design_id=design.design_id(), target_mhz=design.target_freq_mhz)
    if not backend.is_available():
        report.error = "timing sign-off requires yosys and nextpnr-ice40 on PATH"
        return report

    run = backend.run(design, FlowOptions(seed=seed), Path(workdir))
    report.routed_ok = run.metrics.routed_ok
    report.fmax_mhz = run.metrics.fmax_mhz
    report.sdf_path = run.sdf_path
    report.routed_netlist_path = run.routed_netlist_path
    if not run.metrics.routed_ok:
        report.error = run.error or "place-and-route failed"
        return report

    report.meets_timing = run.metrics.meets_timing
    if report.fmax_mhz > 0 and report.target_mhz > 0:
        report.slack_ns = 1000.0 / report.target_mhz - 1000.0 / report.fmax_mhz

    paths = parse_critical_paths(run.log)
    if paths:
        # Fmax is set by register-to-register paths; prefer those over
        # clk-to-output / cross-domain (async) paths when reporting the worst.
        intra = [
            p for p in paths
            if "cross-domain" not in p.clock.lower() and "async" not in p.clock.lower()
        ]
        report.worst_path = max(intra or paths, key=lambda p: p.total_ns)
    return report
