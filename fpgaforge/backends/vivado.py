"""VivadoBackend: the AMD/Xilinx vendor flow (Vivado in batch mode).

AMD parts (7-series, UltraScale+, Versal, Zynq) have no open bitstream tools, so
we drive the vendor's Vivado in ``-mode batch`` with a generated Tcl script:
``synth_design -> opt/place/route -> report_{utilization,timing,power} ->
write_verilog (post-impl netlist) -> write_sdf -> write_bitstream``. We then
parse the reports into the same :class:`RunMetrics` every other backend fills,
so the corpus/model/optimizer/readiness stack works unchanged.

The bitstream is encrypted/undocumented, so this backend deliberately does NOT
support bit-level reconstruction. Instead it emits the post-implementation
netlist + SDF, which the timing engine and the *netlist-level* equivalence tier
consume (see fpgaforge/devices.py: DeviceInfo.equivalence_tier == "netlist").

Everything is gated behind :meth:`is_available` (``vivado`` on PATH), so the
package imports and behaves cleanly on machines without a Vivado install.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .. import features as feat
from .. import rtl_transform
from ..devices import get as _dev_get
from .base import Backend, Design, FlowOptions, RunMetrics, RunResult

_CLK_RE = re.compile(r".*(clk|clock).*", re.IGNORECASE)


class VivadoBackend(Backend):
    name = "vivado"

    def __init__(
        self,
        vivado: str = "vivado",
        timeout_s: int = 3600,
        emit_timing_artifacts: bool = False,
        clock_port: str | None = None,
    ) -> None:
        self.vivado = vivado
        self.timeout_s = timeout_s
        self.emit_timing_artifacts = emit_timing_artifacts
        self.clock_port = clock_port

    def is_available(self) -> bool:
        return shutil.which(self.vivado) is not None

    def run(self, design: Design, options: FlowOptions, workdir: Path) -> RunResult:
        workdir.mkdir(parents=True, exist_ok=True)
        workdir = workdir.resolve()
        result = RunResult(
            design_id=design.design_id(),
            options=options,
            metrics=RunMetrics(target_freq_mhz=design.target_freq_mhz),
            backend=self.name,
            workdir=str(workdir),
        )
        dev = _dev_get(design.target)
        if dev is None or dev.backend != "vivado":
            result.error = f"target {design.target!r} is not an AMD/Vivado device"
            return result
        if not self.is_available():
            result.error = "Vivado flow requires 'vivado' on PATH"
            return result

        rtl_files = rtl_transform.prepare_rtl(design, options, workdir)
        util_rpt = workdir / "utilization.rpt"
        timing_rpt = workdir / "timing.rpt"
        power_rpt = workdir / "power.rpt"
        sdf_path = workdir / "timing.sdf"
        netlist_v = workdir / "netlist_funcsim.v"

        tcl = self._render_tcl(design, options, rtl_files, workdir,
                               util_rpt, timing_rpt, power_rpt, sdf_path, netlist_v)
        tcl_path = workdir / "impl.tcl"
        tcl_path.write_text(tcl)

        log, ok = self._run_cmd(
            [self.vivado, "-mode", "batch", "-source", str(tcl_path),
             "-nojournal", "-nolog"],
            workdir,
        )
        result.log += log

        util = util_rpt.read_text() if util_rpt.exists() else ""
        timing = timing_rpt.read_text() if timing_rpt.exists() else ""
        result.metrics = build_vivado_metrics(
            utilization=util, timing_summary=timing,
            target_period_ns=design.clock_ns,
            target_freq_mhz=design.target_freq_mhz,
        )
        if self.emit_timing_artifacts:
            if sdf_path.exists():
                result.sdf_path = str(sdf_path)
            if netlist_v.exists():
                result.routed_netlist_path = str(netlist_v)

        try:
            # Vendor synthesis does not emit a yosys JSON, so RTL-side features
            # come from the source (used for the predictor's cold start).
            result.features = feat.from_rtl_files(design.rtl_files)
        except Exception:  # noqa: BLE001
            result.features = {}

        if not result.metrics.routed_ok:
            result.error = result.error or "Vivado implementation did not route"
            result.success = False
            return result
        bit = workdir / "out.bit"
        if bit.exists():
            result.bitstream_path = str(bit)
        result.success = True
        return result

    # ------------------------------------------------------------------ #
    def _clock_constraint(self, design: Design) -> str:
        period = design.clock_ns if design.clock_ns > 0 else 10.0
        if self.clock_port:
            sel = f"[get_ports {{{self.clock_port}}}]"
        else:
            # Best-effort: constrain any port that looks like a clock.
            sel = "[get_ports -quiet -filter {NAME =~ *clk* || NAME =~ *clock*}]"
        return (
            f"if {{[llength {sel}] > 0}} {{\n"
            f"  create_clock -name sysclk -period {period:.3f} {sel}\n"
            f"}}"
        )

    def _render_tcl(self, design, options, rtl_files, workdir, util_rpt,
                    timing_rpt, power_rpt, sdf_path, netlist_v) -> str:
        dev = _dev_get(design.target)
        part = dev.part if dev and dev.part else design.target
        reads = "\n".join(f"read_verilog {{{f}}}" for f in rtl_files)
        directive = "-directive Explore" if options.retime else ""
        emit_artifacts = (
            f"write_verilog -mode funcsim -force {{{netlist_v}}}\n"
            f"write_sdf -force {{{sdf_path}}}\n"
            if self.emit_timing_artifacts else ""
        )
        return (
            f"{reads}\n"
            f"synth_design -top {design.top} -part {part} {directive}\n"
            f"{self._clock_constraint(design)}\n"
            f"opt_design\n"
            f"place_design\n"
            f"route_design\n"
            f"report_utilization -file {{{util_rpt}}}\n"
            f"report_timing_summary -file {{{timing_rpt}}}\n"
            f"report_power -file {{{power_rpt}}}\n"
            f"{emit_artifacts}"
            f"write_bitstream -force {{{workdir / 'out.bit'}}}\n"
        )

    def _run_cmd(self, cmd: list[str], workdir: Path):
        try:
            proc = subprocess.run(
                cmd, cwd=str(workdir), capture_output=True, text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            return f"$ {' '.join(cmd)}\n[not found] {exc}\n", False
        except subprocess.TimeoutExpired:
            return f"$ {' '.join(cmd)}\n[timeout after {self.timeout_s}s]\n", False
        log = (
            f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}\n"
            f"[exit {proc.returncode}]\n"
        )
        return log, proc.returncode == 0


# --------------------------------------------------------------------------- #
# Report parsers (pure functions -- unit-tested without a Vivado install).
# --------------------------------------------------------------------------- #
def _util_row(text: str, *names: str) -> int:
    """Pull the 'Used' column for the first matching row in a Vivado table."""
    for name in names:
        # e.g. "| Slice LUTs              |  123 | ... "
        m = re.search(
            rf"^\|\s*{re.escape(name)}\s*\**\s*\|\s*(\d+)\s*\|",
            text, re.MULTILINE,
        )
        if m:
            return int(m.group(1))
    return 0


def parse_utilization(text: str) -> dict[str, int]:
    """Parse LUT/FF/BRAM/DSP counts from a ``report_utilization`` file."""
    return {
        "luts": _util_row(text, "Slice LUTs", "CLB LUTs"),
        "ffs": _util_row(text, "Slice Registers", "CLB Registers",
                         "Register as Flip Flop"),
        "bram": _util_row(text, "Block RAM Tile", "RAMB36/FIFO*", "Block RAM Tile "),
        "dsp": _util_row(text, "DSPs", "DSP48E1", "DSP48E2"),
    }


def parse_wns_ns(text: str) -> float | None:
    """Worst negative slack (ns) from a ``report_timing_summary`` file.

    Vivado prints a 'Design Timing Summary' table whose first numeric column is
    WNS(ns). Positive = timing met with margin; negative = failing paths.
    """
    m = re.search(r"WNS\s*\(ns\)", text)
    if not m:
        return None
    # The values row is the first line after the header/separators that carries
    # a float; its first float is WNS. Works for both the space-aligned and
    # pipe-delimited variants Vivado emits.
    for line in text[m.end():].splitlines():
        stripped = line.replace("|", " ").strip()
        if not stripped or set(stripped) <= set("-+ "):
            continue  # blank or separator row
        vm = re.search(r"(-?\d+\.\d+)", stripped)
        if vm:
            return float(vm.group(1))
    return None


def parse_total_power_w(text: str) -> float | None:
    m = re.search(r"Total On-Chip Power\s*\(W\)\s*\|\s*(\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def build_vivado_metrics(
    utilization: str,
    timing_summary: str,
    target_period_ns: float,
    target_freq_mhz: float,
) -> RunMetrics:
    u = parse_utilization(utilization)
    wns = parse_wns_ns(timing_summary)
    routed = bool(utilization) and bool(timing_summary) and wns is not None
    fmax = 0.0
    crit = 0.0
    if wns is not None and target_period_ns > 0:
        achieved_period = max(target_period_ns - wns, 1e-3)
        fmax = 1000.0 / achieved_period
        crit = achieved_period
    return RunMetrics(
        fmax_mhz=fmax,
        target_freq_mhz=target_freq_mhz,
        crit_path_ns=crit,
        luts=u["luts"],
        ffs=u["ffs"],
        bram=u["bram"],
        dsp=u["dsp"],
        carries=0,
        routed_ok=routed,
    )
