"""QuartusBackend: the Intel/Altera vendor flow (Quartus Prime command line).

Intel parts (Cyclone, MAX 10, Arria, Stratix, Agilex) have no open bitstream
tools, so we drive Quartus' command-line executables:
``quartus_map (synthesis) -> quartus_fit (place & route) -> quartus_sta (timing)
-> quartus_pow (power) -> quartus_asm (bitstream)`` against a generated project
(.qsf) + timing constraints (.sdc). We parse the ``.rpt`` files into the shared
:class:`RunMetrics`.

As with Vivado, the bitstream is proprietary, so there is no bit-level
reconstruction; the flow targets the *netlist-level* equivalence tier. Gated
behind :meth:`is_available` (``quartus_map`` on PATH).
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


class QuartusBackend(Backend):
    name = "quartus"

    def __init__(
        self,
        quartus_map: str = "quartus_map",
        quartus_fit: str = "quartus_fit",
        quartus_sta: str = "quartus_sta",
        quartus_pow: str = "quartus_pow",
        quartus_asm: str = "quartus_asm",
        timeout_s: int = 3600,
        emit_timing_artifacts: bool = False,
        clock_port: str | None = None,
    ) -> None:
        self.quartus_map = quartus_map
        self.quartus_fit = quartus_fit
        self.quartus_sta = quartus_sta
        self.quartus_pow = quartus_pow
        self.quartus_asm = quartus_asm
        self.timeout_s = timeout_s
        self.emit_timing_artifacts = emit_timing_artifacts
        self.clock_port = clock_port

    def is_available(self) -> bool:
        return shutil.which(self.quartus_map) is not None

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
        if dev is None or dev.backend != "quartus":
            result.error = f"target {design.target!r} is not an Intel/Quartus device"
            return result
        if not self.is_available():
            result.error = "Quartus flow requires 'quartus_map' on PATH"
            return result

        rtl_files = rtl_transform.prepare_rtl(design, options, workdir)
        proj = "proj"
        (workdir / f"{proj}.qsf").write_text(
            self._render_qsf(design, rtl_files)
        )
        (workdir / f"{proj}.sdc").write_text(self._render_sdc(design))

        steps = [
            [self.quartus_map, proj],
            [self.quartus_fit, proj],
            [self.quartus_sta, proj],
            [self.quartus_pow, proj],
        ]
        if self.emit_timing_artifacts:
            steps.append([self.quartus_asm, proj])
        for cmd in steps:
            log, ok = self._run_cmd(cmd, workdir)
            result.log += log
            if not ok:
                break

        fit = _read(workdir / f"{proj}.fit.rpt") or _read(workdir / f"{proj}.map.rpt")
        sta = _read(workdir / f"{proj}.sta.rpt")
        result.metrics = build_quartus_metrics(
            fit_report=fit, sta_report=sta,
            target_freq_mhz=design.target_freq_mhz,
        )
        try:
            result.features = feat.from_rtl_files(design.rtl_files)
        except Exception:  # noqa: BLE001
            result.features = {}

        if not result.metrics.routed_ok:
            result.error = result.error or "Quartus fit did not complete"
            result.success = False
            return result
        sof = workdir / f"{proj}.sof"
        if sof.exists():
            result.bitstream_path = str(sof)
        result.success = True
        return result

    # ------------------------------------------------------------------ #
    def _render_qsf(self, design: Design, rtl_files) -> str:
        dev = _dev_get(design.target)
        part = dev.part if dev and dev.part else design.target
        lines = [
            f"set_global_assignment -name FAMILY \"{dev.family if dev else ''}\"",
            f"set_global_assignment -name DEVICE {part}",
            f"set_global_assignment -name TOP_LEVEL_ENTITY {design.top}",
            "set_global_assignment -name SDC_FILE proj.sdc",
        ]
        for f in rtl_files:
            lines.append(f"set_global_assignment -name VERILOG_FILE {f}")
        return "\n".join(lines) + "\n"

    def _render_sdc(self, design: Design) -> str:
        period = design.clock_ns if design.clock_ns > 0 else 10.0
        port = self.clock_port or "clk"
        return (
            f"create_clock -name sysclk -period {period:.3f} "
            f"[get_ports {{{port}}}]\n"
            "derive_clock_uncertainty\n"
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


def _read(p: Path) -> str:
    return p.read_text() if p.exists() else ""


# --------------------------------------------------------------------------- #
# Report parsers (pure functions -- unit-tested without a Quartus install).
# --------------------------------------------------------------------------- #
def _int(s: str) -> int:
    return int(re.sub(r"[,\s]", "", s))


def _fit_row(text: str, *names: str) -> int:
    for name in names:
        m = re.search(
            rf";\s*{re.escape(name)}\s*;\s*([\d,]+)", text, re.IGNORECASE
        )
        if m:
            return _int(m.group(1))
    return 0


def parse_fit_resources(text: str) -> dict[str, int]:
    """Resource usage from a ``quartus_fit``/``quartus_map`` report."""
    luts = _fit_row(text, "Total logic elements", "Logic utilization (in ALMs)",
                    "ALMs needed", "Combinational ALUTs")
    return {
        "luts": luts,
        "ffs": _fit_row(text, "Total registers", "Total dedicated logic registers"),
        "bram": _fit_row(text, "Total RAM Blocks", "Total block memory bits",
                         "M9Ks", "M10Ks"),
        "dsp": _fit_row(text, "Total DSP Blocks",
                        "DSP block 18-bit elements", "DSP Blocks"),
    }


def parse_sta_fmax_mhz(text: str) -> float | None:
    """Minimum restricted Fmax (MHz) across clocks from a ``quartus_sta`` report.

    The 'Fmax Summary' table rows look like:
      ; 123.45 MHz ; 120.0 MHz ; clk ; ;
    where the second column is the *restricted* Fmax (the number the device can
    actually be clocked at). We take the worst (min) across all clocks.
    """
    fmaxes: list[float] = []
    for m in re.finditer(
        r";\s*([\d.]+)\s*MHz\s*;\s*([\d.]+)\s*MHz\s*;", text
    ):
        fmaxes.append(float(m.group(2)))
    return min(fmaxes) if fmaxes else None


def parse_total_power_mw(text: str) -> float | None:
    m = re.search(
        r"Total Thermal Power Dissipation\s*;\s*([\d.]+)\s*mW", text
    )
    if m:
        return float(m.group(1))
    m = re.search(
        r"Total Thermal Power Dissipation\s*;\s*([\d.]+)\s*W", text
    )
    return float(m.group(1)) * 1000.0 if m else None


def build_quartus_metrics(
    fit_report: str, sta_report: str, target_freq_mhz: float
) -> RunMetrics:
    r = parse_fit_resources(fit_report)
    fmax = parse_sta_fmax_mhz(sta_report)
    routed = bool(fit_report) and any(r.values())
    return RunMetrics(
        fmax_mhz=fmax or 0.0,
        target_freq_mhz=target_freq_mhz,
        crit_path_ns=(1000.0 / fmax) if fmax else 0.0,
        luts=r["luts"],
        ffs=r["ffs"],
        bram=r["bram"],
        dsp=r["dsp"],
        carries=0,
        routed_ok=routed,
    )
