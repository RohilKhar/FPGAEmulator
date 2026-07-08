"""Ice40Backend: the real open-source iCE40 flow.

Pipeline: yosys (synth_ice40) -> nextpnr-ice40 (place, route, timing) -> icepack.
Every step's stdout/stderr is captured; on any failure we return a RunResult
with success=False rather than raising, so the optimizer can keep going.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .. import features as feat
from .. import reports
from .. import rtl_transform
from .base import Backend, Design, FlowOptions, RunMetrics, RunResult

from ..devices import by_backend

# Device tables are derived from the central registry (fpgaforge/devices.py).
# target -> (nextpnr device flag, default package)
_DEVICES: dict[str, tuple[str, str]] = {
    d.target: (d.pnr_flag, d.package) for d in by_backend("ice40")
}

# Only the UltraPlus family has hardened DSP (SB_MAC16) blocks. On other
# families a "-dsp" mapping produces cells nextpnr cannot place, so the DSP
# knob is silently ignored there.
_DSP_TARGETS: frozenset[str] = frozenset(
    d.target for d in by_backend("ice40") if d.has_dsp
)


class Ice40Backend(Backend):
    name = "ice40"

    def __init__(
        self,
        yosys: str = "yosys",
        nextpnr: str = "nextpnr-ice40",
        icepack: str = "icepack",
        timeout_s: int = 600,
        emit_timing_artifacts: bool = False,
    ) -> None:
        self.yosys = yosys
        self.nextpnr = nextpnr
        self.icepack = icepack
        self.timeout_s = timeout_s
        # When True, ask nextpnr to also write an SDF delay file and the routed
        # netlist (for delay-annotated simulation / timing sign-off).
        self.emit_timing_artifacts = emit_timing_artifacts

    def is_available(self) -> bool:
        return shutil.which(self.yosys) is not None and (
            shutil.which(self.nextpnr) is not None
        )

    def run(self, design: Design, options: FlowOptions, workdir: Path) -> RunResult:
        workdir.mkdir(parents=True, exist_ok=True)
        # Commands run with cwd=workdir, so all derived paths must be absolute.
        workdir = workdir.resolve()
        result = RunResult(
            design_id=design.design_id(),
            options=options,
            metrics=RunMetrics(target_freq_mhz=design.target_freq_mhz),
            backend=self.name,
            workdir=str(workdir),
        )

        if design.target not in _DEVICES:
            result.error = f"unsupported ice40 target: {design.target}"
            return result
        device_flag, package = _DEVICES[design.target]

        rtl_files = rtl_transform.prepare_rtl(design, options, workdir)
        netlist_path = workdir / "netlist.json"
        asc_path = workdir / "out.asc"
        bin_path = workdir / "out.bin"
        sdf_path = workdir / "timing.sdf"
        routed_path = workdir / "routed.json"

        # ---- Synthesis (yosys) ----
        use_dsp = options.dsp and design.target in _DSP_TARGETS
        synth_log, netlist, ok = self._run_yosys(
            rtl_files, design.top, options, netlist_path, workdir, use_dsp
        )
        result.log += synth_log
        if not ok or netlist is None:
            result.error = "synthesis failed"
            return result

        result.features = feat.from_yosys_json(netlist, design.top)
        yosys_counts = reports.parse_yosys_stat_text(synth_log)

        # ---- Place & route + timing (nextpnr) ----
        # nextpnr exits non-zero when a design routes but misses the timing
        # target; that is a valid outcome we want to record, so success is based
        # on whether the design actually routed, not on the exit code.
        pnr_log, _pnr_exit_ok = self._run_nextpnr(
            netlist_path, asc_path, device_flag, package, design, options, workdir,
            sdf_path if self.emit_timing_artifacts else None,
            routed_path if self.emit_timing_artifacts else None,
        )
        result.log += "\n" + pnr_log
        if self.emit_timing_artifacts:
            if sdf_path.exists():
                result.sdf_path = str(sdf_path)
            if routed_path.exists():
                result.routed_netlist_path = str(routed_path)
        result.metrics = reports.build_metrics(
            nextpnr_log=pnr_log,
            yosys_stat=yosys_counts,
            target_freq_mhz=design.target_freq_mhz,
        )

        if not result.metrics.routed_ok:
            result.error = "place-and-route failed"
            result.success = False
            return result

        # ---- Bitstream (icepack, optional) ----
        if shutil.which(self.icepack) and asc_path.exists():
            pack_log, pack_ok = self._run_cmd(
                [self.icepack, str(asc_path), str(bin_path)], workdir
            )
            result.log += "\n" + pack_log
            if pack_ok and bin_path.exists():
                result.bitstream_path = str(bin_path)

        result.success = True
        return result

    # ------------------------------------------------------------------ #

    def _run_yosys(
        self,
        rtl_files: list[str],
        top: str,
        options: FlowOptions,
        netlist_path: Path,
        workdir: Path,
        use_dsp: bool,
    ):
        synth_flags = []
        if use_dsp:
            synth_flags.append("-dsp")
        # synth_ice40 enables abc9 by default; -retime is incompatible with it,
        # so retiming requires -noabc9. retime and abc9 are mutually exclusive.
        if options.retime:
            synth_flags += ["-retime", "-noabc9"]
        elif options.abc9:
            synth_flags.append("-abc9")
        synth_flags_str = " ".join(synth_flags)

        read = "\n".join(f"read_verilog {f}" for f in rtl_files)
        script = (
            f"{read}\n"
            f"synth_ice40 -top {top} {synth_flags_str} "
            f"-json {netlist_path}\n"
            f"stat\n"
        )
        script_path = workdir / "synth.ys"
        script_path.write_text(script)

        log, ok = self._run_cmd([self.yosys, "-q", "-s", str(script_path)], workdir)
        netlist = None
        if netlist_path.exists():
            try:
                netlist = json.loads(netlist_path.read_text())
            except json.JSONDecodeError:
                netlist = None
        return log, netlist, ok and netlist is not None

    def _run_nextpnr(
        self,
        netlist_path: Path,
        asc_path: Path,
        device_flag: str,
        package: str,
        design: Design,
        options: FlowOptions,
        workdir: Path,
        sdf_path: Path | None = None,
        routed_path: Path | None = None,
    ):
        cmd = [
            self.nextpnr,
            device_flag,
            "--package",
            package,
            "--json",
            str(netlist_path),
            "--asc",
            str(asc_path),
            "--freq",
            f"{design.target_freq_mhz:.3f}",
            "--seed",
            str(options.seed),
            "--placer",
            options.placer,
            "--pcf-allow-unconstrained",
        ]
        if design.pcf:
            cmd += ["--pcf", str(Path(design.pcf).resolve())]
        if sdf_path is not None:
            cmd += ["--sdf", str(sdf_path)]
        if routed_path is not None:
            cmd += ["--write", str(routed_path)]
        return self._run_cmd(cmd, workdir)

    def _run_cmd(self, cmd: list[str], workdir: Path):
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            return f"$ {' '.join(cmd)}\n[not found] {exc}\n", False
        except subprocess.TimeoutExpired:
            return f"$ {' '.join(cmd)}\n[timeout after {self.timeout_s}s]\n", False
        log = (
            f"$ {' '.join(cmd)}\n"
            f"{proc.stdout}\n{proc.stderr}\n"
            f"[exit {proc.returncode}]\n"
        )
        return log, proc.returncode == 0
