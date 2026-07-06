"""Ecp5Backend: the open-source Lattice ECP5 flow.

Pipeline: yosys (synth_ecp5) -> nextpnr-ecp5 (place, route, timing) -> ecppack.
Mirrors :class:`Ice40Backend` so the whole stack (emulate/verify/prove/physics/
reward) works on a second, larger architecture -- proving the engine is not
iCE40-specific. ECP5 brings bigger devices, more BRAM, and hardened DSP, which
also unblocks memory-heavy designs.

Like the iCE40 backend, every step is captured and failures return a RunResult
with success=False rather than raising.
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

# device target -> (nextpnr size flag, default package)
_DEVICES: dict[str, tuple[str, str]] = {
    "ecp5_12k": ("--12k", "CABGA256"),
    "ecp5_25k": ("--25k", "CABGA256"),
    "ecp5_45k": ("--45k", "CABGA381"),
    "ecp5_85k": ("--85k", "CABGA381"),
}


class Ecp5Backend(Backend):
    name = "ecp5"

    def __init__(
        self,
        yosys: str = "yosys",
        nextpnr: str = "nextpnr-ecp5",
        ecppack: str = "ecppack",
        timeout_s: int = 900,
        emit_timing_artifacts: bool = False,
    ) -> None:
        self.yosys = yosys
        self.nextpnr = nextpnr
        self.ecppack = ecppack
        self.timeout_s = timeout_s
        self.emit_timing_artifacts = emit_timing_artifacts

    def is_available(self) -> bool:
        return shutil.which(self.yosys) is not None and (
            shutil.which(self.nextpnr) is not None
        )

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

        if design.target not in _DEVICES:
            result.error = f"unsupported ecp5 target: {design.target}"
            return result
        size_flag, package = _DEVICES[design.target]

        rtl_files = rtl_transform.prepare_rtl(design, options, workdir)
        netlist_path = workdir / "netlist.json"
        config_path = workdir / "out.config"
        bit_path = workdir / "out.bit"
        sdf_path = workdir / "timing.sdf"
        routed_path = workdir / "routed.json"

        # ---- Synthesis (yosys) ----
        synth_log, netlist, ok = self._run_yosys(
            rtl_files, design.top, options, netlist_path, workdir
        )
        result.log += synth_log
        if not ok or netlist is None:
            result.error = "synthesis failed"
            return result

        result.features = feat.from_yosys_json(netlist, design.top)
        yosys_counts = reports.parse_yosys_stat_text(synth_log)

        # ---- Place & route + timing (nextpnr) ----
        pnr_log, _ = self._run_nextpnr(
            netlist_path, config_path, size_flag, package, design, options, workdir,
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

        if not result.metrics.routed_ok and not config_path.exists():
            result.error = "place-and-route failed"
            result.success = False
            return result

        # ---- Bitstream (ecppack, optional) ----
        if shutil.which(self.ecppack) and config_path.exists():
            pack_log, pack_ok = self._run_cmd(
                [self.ecppack, str(config_path), str(bit_path)], workdir
            )
            result.log += "\n" + pack_log
            if pack_ok and bit_path.exists():
                result.bitstream_path = str(bit_path)

        result.success = result.metrics.routed_ok or config_path.exists()
        return result

    # ------------------------------------------------------------------ #
    def _run_yosys(self, rtl_files, top, options, netlist_path, workdir):
        synth_flags = []
        if options.retime:
            synth_flags.append("-retime")
        if not options.dsp:
            synth_flags.append("-nodsp")
        synth_flags_str = " ".join(synth_flags)

        read = "\n".join(f"read_verilog {f}" for f in rtl_files)
        script = (
            f"{read}\n"
            f"synth_ecp5 -top {top} {synth_flags_str} -json {netlist_path}\n"
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

    def _run_nextpnr(self, netlist_path, config_path, size_flag, package, design,
                     options, workdir, sdf_path=None, routed_path=None):
        cmd = [
            self.nextpnr,
            size_flag,
            "--package",
            package,
            "--json",
            str(netlist_path),
            "--textcfg",
            str(config_path),
            "--freq",
            f"{design.target_freq_mhz:.3f}",
            "--seed",
            str(options.seed),
            "--placer",
            options.placer,
            "--lpf-allow-unconstrained",
        ]
        if sdf_path is not None:
            cmd += ["--sdf", str(sdf_path)]
        if routed_path is not None:
            cmd += ["--write", str(routed_path)]
        return self._run_cmd(cmd, workdir)

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
            f"$ {' '.join(cmd)}\n"
            f"{proc.stdout}\n{proc.stderr}\n"
            f"[exit {proc.returncode}]\n"
        )
        return log, proc.returncode == 0
