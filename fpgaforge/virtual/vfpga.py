"""VirtualFPGA: synthesize RTL to FPGA primitives and run it in a virtual fabric.

Flow: Yosys (synth_ice40 -> gate-level netlist of SB_* cells) -> Icarus Verilog
(compile the mapped netlist + Yosys sim models + a virtual-board harness) ->
vvp (run). The result reports whether the implemented design synthesized,
compiled, and came up, plus a VCD waveform and sampled outputs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .. import rtl_transform
from ..backends.base import Design, FlowOptions
from .board import BringUpConfig, Port, render_testbench


@dataclass
class BringUpResult:
    design_id: str
    synthesized: bool = False
    compiled: bool = False
    ran: bool = False
    success: bool = False
    timed_out: bool = False
    cycles: int = 0
    ports: list[Port] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    vcd_path: str | None = None
    sim_output: str = ""
    workdir: str | None = None
    log: str = ""
    error: str | None = None
    # Optional timing sign-off (populated when bring-up is run with timing=True).
    timing: "object | None" = None

    def diagnostics(self, limit: int = 25):
        """Structured errors/warnings parsed from the captured tool log."""
        from ..diagnostics import extract

        return extract(self.log, limit)

    def errors(self, limit: int = 25):
        return [d for d in self.diagnostics(limit) if d.severity == "error"]

    def summary(self) -> str:
        status = "UP" if self.success else "DOWN"
        lines = [
            f"virtual bring-up: {status}",
            f"design   : {self.design_id}",
            f"stages   : synth={self.synthesized} compile={self.compiled} run={self.ran}",
        ]
        if self.timed_out:
            lines.append("warning  : simulation hit the watchdog timeout")
        if self.outputs:
            lines.append("outputs  :")
            for k, v in self.outputs.items():
                lines.append(f"           {k} = {v}")
        if self.vcd_path:
            lines.append(f"waveform : {self.vcd_path}")
        if self.timing is not None:
            t = self.timing
            verdict = "MET" if t.meets_timing else "VIOLATED"
            lines.append(
                f"timing   : {verdict} - {t.fmax_mhz:.1f} MHz vs {t.target_mhz:.1f} MHz "
                f"target (slack {t.slack_ns:+.2f} ns)"
            )
            if t.worst_path:
                wp = t.worst_path
                lines.append(
                    f"crit path: {wp.total_ns:.2f} ns "
                    f"({wp.logic_ns:.2f} logic / {wp.routing_ns:.2f} routing)"
                )
            if t.sdf_path:
                lines.append(f"sdf      : {t.sdf_path}")
        if self.error:
            lines.append(f"error    : {self.error}")
        diags = self.errors()
        if diags:
            lines.append("tool errors:")
            for d in diags:
                lines.append(f"  {d.format()}")
        return "\n".join(lines)


def _ports_from_netlist(netlist: dict, top: str | None) -> list[Port]:
    modules = netlist.get("modules", {})
    if not modules:
        return []
    if top and top in modules:
        module = modules[top]
    else:
        module = max(modules.values(), key=lambda m: len(m.get("ports", {})), default={})
    ports: list[Port] = []
    for name, spec in module.get("ports", {}).items():
        bits = spec.get("bits", [])
        ports.append(
            Port(name=name, direction=spec.get("direction", "input"), width=len(bits) or 1)
        )
    return ports


class VirtualFPGA:
    """Cycle-accurate virtual FPGA built on Yosys + Icarus Verilog."""

    def __init__(
        self,
        yosys: str = "yosys",
        iverilog: str = "iverilog",
        vvp: str = "vvp",
        timeout_s: int = 300,
    ) -> None:
        self.yosys = yosys
        self.iverilog = iverilog
        self.vvp = vvp
        self.timeout_s = timeout_s

    def is_available(self) -> bool:
        return all(
            shutil.which(t) is not None for t in (self.yosys, self.iverilog, self.vvp)
        )

    def cells_sim_path(self) -> Path | None:
        """Locate Yosys's iCE40 simulation model library."""
        try:
            out = subprocess.run(
                [f"{self.yosys}-config", "--datdir"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if out.returncode == 0:
                cand = Path(out.stdout.strip()) / "ice40" / "cells_sim.v"
                if cand.exists():
                    return cand
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Fallback: derive from the yosys binary location.
        exe = shutil.which(self.yosys)
        if exe:
            cand = Path(exe).resolve().parent.parent / "share" / "yosys" / "ice40" / "cells_sim.v"
            if cand.exists():
                return cand
        return None

    def bringup(
        self,
        design: Design,
        workdir: Path,
        config: BringUpConfig | None = None,
        testbench: str | Path | None = None,
    ) -> BringUpResult:
        cfg = config or BringUpConfig()
        workdir.mkdir(parents=True, exist_ok=True)
        workdir = workdir.resolve()
        result = BringUpResult(design_id=design.design_id(), workdir=str(workdir))

        if not self.is_available():
            result.error = "virtual FPGA requires yosys, iverilog, and vvp on PATH"
            return result

        cells_sim = self.cells_sim_path()
        if cells_sim is None:
            result.error = "could not locate Yosys ice40 cells_sim.v"
            return result

        # ---- Synthesis to a gate-level netlist of FPGA primitives ----
        rtl_files = rtl_transform.prepare_rtl(design, FlowOptions(), workdir)
        netlist_v = workdir / "mapped.v"
        netlist_json = workdir / "mapped.json"
        synth_ok, log = self._synth(rtl_files, design.top, netlist_v, netlist_json, workdir)
        result.log += log
        if not synth_ok:
            result.error = "synthesis failed"
            return result
        result.synthesized = True

        try:
            netlist = json.loads(netlist_json.read_text())
        except (OSError, json.JSONDecodeError):
            netlist = {}
        result.ports = _ports_from_netlist(netlist, design.top)

        # ---- Testbench (virtual board harness) ----
        vcd_path = workdir / "bringup.vcd"
        cfg.vcd_path = str(vcd_path)
        tb_path = workdir / "tb.v"
        if testbench is not None:
            tb_src = Path(testbench)
            if not tb_src.exists():
                result.error = f"testbench not found: {testbench}"
                return result
            tb_path.write_text(tb_src.read_text())
        else:
            try:
                tb_path.write_text(render_testbench(design.top, result.ports, cfg))
            except ValueError as exc:
                result.error = str(exc)
                return result

        # ---- Compile (iverilog) ----
        sim_vvp = workdir / "sim.vvp"
        compile_ok, log = self._run(
            [
                self.iverilog, "-g2012", "-o", str(sim_vvp),
                str(netlist_v), str(cells_sim), str(tb_path),
            ],
            workdir,
        )
        result.log += "\n" + log
        if not compile_ok or not sim_vvp.exists():
            result.error = "compilation failed"
            return result
        result.compiled = True

        # ---- Run (vvp) ----
        run_ok, log = self._run([self.vvp, str(sim_vvp)], workdir)
        result.log += "\n" + log
        result.sim_output = log
        result.ran = run_ok or "VFPGA_DONE" in log or "VFPGA_TIMEOUT" in log
        if vcd_path.exists():
            result.vcd_path = str(vcd_path)

        result.timed_out = "VFPGA_TIMEOUT" in log
        result.cycles = cfg.cycles
        result.outputs = _parse_outputs(log)
        checks_failed = any(
            tok in log for tok in ("VFPGA_FAIL", "ASSERTION FAILED", "Error:")
        )
        result.success = (
            result.compiled
            and result.ran
            and not result.timed_out
            and not checks_failed
            and "VFPGA_DONE" in log
        )
        if not result.success and result.error is None:
            if result.timed_out:
                result.error = "simulation timed out (possible hang / no progress)"
            elif checks_failed:
                result.error = "self-checks failed during simulation"
            elif "VFPGA_DONE" not in log:
                result.error = "simulation did not complete"
        return result

    # ------------------------------------------------------------------ #
    def _synth(self, rtl_files, top, netlist_v, netlist_json, workdir):
        read = "\n".join(f"read_verilog {f}" for f in rtl_files)
        script = (
            f"{read}\n"
            f"synth_ice40 -top {top}\n"
            f"write_json {netlist_json}\n"
            f"write_verilog -noattr {netlist_v}\n"
        )
        script_path = workdir / "bringup_synth.ys"
        script_path.write_text(script)
        return self._run([self.yosys, "-q", "-s", str(script_path)], workdir)

    def _run(self, cmd: list[str], workdir: Path):
        try:
            proc = subprocess.run(
                cmd, cwd=str(workdir), capture_output=True, text=True,
                timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            return False, f"$ {' '.join(cmd)}\n[not found] {exc}\n"
        except subprocess.TimeoutExpired:
            return False, f"$ {' '.join(cmd)}\n[timeout after {self.timeout_s}s]\n"
        log = (
            f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}\n[exit {proc.returncode}]\n"
        )
        return proc.returncode == 0, log


def _parse_outputs(sim_output: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for line in sim_output.splitlines():
        line = line.strip()
        if line.startswith("VFPGA_OUT "):
            body = line[len("VFPGA_OUT "):]
            if "=" in body:
                name, _, val = body.partition("=")
                outputs[name.strip()] = val.strip()
    return outputs


def bringup(
    rtl: str | Sequence[str],
    top: str,
    target_fpga: str = "ice40_up5k",
    cycles: int = 64,
    clock: str | None = None,
    reset: str | None = None,
    reset_active_high: bool | None = None,
    testbench: str | Path | None = None,
    config: BringUpConfig | None = None,
    workdir: str | Path = ".runs/bringup",
    engine: VirtualFPGA | None = None,
    timing: bool = False,
    clock_ns: float = 10.0,
) -> BringUpResult:
    """Virtually bring up a design: synthesize to primitives and simulate.

    Args:
        rtl: RTL file path(s) (or inline RTL).
        top: top module name.
        target_fpga: device family used for primitive mapping.
        cycles: number of clock cycles to run after reset.
        clock/reset: override auto-detected clock/reset port names.
        reset_active_high: override reset polarity inference.
        testbench: optional custom testbench (with your own self-checks).
        config: full BringUpConfig (overrides the convenience args above).
        timing: also run a real timing sign-off (place & route + STA); the
            result's `success` then requires both functional up AND met timing.
        clock_ns: target clock period for the timing sign-off.
    """
    rtl_files = [rtl] if isinstance(rtl, str) else list(rtl)
    design = Design(rtl_files=tuple(rtl_files), top=top, target=target_fpga)
    cfg = config or BringUpConfig(
        cycles=cycles, clock=clock, reset=reset, reset_active_high=reset_active_high
    )
    engine = engine or VirtualFPGA()
    result = engine.bringup(design, Path(workdir), cfg, testbench=testbench)

    if timing:
        from ..timing import signoff

        report = signoff(
            rtl=rtl_files, top=top, target_fpga=target_fpga, clock_ns=clock_ns,
            workdir=Path(workdir) / "timing",
        )
        result.timing = report
        # Full sign-off: must both come up functionally and meet timing.
        result.success = result.success and report.meets_timing
        if not report.meets_timing and result.error is None:
            result.error = report.error or "timing not met at target clock"
    return result
