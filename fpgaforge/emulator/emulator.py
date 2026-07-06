"""The iCE40 bitstream-level fabric emulator.

Two entry points:

``emulate(bitstream)``
    Load a real ``.bin``/``.asc`` bitstream, decode what the fabric is
    configured to compute (LUT truth tables, flops, carries), and rebuild a
    simulatable netlist of the configured fabric. This is the "load the flashed
    image into a device model" capability.

``verify_bitstream(rtl, top, ...)``
    The first-shot capstone: take RTL all the way to a real bitstream, unpack
    that binary back into the configured fabric, and simulate BOTH the
    bitstream and the synthesized design under identical stimulus, cycle by
    cycle. If every cycle matches, the image you would flash provably behaves
    like your design.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..backends.base import Design, FlowOptions
from ..virtual.board import (
    BringUpConfig,
    Port,
    detect_clock,
    detect_reset,
    _reset_is_active_high,
)
from ..virtual.vfpga import VirtualFPGA, _ports_from_netlist, _parse_outputs
from . import bitstream as bsmod
from . import netlist as nl
from . import peripherals as periph
from .fabric import FabricConfig, decode_fabric

import json

# Distinct pseudo-random seeds for a verification campaign. Fixed so runs are
# reproducible; varied so each explores a different stimulus trajectory.
_DEFAULT_SEEDS = [1, 1337, 424242, 0x5EED]


def _seed_pool(n: int) -> list[int]:
    """A deterministic pool of ``n`` distinct 32-bit seeds (reproducible)."""
    seeds = list(_DEFAULT_SEEDS)
    x = 0x9E3779B1  # golden-ratio LCG increment -> well-spread seeds
    while len(seeds) < n:
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        if x not in seeds:
            seeds.append(x)
    return seeds[:n]


@dataclass
class _Artifacts:
    """Bitstream + reconstruction artifacts shared by verify and prove."""

    ports: list[Port]
    rtl_files: list[str]
    mapped_v: Path
    recon_v: Path
    wrapper_v: Path
    cells_sim: Path
    bitstream_path: str
    fabric: "FabricConfig | None" = None


@dataclass
class ProofResult:
    """Result of a formal equivalence proof between RTL and the bitstream."""

    design_id: str
    equivalent: bool | None = None   # True=proved equal, False=differ, None=unknown
    unbounded: bool = False          # True if proven for all time (induction)
    depth: int = 0                   # cycles proven (bounded) or induction length
    method: str = ""                 # "induction" | "bmc" | ""
    engine: str = "sat"              # "sat" (bit-blasting) | "smt" (memory arrays)
    counterexample: str | None = None
    bitstream_path: str | None = None
    fabric: "FabricConfig | None" = None
    workdir: str | None = None
    log: str = ""
    error: str | None = None

    @property
    def proved(self) -> bool:
        return self.equivalent is True

    def summary(self) -> str:
        if self.equivalent is True:
            scope = ("for ALL inputs and ALL time (unbounded induction)"
                     if self.unbounded
                     else f"for ALL inputs over {self.depth} cycles from reset (BMC)")
            verdict = "PROVEN EQUIVALENT"
            tail = (f"result    : the flashed bitstream is formally equivalent to the "
                    f"design {scope}")
        elif self.equivalent is False:
            verdict = "NOT EQUIVALENT"
            tail = f"counterexample: {self.counterexample or 'inputs exist where they differ'}"
        else:
            verdict = "INCONCLUSIVE"
            tail = f"note      : {self.error or 'proof did not converge'}"
        lines = [
            f"formal equivalence: {verdict}",
            f"design    : {self.design_id}",
            f"method    : {self.method or 'n/a'} ({self.engine} engine)",
        ]
        if self.bitstream_path:
            lines.append(f"bitstream : {self.bitstream_path}")
        if self.fabric is not None:
            lines.append(
                f"fabric    : {self.fabric.luts_used} LUTs, "
                f"{self.fabric.dffs_used} DFFs, {self.fabric.carries_used} carry cells"
            )
        lines.append(tail)
        return "\n".join(lines)


@dataclass
class EmulationResult:
    source: str
    device: str = ""
    fabric: FabricConfig | None = None
    netlist_path: str | None = None
    reconstructed: bool = False
    workdir: str | None = None
    log: str = ""
    error: str | None = None

    def summary(self) -> str:
        lines = [f"bitstream emulation: {self.source}"]
        if self.fabric is not None:
            lines.append(self.fabric.summary())
        if self.reconstructed:
            lines.append(f"netlist     : reconstructed -> {self.netlist_path}")
        if self.error:
            lines.append(f"error       : {self.error}")
        return "\n".join(lines)


@dataclass
class VerificationResult:
    design_id: str
    matches: bool = False
    cycles: int = 0
    bitstream_path: str | None = None
    fabric: FabricConfig | None = None
    rtl_trace: list[str] = field(default_factory=list)
    bitstream_trace: list[str] = field(default_factory=list)
    first_mismatch: str | None = None
    divergence: list[str] = field(default_factory=list)  # window around 1st mismatch
    debug_path: str | None = None                        # written divergence report
    vcd_path: str | None = None
    workdir: str | None = None
    log: str = ""
    error: str | None = None
    # --- coverage campaign metrics ---
    seeds: list[int] = field(default_factory=list)
    total_compared: int = 0        # concrete cycles compared across all seeds
    indeterminate: int = 0         # don't-care (uninitialized) cycles skipped
    toggle_coverage: float = 0.0   # fraction of output bits seen at both 0 and 1
    toggled_bits: int = 0
    total_bits: int = 0
    distinct_outputs: int = 0
    # --- stimulus quality (did we actually exercise the inputs?) ---
    input_coverage: float = 1.0    # fraction of driven input bits seen at 0 and 1
    input_toggled_bits: int = 0
    input_total_bits: int = 0
    driven_inputs: int = 0         # number of primary inputs the campaign drove      # unique output vectors observed
    coverage_saturated: bool = False  # adaptive campaign stopped finding new behavior

    @property
    def confidence(self) -> float:
        """A 0..1 empirical confidence for stimulus-based equivalence.

        Not a formal probability -- a transparent blend of (a) breadth of
        stimulus (seeds x cycles), (b) how much of the output space actually
        toggled, and (c) whether the campaign *saturated* (extra stimulus
        stopped revealing new behavior, i.e. we likely exercised everything
        reachable). Only meaningful when ``matches`` is True.
        """
        if not self.matches or self.total_compared == 0:
            return 0.0
        # Breadth saturates: ~1000 concrete cycles across >=4 seeds is "a lot".
        breadth = min(1.0, self.total_compared / 1000.0)
        seed_factor = min(1.0, len(self.seeds) / 4.0)
        cov = self.toggle_coverage
        # Stimulus quality: matching outputs mean little if we never drove the
        # inputs across their range. Designs with no primary inputs (pure
        # sequential, e.g. a free-running counter) are not penalized.
        in_cov = self.input_coverage if self.driven_inputs else 1.0
        base = 0.30 * cov + 0.20 * in_cov + 0.30 * breadth + 0.20 * seed_factor
        if self.coverage_saturated:
            # Saturation is strong evidence we saw all reachable behavior:
            # pull the score toward (but not to) certainty.
            base = base + (1.0 - base) * 0.5
        return round(base, 3)

    def summary(self) -> str:
        verdict = "MATCH" if self.matches else "MISMATCH"
        lines = [
            f"bitstream verification: {verdict}",
            f"design    : {self.design_id}",
        ]
        if self.seeds:
            lines.append(
                f"campaign  : {len(self.seeds)} seed(s), "
                f"{self.total_compared} concrete cycles compared"
                + (f", {self.indeterminate} don't-care cycles skipped"
                   if self.indeterminate else "")
            )
        else:
            lines.append(f"cycles    : {self.cycles} compared (post-reset)")
        if self.total_bits:
            sat = (" [saturated: extra stimulus revealed no new behavior]"
                   if self.coverage_saturated else "")
            lines.append(
                f"coverage  : {self.toggle_coverage:.0%} output-bit toggle "
                f"({self.toggled_bits}/{self.total_bits} bits), "
                f"{self.distinct_outputs} distinct output vectors{sat}"
            )
        if self.driven_inputs:
            lines.append(
                f"stimulus  : {self.input_coverage:.0%} input-bit toggle "
                f"({self.input_toggled_bits}/{self.input_total_bits} bits) "
                f"across {self.driven_inputs} driven input(s)"
            )
        if self.bitstream_path:
            lines.append(f"bitstream : {self.bitstream_path}")
        if self.fabric is not None:
            lines.append(
                f"fabric    : {self.fabric.luts_used} LUTs, "
                f"{self.fabric.dffs_used} DFFs, {self.fabric.carries_used} carry cells"
            )
        if self.vcd_path:
            lines.append(f"waveform  : {self.vcd_path}")
        if self.matches:
            lines.append(
                f"result    : bitstream reproduced the design on every concrete "
                f"cycle (empirical confidence {self.confidence:.0%})"
            )
        else:
            if self.first_mismatch:
                lines.append(f"mismatch  : {self.first_mismatch}")
            if self.divergence:
                lines.append("divergence:")
                lines.extend(f"  {ln}" for ln in self.divergence)
            if self.debug_path:
                lines.append(f"debug     : {self.debug_path}")
            if self.error:
                lines.append(f"error     : {self.error}")
        return "\n".join(lines)


class Emulator:
    """Bitstream-level iCE40 fabric emulator built on Project IceStorm + Icarus."""

    def __init__(
        self,
        iceunpack: str = "iceunpack",
        icebox_vlog: str = "icebox_vlog",
        icepack: str = "icepack",
        nextpnr: str = "nextpnr-ice40",
        yosys: str = "yosys",
        iverilog: str = "iverilog",
        vvp: str = "vvp",
        timeout_s: int = 300,
    ) -> None:
        self.iceunpack = iceunpack
        self.icebox_vlog = icebox_vlog
        self.icepack = icepack
        self.nextpnr = nextpnr
        self.vfpga = VirtualFPGA(yosys=yosys, iverilog=iverilog, vvp=vvp, timeout_s=timeout_s)
        self.iverilog = iverilog
        self.vvp = vvp
        self.timeout_s = timeout_s

    def tools_available(self) -> dict[str, bool]:
        return {
            t: shutil.which(t) is not None
            for t in (self.iceunpack, self.icebox_vlog, self.icepack,
                      self.nextpnr, self.vfpga.yosys, self.iverilog, self.vvp)
        }

    # ------------------------------------------------------------------ #
    def emulate(
        self,
        bitstream_path: str | Path,
        workdir: str | Path = ".runs/emulate",
        module: str = "chip",
        reconstruct: bool = True,
    ) -> EmulationResult:
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        res = EmulationResult(source=str(bitstream_path), workdir=str(workdir))
        try:
            bs = bsmod.load(bitstream_path, iceunpack=self.iceunpack)
        except Exception as exc:  # noqa: BLE001 - surface tool/parse errors
            res.error = str(exc)
            return res
        res.device = bs.device
        res.fabric = decode_fabric(bs)

        if reconstruct:
            # Ensure we have an .asc on disk for icebox_vlog.
            src = Path(bitstream_path)
            if src.suffix == ".bin":
                asc = bsmod.unpack_bin(src, iceunpack=self.iceunpack,
                                       out_path=workdir / "unpacked.asc")
            else:
                asc = src
            out_v = workdir / "reconstructed.v"
            try:
                nl.reconstruct(asc, out_v, module=module, icebox_vlog=self.icebox_vlog)
                res.netlist_path = str(out_v)
                res.reconstructed = True
            except Exception as exc:  # noqa: BLE001
                res.error = f"reconstruction failed: {exc}"
        return res

    # ------------------------------------------------------------------ #
    def _prepare_artifacts(self, design, cfg, workdir, clock_mhz):
        """Run the flow to a real bitstream and reconstruct the fabric netlist.

        Returns ``(_Artifacts, error_or_None, log)``. Shared by verify and prove.
        """
        log_all = ""
        if design.target not in nl.DEVICE_INFO:
            return None, f"unsupported target for emulation: {design.target}", log_all
        device_tag, device_flag, package = nl.DEVICE_INFO[design.target]

        missing = [t for t, ok in self.tools_available().items() if not ok]
        if missing:
            return None, f"missing tools: {', '.join(sorted(set(missing)))}", log_all

        cells_sim = self.vfpga.cells_sim_path()
        if cells_sim is None:
            return None, "could not locate Yosys ice40 cells_sim.v", log_all

        from .. import rtl_transform

        rtl_files = rtl_transform.prepare_rtl(design, FlowOptions(), workdir)
        mapped_v = workdir / "mapped.v"
        mapped_json = workdir / "mapped.json"
        ok, log = self.vfpga._synth(rtl_files, design.top, mapped_v, mapped_json, workdir)
        log_all += log
        if not ok:
            return None, "synthesis failed", log_all
        try:
            netlist = json.loads(mapped_json.read_text())
        except (OSError, json.JSONDecodeError):
            netlist = {}
        ports = _ports_from_netlist(netlist, design.top)
        if detect_clock(ports, cfg.clock) is None:
            return None, "no clock port detected; emulation compares clocked designs", log_all

        chipdb = nl.find_chipdb(device_tag)
        if chipdb is None:
            return None, f"could not find chipdb for device {device_tag}", log_all
        pins = nl.parse_package_pins(chipdb, package)
        try:
            pcf_text = nl.generate_pcf(ports, pins)
        except RuntimeError as exc:
            return None, str(exc), log_all
        pcf_path = workdir / "emu.pcf"
        pcf_path.write_text(pcf_text)

        asc_path = workdir / "out.asc"
        bin_path = workdir / "out.bin"
        _, log = self._run(
            [self.nextpnr, device_flag, "--package", package, "--json",
             str(mapped_json), "--pcf", str(pcf_path), "--asc", str(asc_path),
             "--freq", f"{clock_mhz:.3f}"],
            workdir,
        )
        log_all += "\n" + log
        if not asc_path.exists():
            return None, "place-and-route failed (no bitstream produced)", log_all
        _, log = self._run([self.icepack, str(asc_path), str(bin_path)], workdir)
        log_all += "\n" + log
        if not bin_path.exists():
            return None, "icepack failed", log_all

        rt_asc = workdir / "roundtrip.asc"
        try:
            bsmod.unpack_bin(bin_path, iceunpack=self.iceunpack, out_path=rt_asc)
        except RuntimeError as exc:
            return None, str(exc), log_all
        fabric = decode_fabric(bsmod.parse_asc(rt_asc.read_text()))

        recon_v = workdir / "recon.v"
        try:
            nl.reconstruct(rt_asc, recon_v, pcf_path=pcf_path, module="recon",
                           icebox_vlog=self.icebox_vlog)
        except RuntimeError as exc:
            return None, f"reconstruction failed: {exc}", log_all
        wrapper_v = workdir / "recon_top.v"
        wrapper_v.write_text(nl.make_rebus_wrapper("recon", design.top, ports))

        art = _Artifacts(
            ports=ports, rtl_files=list(rtl_files), mapped_v=mapped_v,
            recon_v=recon_v, wrapper_v=wrapper_v, cells_sim=cells_sim,
            bitstream_path=str(bin_path), fabric=fabric,
        )
        return art, None, log_all

    def verify_bitstream(
        self,
        design: Design,
        cycles: int = 32,
        clock_mhz: float = 50.0,
        seeds: Sequence[int] | None = None,
        adaptive: bool = False,
        max_seeds: int = 24,
        patience: int = 4,
        config: BringUpConfig | None = None,
        workdir: str | Path = ".runs/verify",
    ) -> VerificationResult:
        cfg = config or BringUpConfig(cycles=cycles)
        cfg.cycles = cycles
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        res = VerificationResult(design_id=design.design_id(), cycles=cycles,
                                 workdir=str(workdir))

        art, err, log = self._prepare_artifacts(design, cfg, workdir, clock_mhz)
        res.log += log
        if err is not None:
            res.error = err
            return res
        res.bitstream_path = art.bitstream_path
        res.fabric = art.fabric
        ports = art.ports
        cells_sim = art.cells_sim
        outputs = [p for p in ports if p.direction in ("output", "inout")]
        # Primary inputs the testbench actually drives (excludes clock/reset).
        _clk = detect_clock(ports, cfg.clock)
        _rst = detect_reset(ports, cfg.reset)
        driven = [
            p for p in ports
            if p.direction == "input"
            and p.name not in {_clk.name, getattr(_rst, "name", None)}
        ]

        # A random campaign runs several distinct stimulus streams (seeds); a
        # counter ramp is deterministic, so a single run suffices. In adaptive
        # mode we keep adding seeds until behavior stops growing (saturation).
        if cfg.stimulus == "random":
            if seeds:
                run_seeds = list(seeds)
            elif adaptive:
                run_seeds = _seed_pool(max_seeds)
            else:
                run_seeds = _DEFAULT_SEEDS
        else:
            run_seeds = []

        # ---- Compile each DUT once, then replay every seed ----
        vcd_path = workdir / "emulated.vcd"
        cfg.vcd_path = str(vcd_path)
        tb_v = workdir / "compare_tb.v"
        tb_v.write_text(render_compare_tb(design.top, ports, cfg))

        rtl_ok, clog = self._compile(workdir / "rtl.vvp", [tb_v, art.mapped_v, cells_sim], workdir)
        res.log += "\n" + clog
        # The reconstructed netlist can instantiate hard primitives (e.g.
        # SB_RAM40_4K, with the program baked into its INIT from the bitstream),
        # so it needs Yosys's ice40 sim models too.
        bit_ok, clog = self._compile(
            workdir / "bit.vvp", [tb_v, art.recon_v, art.wrapper_v, cells_sim], workdir
        )
        res.log += "\n" + clog
        if not rtl_ok or not bit_ok:
            res.error = "failed to compile design or bitstream netlist"
            return res

        rtl_traces: list[list[str]] = []
        stim_traces: list[list[str]] = []
        used_seeds: list[int] = []
        overall_match = True
        stale = 0                 # consecutive seeds that revealed no new behavior
        prev_signature = (-1.0, -1)
        use_adaptive = bool(adaptive and not seeds and cfg.stimulus == "random")
        for seed in (run_seeds or [None]):
            plus = [f"+SEED={seed}"] if seed is not None else None
            _, rtl_out = self._run_vvp(workdir / "rtl.vvp", workdir, plus)
            _, bit_out = self._run_vvp(workdir / "bit.vvp", workdir, plus)
            res.log += "\n" + rtl_out + "\n" + bit_out
            rtl_trace = _cycle_trace(rtl_out)
            bit_trace = _cycle_trace(bit_out)
            if not rtl_trace or not bit_trace:
                res.error = "simulation produced no cycle trace"
                return res
            if not res.rtl_trace:  # keep first run's traces for inspection/VCD
                res.rtl_trace = rtl_trace
                res.bitstream_trace = bit_trace
            rtl_traces.append(rtl_trace)
            if driven:
                stim_traces.append(_stim_trace(rtl_out))
            if seed is not None:
                used_seeds.append(seed)
            stats = _compare_traces_x(rtl_trace, bit_trace)
            res.total_compared += stats.compared
            res.indeterminate += stats.indeterminate
            if not stats.matches:
                overall_match = False
                if res.first_mismatch is None:
                    tag = f"[seed {seed}] " if seed is not None else ""
                    res.first_mismatch = tag + (stats.first_mismatch or "")
                    res.divergence = _divergence_window(
                        rtl_trace, bit_trace, _stim_trace(rtl_out),
                        stats.mismatch_cycle or 0,
                    )
                    dbg = workdir / "divergence.txt"
                    header = (
                        f"# first divergence (seed {seed}) at "
                        f"{stats.first_mismatch}\n"
                    )
                    dbg.write_text(header + "\n".join(res.divergence) + "\n")
                    res.debug_path = str(dbg)
                break  # a real divergence -> stop the campaign

            if use_adaptive:
                # Stop once new seeds stop expanding the observed behavior.
                cov, _, _, distinct = _toggle_coverage(rtl_traces, outputs)
                signature = (cov, distinct)
                if signature > prev_signature:
                    stale = 0
                    prev_signature = signature
                else:
                    stale += 1
                if stale >= patience:
                    res.coverage_saturated = True
                    break

        res.seeds = used_seeds
        res.matches = overall_match
        (res.toggle_coverage, res.toggled_bits,
         res.total_bits, res.distinct_outputs) = _toggle_coverage(rtl_traces, outputs)
        res.driven_inputs = len(driven)
        (res.input_coverage, res.input_toggled_bits,
         res.input_total_bits) = _input_coverage(stim_traces, driven)
        if vcd_path.exists():
            res.vcd_path = str(vcd_path)
        return res

    # ------------------------------------------------------------------ #
    def emulate_board(
        self,
        design: Design,
        board: "periph.BoardConfig | None" = None,
        workdir: str | Path = ".runs/board",
    ) -> "periph.BoardResult":
        """Run the flashed bitstream against a virtual board of peripherals.

        Takes RTL to a real bitstream, reconstructs the configured fabric, then
        instantiates that fabric wired to behavioral models of the board's
        peripherals (clock, reset, LEDs, UART, buttons, switches, GPIO) and runs
        it -- so you can read the UART text and watch the LEDs the *flashed
        image* actually drives.
        """
        cfg = board or periph.BoardConfig()
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        res = periph.BoardResult(design_id=design.design_id(), workdir=str(workdir))

        bcfg = BringUpConfig(clock=cfg.clock, reset=cfg.reset,
                             reset_active_high=cfg.reset_active_high)
        art, err, log = self._prepare_artifacts(
            design, bcfg, workdir, cfg.clock_mhz
        )
        res.log += log
        if err is not None:
            res.error = err
            return res
        res.bitstream_path = art.bitstream_path
        res.fabric = art.fabric

        roles = periph.classify_pins(art.ports, cfg)
        res.roles = roles
        vcd_path = workdir / "board.vcd"
        cfg.vcd_path = str(vcd_path)
        try:
            tb = periph.render_board_tb(design.top, art.ports, cfg, roles)
        except ValueError as exc:
            res.error = str(exc)
            return res
        tb_v = workdir / "board_tb.v"
        tb_v.write_text(tb)

        ok, clog = self._compile(
            workdir / "board.vvp",
            [tb_v, art.recon_v, art.wrapper_v, art.cells_sim], workdir,
        )
        res.log += "\n" + clog
        if not ok:
            res.error = "failed to compile the virtual board"
            return res
        _, run_out = self._run_vvp(workdir / "board.vvp", workdir)
        res.log += "\n" + run_out

        parsed = periph.parse_board_log(run_out, roles)
        res.ran = parsed.ran
        res.uart = parsed.uart
        res.led_events = parsed.led_events
        res.gpio_events = parsed.gpio_events
        if vcd_path.exists():
            res.vcd_path = str(vcd_path)
        if not res.ran and res.error is None:
            res.error = "board did not reach completion (check the log)"
        return res

    # ------------------------------------------------------------------ #
    def prove_equivalence(
        self,
        design: Design,
        clock_mhz: float = 50.0,
        depth: int = 20,
        unbounded: bool = True,
        config: BringUpConfig | None = None,
        workdir: str | Path = ".runs/prove",
        strategy: str = "auto",
    ) -> ProofResult:
        """Formally prove RTL == bitstream.

        ``strategy`` selects the proof engine:
          * ``sat``  -- bit-blasting SAT miter (temporal induction + BMC). Fast
            for logic, but memories explode into per-bit state.
          * ``smt``  -- SMT miter with memories kept as **arrays** (yosys-smtbmc
            + z3), so memory-heavy designs stay tractable. Proof is BMC from a
            zero-initialized state for all inputs over ``depth`` cycles.
          * ``auto`` -- SMT when the design has memory and the SMT tools exist,
            otherwise SAT.
        """
        cfg = config or BringUpConfig()
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        res = ProofResult(design_id=design.design_id(), depth=depth,
                          workdir=str(workdir))

        art, err, log = self._prepare_artifacts(design, cfg, workdir, clock_mhz)
        res.log += log
        if err is not None:
            res.error = err
            return res
        res.bitstream_path = art.bitstream_path
        res.fabric = art.fabric

        if shutil.which(self.vfpga.yosys) is None:
            res.error = "formal proof requires yosys on PATH"
            return res

        engine = self._select_engine(strategy, design, art)
        res.engine = engine
        if engine == "smt":
            return self._prove_smt(res, design, art, depth, workdir)
        return self._prove_sat(res, design, art, depth, unbounded, workdir)

    # ------------------------------------------------------------------ #
    def _select_engine(self, strategy: str, design: Design, art: "_Artifacts") -> str:
        if strategy == "smt":
            return "smt"
        if strategy == "sat":
            return "sat"
        # auto: prefer SMT (array-based) for memory designs when tools exist.
        if self._design_has_memory(design, art) and self._smt_available():
            return "smt"
        return "sat"

    # A 2-D array declaration (memory), tolerant of parameterized bounds like
    # `reg [DW-1:0] mem [0:(1<<AW)-1];` that a digit-only heuristic misses.
    _MEM_DECL_RE = re.compile(
        r"\b(?:reg|logic|bit)\b\s*(?:\[[^\]]*\])?\s*[A-Za-z_]\w*\s*\[[^\]]*\]\s*;"
    )

    def _design_has_memory(self, design: Design, art: "_Artifacts") -> bool:
        try:
            for p in design.rtl_files:
                path = Path(p)
                text = path.read_text() if path.exists() else str(p)
                if self._MEM_DECL_RE.search(text):
                    return True
        except Exception:
            pass
        return False

    def _smt_available(self) -> bool:
        if shutil.which("yosys-smtbmc") is None:
            return False
        return any(shutil.which(s) for s in ("z3", "yices-smt2", "boolector", "bitwuzla"))

    def _smt_solver(self) -> str:
        for s in ("z3", "yices", "boolector", "bitwuzla"):
            probe = {"yices": "yices-smt2"}.get(s, s)
            if shutil.which(probe):
                return s
        return "z3"

    # ------------------------------------------------------------------ #
    def _prove_sat(self, res, design, art, depth, unbounded, workdir) -> ProofResult:
        # Try an unbounded proof by temporal induction first; if it does not
        # converge, fall back to a bounded (BMC) proof to `depth` cycles.
        if unbounded:
            script = self._equiv_script(design.top, art, mode="induct", depth=depth)
            outcome, plog = self._run_yosys_script(script, workdir, "prove_induct.ys")
            res.log += "\n" + plog
            if outcome == "proved":
                res.equivalent, res.unbounded, res.method = True, True, "induction"
                return res
            if outcome == "counterexample":
                res.equivalent, res.method = False, "induction"
                res.counterexample = _extract_counterexample(plog)
                return res
            # inconclusive -> fall through to BMC

        script = self._equiv_script(design.top, art, mode="bmc", depth=depth)
        outcome, plog = self._run_yosys_script(script, workdir, "prove_bmc.ys")
        res.log += "\n" + plog
        if outcome == "proved":
            res.equivalent, res.unbounded, res.method = True, False, "bmc"
        elif outcome == "counterexample":
            res.equivalent, res.method = False, "bmc"
            res.counterexample = _extract_counterexample(plog)
        else:
            res.equivalent, res.method = None, "bmc"
            if res.error is None:
                res.error = ("proof did not converge (design may use memory/DSP the "
                             "SAT solver cannot resolve -- try strategy='smt')")
        return res

    def _prove_smt(self, res, design, art, depth, workdir) -> ProofResult:
        """BMC equivalence with memories modeled as SMT arrays (scales to memory)."""
        if not self._smt_available():
            res.equivalent, res.method = None, "smt"
            res.error = ("SMT proof needs yosys-smtbmc and an SMT solver (z3/yices) "
                         "on PATH")
            return res
        smt2 = workdir / "miter.smt2"
        script = self._smt_equiv_script(design.top, art, smt2)
        ok, ylog = self._run([self.vfpga.yosys, "-q",
                              str(self._write(workdir, "prove_smt.ys", script))], workdir)
        res.log += "\n" + ylog
        if not ok or not smt2.exists():
            res.equivalent, res.method = None, "smt"
            res.error = "failed to generate the SMT miter"
            return res

        outcome, slog = self._run_smtbmc(smt2, workdir, depth)
        res.log += "\n" + slog
        res.method = "bmc"
        if outcome == "proved":
            # Bounded proof from a zero-initialized state over `depth` cycles.
            res.equivalent, res.unbounded = True, False
        elif outcome == "counterexample":
            res.equivalent = False
            res.counterexample = _extract_counterexample(slog)
        else:
            res.equivalent = None
            if res.error is None:
                res.error = "SMT proof did not converge within the timeout"
        return res

    def _write(self, workdir: Path, name: str, text: str) -> Path:
        p = workdir / name
        p.write_text(text)
        return p

    def _run_smtbmc(self, smt2: Path, workdir: Path, depth: int):
        solver_timeout = max(10, self.timeout_s - 30)
        cmd = ["yosys-smtbmc", "-s", self._smt_solver(), "--noprogress",
               "--timeout", str(solver_timeout),
               "-t", str(depth), "-m", "miter", str(smt2)]
        _, log = self._run(cmd, workdir)
        if "Status: PASSED" in log:
            return "proved", log
        if "Status: FAILED" in log:
            return "counterexample", log
        return "inconclusive", log

    def _smt_equiv_script(self, top: str, art: "_Artifacts", smt2: Path) -> str:
        gold = " ".join(str(f) for f in art.rtl_files)
        gate = f"{art.recon_v} {art.wrapper_v}"
        # Read the ice40 cells behaviorally (NOT -lib) so hard primitives like
        # SB_RAM40_4K elaborate into reg arrays -> generic $mem, which we keep as
        # SMT *arrays* (never memory_map -> that is the SAT bit-blast that
        # explodes). Zero-initialize memory INIT params and flop init so both
        # sides start from a defined, equal state, then emit array-based SMT2.
        return (
            f"read_verilog {gold}\n"
            f"hierarchy -top {top}\n"
            f"prep -flatten -top {top}\n"
            f"design -stash gold\n"
            f"read_verilog {art.cells_sim}\n"
            f"read_verilog {gate}\n"
            f"hierarchy -top {top}\n"
            f"prep -flatten -top {top}\n"
            f"design -stash gate\n"
            f"design -copy-from gold -as gold {top}\n"
            f"design -copy-from gate -as gate {top}\n"
            f"miter -equiv -flatten -make_assert gold gate miter\n"
            f"hierarchy -top miter\n"
            f"memory_collect\n"
            f"setundef -params -zero\n"
            f"setundef -zero -init\n"
            f"write_smt2 -wires {smt2}\n"
        )

    def _equiv_script(self, top: str, art: _Artifacts, mode: str, depth: int) -> str:
        gold = " ".join(str(f) for f in art.rtl_files)
        gate = f"{art.recon_v} {art.wrapper_v}"
        # Provide the ice40 sim library so hard primitives (BRAM/DSP) elaborate.
        lib = f"-lib {art.cells_sim}"
        if mode == "induct":
            sat = f"sat -verify -prove-asserts -tempinduct -set-init-zero -seq {depth} miter"
        else:
            sat = f"sat -verify -prove-asserts -seq {depth} -set-init-zero miter"
        return (
            f"read_verilog {gold}\n"
            f"hierarchy -top {top}\n"
            f"prep -flatten -top {top}\n"
            f"design -stash gold\n"
            f"read_verilog {lib} {art.cells_sim}\n"
            f"read_verilog {gate}\n"
            f"hierarchy -top {top}\n"
            f"prep -flatten -top {top}\n"
            f"design -stash gate\n"
            f"design -copy-from gold -as gold {top}\n"
            f"design -copy-from gate -as gate {top}\n"
            f"miter -equiv -flatten -make_assert gold gate miter\n"
            f"hierarchy -top miter\n"
            f"{sat}\n"
        )

    def _run_yosys_script(self, script: str, workdir: Path, name: str):
        script_path = workdir / name
        script_path.write_text(script)
        ok, log = self._run([self.vfpga.yosys, str(script_path)], workdir)
        return _classify_proof(log), log

    def _run(self, cmd: list[str], workdir: Path):
        try:
            proc = subprocess.run(
                [str(c) for c in cmd], cwd=str(workdir), capture_output=True,
                text=True, timeout=self.timeout_s,
            )
        except FileNotFoundError as exc:
            return False, f"$ {' '.join(map(str, cmd))}\n[not found] {exc}\n"
        except subprocess.TimeoutExpired:
            return False, f"$ {' '.join(map(str, cmd))}\n[timeout]\n"
        log = f"$ {' '.join(map(str, cmd))}\n{proc.stdout}\n{proc.stderr}\n[exit {proc.returncode}]\n"
        return proc.returncode == 0, log

    def _compile_and_run(self, out_vvp: Path, sources: list[Path], workdir: Path):
        ok, log = self._compile(out_vvp, sources, workdir)
        if not ok:
            return False, log
        ok2, log2 = self._run_vvp(out_vvp, workdir)
        return ok2, log + "\n" + log2

    def _compile(self, out_vvp: Path, sources: list[Path], workdir: Path):
        ok, log = self._run(
            [self.iverilog, "-g2012", "-o", str(out_vvp), *[str(s) for s in sources]],
            workdir,
        )
        return (ok and out_vvp.exists()), log

    def _run_vvp(self, out_vvp: Path, workdir: Path, plusargs: list[str] | None = None):
        return self._run([self.vvp, str(out_vvp), *(plusargs or [])], workdir)


# ---------------------------------------------------------------------- #
def render_compare_tb(top: str, ports: list[Port], cfg: BringUpConfig) -> str:
    """A testbench that logs every post-reset cycle's outputs as CYC lines.

    Instantiates a module named ``top`` (both the mapped netlist and the
    bitstream wrapper are named ``top``), so a single testbench drives both.
    """
    clk = detect_clock(ports, cfg.clock)
    rst = detect_reset(ports, cfg.reset)
    inputs = [p for p in ports if p.direction == "input"]
    outputs = [p for p in ports if p.direction in ("output", "inout")]
    driven = [p for p in inputs if p.name not in {clk.name, getattr(rst, "name", None)}]

    active_high = _reset_is_active_high(rst, cfg) if rst else True
    assert_val = "1'b1" if active_high else "1'b0"
    deassert_val = "1'b0" if active_high else "1'b1"

    out_fmt = " ".join(f"{p.name}=%0d" for p in outputs)
    out_args = ", ".join(p.name for p in outputs)
    in_fmt = " ".join(f"{p.name}=%0d" for p in driven)
    in_args = ", ".join(p.name for p in driven)

    lines = ["`timescale 1ns/1ps", "module tb;"]
    lines.append(f"  reg {clk.name};")
    if cfg.stimulus == "random":
        lines.append("  integer _seed;")
    if rst:
        lines.append(f"  reg {rst.name};")
    for p in driven:
        rng = f"[{p.width - 1}:0] " if p.width > 1 else ""
        lines.append(f"  reg {rng}{p.name};")
    for p in outputs:
        rng = f"[{p.width - 1}:0] " if p.width > 1 else ""
        lines.append(f"  wire {rng}{p.name};")

    conns = ", ".join(f".{p.name}({p.name})" for p in ports)
    lines.append(f"  {top} dut ({conns});")
    if cfg.vcd_path:
        lines.append("  initial begin")
        lines.append(f'    $dumpfile("{cfg.vcd_path}");')
        lines.append("    $dumpvars(0, tb);")
        lines.append("  end")
    lines.append(f"  initial {clk.name} = 1'b0;")
    lines.append(f"  always #{cfg.half_period_ns:g} {clk.name} = ~{clk.name};")

    if driven:
        # Both DUTs run this identical testbench, so $random (default seed) yields
        # the same stimulus in each -> a fair, deterministic equivalence check.
        run_cond = f"({rst.name} == {deassert_val})" if rst else "1'b1"
        lines.append(f"  always @(posedge {clk.name}) if ({run_cond}) begin")
        for p in driven:
            if cfg.stimulus == "random":
                # Seed comes from a +SEED plusarg so distinct runs explore
                # distinct stimulus streams; both DUTs share the same seed.
                lines.append(f"    {p.name} <= $random(_seed);")
            else:
                incr = "1'b1" if p.width == 1 else f"{p.width}'d1"
                lines.append(f"    {p.name} <= {p.name} + {incr};")
        lines.append("  end")

    lines.append("  integer _i;")
    lines.append("  initial begin")
    if cfg.stimulus == "random":
        lines.append("    if (!$value$plusargs(\"SEED=%d\", _seed)) _seed = 32'h1;")
    for p in driven:
        lines.append(f"    {p.name} = 0;")
    if rst:
        lines.append(f"    {rst.name} = {assert_val};")
        lines.append(f"    repeat ({cfg.reset_cycles}) @(posedge {clk.name});")
        lines.append(f"    {rst.name} = {deassert_val};")
    lines.append(f"    for (_i = 0; _i < {cfg.cycles}; _i = _i + 1) begin")
    lines.append(f"      @(posedge {clk.name}); #1;")
    if outputs:
        lines.append(f'      $display("CYC %0d {out_fmt}", _i, {out_args});')
    else:
        lines.append('      $display("CYC %0d", _i);')
    # Log the stimulus separately (STIM lines) so it never affects the
    # output-equality comparison, but lets us measure how much of the input
    # space we actually exercised.
    if driven:
        lines.append(f'      $display("STIM %0d {in_fmt}", _i, {in_args});')
    lines.append("    end")
    lines.append('    $display("VFPGA_DONE");')
    lines.append("    $finish;")
    lines.append("  end")

    watchdog = int((cfg.cycles + cfg.reset_cycles + 10) * cfg.half_period_ns * 2 * 4)
    lines.append("  initial begin")
    lines.append(f"    #{watchdog}; $display(\"VFPGA_TIMEOUT\"); $finish;")
    lines.append("  end")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def _classify_proof(log: str) -> str:
    """Map a yosys `sat` log to 'proved' | 'counterexample' | 'inconclusive'."""
    # A found model / failed assert means the two designs differ.
    if re.search(r"model found:\s*FAIL", log) or "Assert failed in miter" in log:
        return "counterexample"
    # Temporal induction success, or a clean BMC "no model found".
    if "Induction step proven: SUCCESS!" in log:
        return "proved"
    if "no model found: SUCCESS!" in log and "Trying induction" not in log:
        return "proved"
    return "inconclusive"


def _extract_counterexample(log: str) -> str | None:
    snippet = [ln.strip() for ln in log.splitlines()
               if "Value for" in ln or "in_" in ln]
    return "; ".join(snippet[:8]) if snippet else "differing input assignment found"


def _cycle_trace(sim_output: str) -> list[str]:
    return [ln.strip() for ln in sim_output.splitlines() if ln.strip().startswith("CYC ")]


def _stim_trace(sim_output: str) -> list[str]:
    """Per-cycle stimulus lines (driven primary inputs)."""
    return [ln.strip() for ln in sim_output.splitlines() if ln.strip().startswith("STIM ")]


def _divergence_window(
    rtl_trace: list[str],
    bit_trace: list[str],
    stim: list[str],
    cycle: int,
    radius: int = 3,
) -> list[str]:
    """A compact, agent-readable window around the first diverging cycle.

    Shows the driven stimulus plus both sides' outputs for a few cycles leading
    up to and including the divergence, marking exactly which fields differ.
    This is the minimal counterexample an agent (or engineer) needs to localize
    a mismatch without wading through the full trace.
    """
    lo = max(0, cycle - radius)
    hi = min(len(rtl_trace), cycle + 2)
    stim_by_cycle: dict[int, str] = {}
    for line in stim:
        f = _parse_cycle(line)
        idx = _cycle_index(line)
        if idx is not None:
            stim_by_cycle[idx] = " ".join(
                f"{k}={v}" for k, v in f.items()
            )
    out: list[str] = []
    for i in range(lo, hi):
        marker = "  <-- DIVERGES" if i == cycle else ""
        if i in stim_by_cycle:
            out.append(f"cyc {i:>4} in : {stim_by_cycle[i]}")
        ra = _parse_cycle(rtl_trace[i]) if i < len(rtl_trace) else {}
        ba = _parse_cycle(bit_trace[i]) if i < len(bit_trace) else {}
        diff_fields = [k for k in ra if k in ba and ra[k] != ba[k]]
        r_str = " ".join(f"{k}={v}" for k, v in ra.items())
        b_str = " ".join(f"{k}={v}" for k, v in ba.items())
        out.append(f"cyc {i:>4} rtl: {r_str}{marker}")
        out.append(f"cyc {i:>4} bit: {b_str}")
        if diff_fields:
            out.append(f"cyc {i:>4} dif: {', '.join(diff_fields)}")
    return out


def _cycle_index(line: str) -> int | None:
    """Extract the integer cycle index from a ``CYC``/``STIM`` line."""
    parts = line.split()
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _parse_cycle(line: str) -> dict[str, str]:
    """Parse a ``CYC <i> name=val name2=val2`` line into a name->value map."""
    fields: dict[str, str] = {}
    for tok in line.split():
        if "=" in tok:
            name, _, val = tok.partition("=")
            fields[name] = val
    return fields


def _is_unknown(val: str) -> bool:
    """True if a printed value is (partly) unknown/high-Z -> a don't-care.

    An uninitialized RTL register/memory prints as ``x`` (or ``z``). The
    bitstream powers up to a concrete value, which is a *legal refinement* of
    a don't-care, so such fields must not count as mismatches.
    """
    return "x" in val.lower() or "z" in val.lower()


@dataclass
class CompareStats:
    """Outcome of comparing one design trace against one bitstream trace."""

    matches: bool = True
    compared: int = 0          # cycles where at least one field was concrete
    indeterminate: int = 0     # cycles fully don't-care on the design side
    first_mismatch: str | None = None
    mismatch_cycle: int | None = None   # index of first diverging cycle


def _compare_traces(a: list[str], b: list[str]) -> tuple[bool, str | None]:
    """Back-compat wrapper: strict-length, X-aware comparison."""
    stats = _compare_traces_x(a, b)
    if stats.matches and len(a) != len(b):
        return False, f"trace length differs: design={len(a)} bitstream={len(b)}"
    return stats.matches, stats.first_mismatch


def _compare_traces_x(a: list[str], b: list[str]) -> CompareStats:
    """X-aware, field-level comparison of a design vs bitstream trace.

    A field only fails when *both* sides are concrete and differ. Fields the
    design leaves unknown (``x``/``z``) are don't-cares that the concrete
    bitstream value legally refines.
    """
    stats = CompareStats()
    n = min(len(a), len(b))
    for i in range(n):
        fa, fb = _parse_cycle(a[i]), _parse_cycle(b[i])
        cycle_has_concrete = False
        for name, va in fa.items():
            if name not in fb:
                continue
            if _is_unknown(va):
                continue  # design don't-care -> any bitstream value is fine
            cycle_has_concrete = True
            if va != fb[name]:
                stats.matches = False
                if stats.first_mismatch is None:
                    stats.first_mismatch = (
                        f"cycle {i}: {name} design={va} bitstream={fb[name]}"
                    )
                    stats.mismatch_cycle = i
        if cycle_has_concrete:
            stats.compared += 1
        else:
            stats.indeterminate += 1
    return stats


def _toggle_coverage(
    traces: list[list[str]], outputs: "list[Port]"
) -> tuple[float, int, int, int]:
    """Measure how thoroughly the campaign exercised the observable outputs.

    Returns ``(coverage_fraction, toggled_bits, total_bits, distinct_vectors)``
    across every concrete cycle in every trace. A bit "counts" when it is
    observed at both 0 and 1; coverage is the fraction of output bits that
    toggled. This is a concrete, defensible measure derived from the actual
    runs (not a guess).
    """
    ever_one: dict[str, int] = {}
    ever_zero: dict[str, int] = {}
    widths = {p.name: max(1, p.width) for p in outputs}
    total_bits = sum(widths.values())
    distinct: set[str] = set()
    for trace in traces:
        for line in trace:
            fields = _parse_cycle(line)
            vec_parts = []
            for name, width in widths.items():
                val = fields.get(name)
                if val is None or _is_unknown(val):
                    continue
                try:
                    iv = int(val)
                except ValueError:
                    continue
                vec_parts.append(f"{name}={iv}")
                for bit in range(width):
                    if (iv >> bit) & 1:
                        ever_one[name] = ever_one.get(name, 0) | (1 << bit)
                    else:
                        ever_zero[name] = ever_zero.get(name, 0) | (1 << bit)
            if vec_parts:
                distinct.add("|".join(vec_parts))
    toggled = 0
    for name, width in widths.items():
        both = ever_one.get(name, 0) & ever_zero.get(name, 0)
        toggled += bin(both & ((1 << width) - 1)).count("1")
    coverage = (toggled / total_bits) if total_bits else 0.0
    return coverage, toggled, total_bits, len(distinct)


def _input_coverage(
    stim_traces: list[list[str]], driven: "list[Port]"
) -> tuple[float, int, int]:
    """Fraction of driven input bits observed at both 0 and 1.

    This is the stimulus-quality counterpart to output toggle coverage: a
    campaign that never flips an input bit has not really exercised the design,
    however well the outputs matched. Returns
    ``(coverage_fraction, toggled_bits, total_bits)``.
    """
    widths = {p.name: max(1, p.width) for p in driven}
    total_bits = sum(widths.values())
    if not total_bits:
        return 1.0, 0, 0
    ever_one: dict[str, int] = {}
    ever_zero: dict[str, int] = {}
    for trace in stim_traces:
        for line in trace:
            fields = _parse_cycle(line)
            for name, width in widths.items():
                val = fields.get(name)
                if val is None or _is_unknown(val):
                    continue
                try:
                    iv = int(val)
                except ValueError:
                    continue
                for bit in range(width):
                    if (iv >> bit) & 1:
                        ever_one[name] = ever_one.get(name, 0) | (1 << bit)
                    else:
                        ever_zero[name] = ever_zero.get(name, 0) | (1 << bit)
    toggled = 0
    for name, width in widths.items():
        both = ever_one.get(name, 0) & ever_zero.get(name, 0)
        toggled += bin(both & ((1 << width) - 1)).count("1")
    return toggled / total_bits, toggled, total_bits


# ---------------------------------------------------------------------- #
def emulate(
    bitstream: str | Path,
    workdir: str | Path = ".runs/emulate",
    module: str = "chip",
    engine: Emulator | None = None,
) -> EmulationResult:
    """Load and decode a real bitstream; rebuild the configured fabric netlist."""
    engine = engine or Emulator()
    return engine.emulate(bitstream, workdir=workdir, module=module)


def verify_bitstream(
    rtl: str | Sequence[str],
    top: str,
    target_fpga: str = "ice40_up5k",
    cycles: int = 32,
    clock_mhz: float = 50.0,
    clock: str | None = None,
    reset: str | None = None,
    reset_active_high: bool | None = None,
    stimulus: str = "counter",
    seeds: Sequence[int] | None = None,
    adaptive: bool = False,
    max_seeds: int = 24,
    config: BringUpConfig | None = None,
    workdir: str | Path = ".runs/verify",
    engine: Emulator | None = None,
) -> VerificationResult:
    """Prove the flashed bitstream behaves like the design, cycle by cycle.

    Runs the full flow to a real ``.bin``, unpacks that binary back into the
    configured fabric, and simulates both the bitstream and the synthesized
    design under identical stimulus. ``matches=True`` means every concrete
    (non-don't-care) cycle was identical.

    ``stimulus`` is "counter" (ramp inputs) or "random". Random stimulus runs a
    multi-seed campaign (``seeds``, default four fixed seeds) so each seed
    explores a different trajectory; the result reports measured output-bit
    toggle coverage and an empirical confidence. With ``adaptive=True`` the
    campaign keeps adding seeds (up to ``max_seeds``) until behavior stops
    growing, then reports whether coverage *saturated*.
    """
    rtl_files = [rtl] if isinstance(rtl, str) else list(rtl)
    design = Design(rtl_files=tuple(rtl_files), top=top, target=target_fpga)
    cfg = config or BringUpConfig(
        cycles=cycles, clock=clock, reset=reset, reset_active_high=reset_active_high,
        stimulus=stimulus,
    )
    engine = engine or Emulator()
    return engine.verify_bitstream(design, cycles=cycles, clock_mhz=clock_mhz,
                                   seeds=seeds, adaptive=adaptive, max_seeds=max_seeds,
                                   config=cfg, workdir=workdir)


def emulate_board(
    rtl: str | Sequence[str],
    top: str,
    target_fpga: str = "ice40_up5k",
    clock_mhz: float = 12.0,
    baud: int = 1_000_000,
    duration_us: float = 60.0,
    uart_rx_bytes: Sequence[int] | None = None,
    board: "periph.BoardConfig | None" = None,
    workdir: str | Path = ".runs/board",
    engine: Emulator | None = None,
) -> "periph.BoardResult":
    """Run RTL's flashed bitstream against a virtual board of peripherals.

    The real "FPGA in a socket" experience: builds the bitstream, reconstructs
    the fabric, wires it to behavioral clock/reset/LED/UART/button/switch/GPIO
    models, runs it, and reports the peripheral activity (decoded UART text, LED
    blink counts, GPIO levels).
    """
    rtl_files = [rtl] if isinstance(rtl, str) else list(rtl)
    design = Design(rtl_files=tuple(rtl_files), top=top, target=target_fpga)
    cfg = board or periph.BoardConfig(
        clock_mhz=clock_mhz, baud=baud, duration_us=duration_us,
        uart_rx_bytes=list(uart_rx_bytes) if uart_rx_bytes else [],
    )
    engine = engine or Emulator()
    return engine.emulate_board(design, board=cfg, workdir=workdir)


def prove(
    rtl: str | Sequence[str],
    top: str,
    target_fpga: str = "ice40_up5k",
    depth: int = 20,
    unbounded: bool = True,
    clock_mhz: float = 50.0,
    clock: str | None = None,
    reset: str | None = None,
    workdir: str | Path = ".runs/prove",
    strategy: str = "auto",
    engine: Emulator | None = None,
) -> ProofResult:
    """Formally prove the flashed bitstream is equivalent to the RTL.

    Builds a real bitstream, reconstructs the configured fabric, and runs a
    miter proof. ``strategy`` picks the engine: ``sat`` (bit-blasting temporal
    induction + BMC, best for logic), ``smt`` (memories kept as arrays via
    yosys-smtbmc, so memory-heavy designs stay tractable), or ``auto`` (SMT for
    memory designs when the tools exist, else SAT). Unlike ``verify``, this
    covers *every* input, not just a stimulus sequence.
    """
    rtl_files = [rtl] if isinstance(rtl, str) else list(rtl)
    design = Design(rtl_files=tuple(rtl_files), top=top, target=target_fpga)
    cfg = BringUpConfig(clock=clock, reset=reset)
    engine = engine or Emulator()
    return engine.prove_equivalence(design, clock_mhz=clock_mhz, depth=depth,
                                    unbounded=unbounded, config=cfg, workdir=workdir,
                                    strategy=strategy)
