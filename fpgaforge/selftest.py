"""On-FPGA self-test (BIST) generation: verify the *silicon*, not just the bits.

Silicon defects, marginal power rails, and environment (temperature, noisy
clocks) are physically outside any simulator's reach. What a simulator *can*
do is generate the artifact that detects them on the bench: a built-in
self-test harness that wraps the design with

* a maximal-length **LFSR** driving every input with pseudo-random stimulus,
* a **MISR** (multiple-input signature register) compressing every output of
  every cycle into a 64-bit signature, and
* a comparator against a **golden signature** computed cycle-accurately in
  simulation and baked into the harness as a parameter.

Flash the harness bitstream; after ``CYCLES`` clock ticks the ``test_done``
pin goes high and ``test_pass`` reports whether the physical chip produced
bit-for-bit the same signature the simulator predicted. Any stuck-at fault,
timing failure at the real voltage/temperature corner, or configuration
upset perturbs the signature and drops ``test_pass``.

The generator is honest about its own soundness: the golden simulation runs
4-state, so if the design has state that reset does not initialize, the
signature comes back X and generation *fails with an explanation* instead of
emitting a self-test that could disagree with hardware.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_MISR_POLY = 0x000000000000001B  # x^64 + x^4 + x^3 + x + 1 (primitive)
_LFSR_SEED = 0x5EED5EED5EED5EED


@dataclass
class SelfTestReport:
    design_id: str = ""
    harness_path: str | None = None
    harness_top: str = ""
    golden_signature: str | None = None    # 16 hex digits
    cycles: int = 0
    warmup: int = 0
    validated: bool = False                # harness re-simulated, test_pass==1
    clock_port: str | None = None
    reset_port: str | None = None
    workdir: str | None = None
    log: str = ""
    error: str | None = None
    files: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.error:
            return f"selftest: FAILED to generate\nerror   : {self.error}"
        lines = [
            "on-FPGA self-test harness generated",
            f"design   : {self.design_id}",
            f"harness  : {self.harness_path} (top: {self.harness_top})",
            f"golden   : 0x{self.golden_signature}",
            f"coverage : {self.cycles} pseudo-random cycles after "
            f"{self.warmup}-cycle reset",
            f"validated: {'yes -- harness simulation asserts test_pass' if self.validated else 'no'}",
            "usage    : build/flash the harness; wire test_pass and test_done "
            "to LEDs.",
            "           test_done high + test_pass high  => the physical chip "
            "matches the simulation bit-for-bit",
            "           test_done high + test_pass low   => silicon/board-level "
            "fault (defect, rail, clock, thermal)",
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------------- #
def _ports_via_yosys(rtl_files: Sequence[str], top: str, workdir: Path):
    from .virtual.vfpga import _ports_from_netlist

    out_json = workdir / "ports.json"
    files = " ".join(str(Path(f).resolve()) for f in rtl_files)
    script = f"read_verilog {files}\nprep -top {top}\nwrite_json {out_json}\n"
    (workdir / "ports.ys").write_text(script)
    proc = subprocess.run(
        ["yosys", "-q", "-s", str(workdir / "ports.ys")],
        capture_output=True, text=True, timeout=300, cwd=str(workdir),
    )
    if proc.returncode != 0 or not out_json.exists():
        return None, proc.stdout + proc.stderr
    return _ports_from_netlist(json.loads(out_json.read_text()), top), ""


def _is_active_low(name: str) -> bool:
    low = name.lower()
    return low.endswith(("_n", "_ni")) or low in ("rstn", "resetn", "nrst",
                                                  "nreset", "rst_n", "reset_n")


def _pad_expr(total: int, have: int) -> str:
    pad = total - have
    return f"{{{{{pad}{{1'b0}}}}, out_bus}}" if pad > 0 else "out_bus"


def _render_harness(top: str, ports, clock, reset, golden: int,
                    cycles: int, warmup: int) -> str:
    ins = [p for p in ports if p.direction == "input"
           and p.name not in {clock.name, reset.name if reset else None}]
    outs = [p for p in ports if p.direction == "output"]
    iw = sum(p.width for p in ins)
    ow = sum(p.width for p in outs)

    # DUT connections: inputs sliced out of the replicated LFSR, outputs
    # concatenated into one bus for the MISR fold.
    conns, off = [f".{clock.name}(clk)"], 0
    if reset is not None:
        conns.append(f".{reset.name}(dut_rst)")
    for p in ins:
        conns.append(f".{p.name}(stim[{off + p.width - 1}:{off}])"
                     if p.width > 1 else f".{p.name}(stim[{off}])")
        off += p.width
    out_names = []
    for p in outs:
        out_names.append(f"dut_{p.name}")
        conns.append(f".{p.name}(dut_{p.name})")

    out_decls = "\n".join(
        f"    wire [{p.width - 1}:0] dut_{p.name};" if p.width > 1
        else f"    wire dut_{p.name};" for p in outs)
    out_concat = "{" + ", ".join(reversed(out_names)) + "}" if outs else "1'b0"

    # Fold the (padded) output bus into 64 bits by XOR.
    folds = max(1, -(-max(ow, 1) // 64))
    fold_terms = " ^ ".join(f"out_pad[{i * 64 + 63}:{i * 64}]"
                            for i in range(folds))
    stim_reps = max(1, -(-max(iw, 1) // 64))

    rst_drive = ""
    if reset is not None:
        level_active = "1'b0" if _is_active_low(reset.name) else "1'b1"
        level_idle = "1'b1" if _is_active_low(reset.name) else "1'b0"
        rst_drive = (f"    // hold the DUT in reset through warmup so its state\n"
                     f"    // (and therefore the signature) is deterministic\n"
                     f"    wire dut_rst = (cyc < WARMUP) ? {level_active} : {level_idle};\n")

    return f"""\
// Auto-generated on-FPGA self-test (BIST) harness for `{top}`.
// LFSR stimulus -> DUT -> MISR signature, compared against a golden
// signature computed by cycle-accurate simulation. See fpgaforge/selftest.py.
module {top}_selftest #(
    parameter [63:0] GOLDEN = 64'h{golden:016x},
    parameter [31:0] CYCLES = 32'd{cycles},
    parameter [31:0] WARMUP = 32'd{warmup}
) (
    input  wire clk,
    output wire test_done,
    output wire test_pass
);
    reg [63:0] lfsr = 64'h{_LFSR_SEED:016x};
    reg [63:0] misr = 64'd0;
    reg [31:0] cyc  = 32'd0;

    wire running = cyc < (WARMUP + CYCLES);
    assign test_done = ~running;
    assign test_pass = test_done && (misr == GOLDEN);

{rst_drive}\
    wire [{stim_reps * 64 - 1}:0] stim_full = {{{stim_reps}{{lfsr}}}};
    wire [{max(iw, 1) - 1}:0] stim = stim_full[{max(iw, 1) - 1}:0];

{out_decls}
    wire [{max(ow, 1) - 1}:0] out_bus = {out_concat};
    wire [{folds * 64 - 1}:0] out_pad = {_pad_expr(folds * 64, max(ow, 1))};
    wire [63:0] fold = {fold_terms};

    {top} dut ({', '.join(conns)});

    wire lfsr_fb = lfsr[63] ^ lfsr[62] ^ lfsr[60] ^ lfsr[59];
    always @(posedge clk) begin
        if (running) begin
            cyc  <= cyc + 32'd1;
            lfsr <= {{lfsr[62:0], lfsr_fb}};
            if (cyc >= WARMUP)
                misr <= ({{misr[62:0], 1'b0}} ^ ({{64{{misr[63]}}}} & 64'h{_MISR_POLY:016x})) ^ fold;
        end
    end
endmodule
"""


_TB = """\
`timescale 1ns/1ps
module selftest_tb;
    reg clk = 1'b0;
    wire done, pass;
    {top}_selftest dut (.clk(clk), .test_done(done), .test_pass(pass));
    always #5 clk = ~clk;
    initial begin
        wait (done === 1'b1);
        @(posedge clk); #1;
        $display("SIGNATURE %h", dut.misr);
        $display("PASS %b", pass);
        $finish;
    end
    initial begin
        #{timeout};
        $display("TIMEOUT");
        $finish;
    end
endmodule
"""


def _simulate(rtl_files, harness_v, tb_v, workdir: Path, top: str):
    out = workdir / "selftest.vvp"
    srcs = [str(Path(f).resolve()) for f in rtl_files] + [str(harness_v), str(tb_v)]
    p1 = subprocess.run(["iverilog", "-g2012", "-o", str(out), *srcs],
                        capture_output=True, text=True, timeout=300,
                        cwd=str(workdir))
    if p1.returncode != 0:
        return None, None, p1.stdout + p1.stderr
    p2 = subprocess.run(["vvp", str(out)], capture_output=True, text=True,
                        timeout=600, cwd=str(workdir))
    log = p1.stdout + p1.stderr + p2.stdout + p2.stderr
    sig = re.search(r"SIGNATURE\s+([0-9a-fA-FxXzZ]+)", p2.stdout)
    ok = re.search(r"PASS\s+([01xz])", p2.stdout)
    return (sig.group(1) if sig else None,
            ok.group(1) if ok else None, log)


# ------------------------------------------------------------------------- #
def generate_selftest(
    rtl: str | Sequence[str],
    top: str,
    cycles: int = 2048,
    warmup: int = 8,
    workdir: str | Path = ".runs/selftest",
) -> SelfTestReport:
    """Generate (and validate in simulation) an on-FPGA self-test harness."""
    from .virtual.board import detect_clock, detect_reset

    rtl_files = [rtl] if isinstance(rtl, str) else list(rtl)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    rep = SelfTestReport(design_id=f"{top}", cycles=cycles, warmup=warmup,
                         workdir=str(workdir), harness_top=f"{top}_selftest")

    for tool in ("yosys", "iverilog", "vvp"):
        if shutil.which(tool) is None:
            rep.error = f"selftest generation requires {tool} on PATH"
            return rep

    ports, log = _ports_via_yosys(rtl_files, top, workdir)
    rep.log += log
    if not ports:
        rep.error = f"could not extract ports of {top!r}: {log[-400:]}"
        return rep
    if any(p.direction == "inout" for p in ports):
        rep.error = "designs with inout ports are not supported by selftest"
        return rep

    clock = detect_clock(ports, None)
    if clock is None:
        rep.error = "no clock port detected; a self-test needs a clocked design"
        return rep
    reset = detect_reset(ports, None)
    rep.clock_port = clock.name
    rep.reset_port = reset.name if reset else None

    harness_v = workdir / f"{top}_selftest.v"
    tb_v = workdir / "selftest_tb.v"
    tb_v.write_text(_TB.format(top=top, timeout=(cycles + warmup) * 10 + 100000))

    # Pass 1: golden signature from a cycle-accurate simulation.
    harness_v.write_text(_render_harness(top, ports, clock, reset,
                                         golden=0, cycles=cycles, warmup=warmup))
    sig, _, slog = _simulate(rtl_files, harness_v, tb_v, workdir, top)
    rep.log += "\n" + slog
    if sig is None:
        rep.error = "golden simulation failed (see log)"
        return rep
    if re.search(r"[xXzZ]", sig):
        rep.error = (
            "golden signature contains X/Z: the design has state that reset "
            "does not initialize, so a simulation-predicted signature cannot "
            "be trusted on hardware. Add reset coverage for all state (or a "
            "longer warmup) and regenerate."
        )
        return rep
    golden = int(sig, 16)

    # Pass 2: bake the golden in, re-simulate, require test_pass == 1.
    harness_v.write_text(_render_harness(top, ports, clock, reset,
                                         golden=golden, cycles=cycles,
                                         warmup=warmup))
    sig2, ok, slog2 = _simulate(rtl_files, harness_v, tb_v, workdir, top)
    rep.log += "\n" + slog2
    rep.golden_signature = f"{golden:016x}"
    rep.harness_path = str(harness_v)
    rep.validated = (ok == "1" and sig2 == sig)
    if not rep.validated:
        rep.error = "harness validation failed: test_pass did not assert"
        return rep

    report_json = workdir / "selftest_report.json"
    report_json.write_text(json.dumps({
        "design": top, "harness": str(harness_v),
        "harness_top": rep.harness_top,
        "golden_signature": rep.golden_signature,
        "cycles": cycles, "warmup": warmup,
        "clock_port": rep.clock_port, "reset_port": rep.reset_port,
        "outputs": {"test_done": "high when the test finished",
                    "test_pass": "high iff the silicon matched the simulation"},
    }, indent=2))
    rep.files = [str(harness_v), str(report_json)]
    return rep
