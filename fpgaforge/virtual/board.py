"""Virtual board harness generation.

Produces a self-contained Icarus-Verilog testbench that instantiates the
synthesized design (the "virtual FPGA"), drives a clock and reset, applies
deterministic stimulus to the remaining inputs, dumps a VCD waveform, and
prints sampled outputs. Pure string generation so it is trivially unit-tested
without any tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Heuristic name sets for auto-detecting special ports.
_CLOCK_NAMES = {"clk", "clock", "sys_clk", "clk_i", "i_clk", "clk_in", "clkin"}
_RESET_NAMES = {
    "rst", "reset", "rst_n", "resetn", "rst_i", "i_rst", "arst",
    "arst_n", "nrst", "rstn", "reset_n",
}


@dataclass
class Port:
    name: str
    direction: str  # "input" | "output" | "inout"
    width: int = 1


@dataclass
class BringUpConfig:
    """Configuration for a virtual bring-up run."""

    cycles: int = 64
    reset_cycles: int = 4
    half_period_ns: float = 5.0            # 5ns -> 100 MHz sim clock
    clock: str | None = None               # override auto-detected clock port
    reset: str | None = None               # override auto-detected reset port
    reset_active_high: bool | None = None  # None -> infer from name
    stimulus: str = "counter"              # "counter" | "zero"
    vcd_path: str = "bringup.vcd"
    dut_instance: str = "dut"
    extra_display: bool = True


def detect_clock(ports: list[Port], override: str | None = None) -> Port | None:
    if override:
        return next((p for p in ports if p.name == override), None)
    inputs = [p for p in ports if p.direction == "input" and p.width == 1]
    for p in inputs:
        if p.name.lower() in _CLOCK_NAMES:
            return p
    for p in inputs:
        low = p.name.lower()
        if "clk" in low or "clock" in low:
            return p
    return None


def detect_reset(ports: list[Port], override: str | None = None) -> Port | None:
    if override:
        return next((p for p in ports if p.name == override), None)
    inputs = [p for p in ports if p.direction == "input" and p.width == 1]
    for p in inputs:
        if p.name.lower() in _RESET_NAMES:
            return p
    for p in inputs:
        low = p.name.lower()
        if "rst" in low or "reset" in low:
            return p
    return None


def _reset_is_active_high(reset: Port, cfg: BringUpConfig) -> bool:
    if cfg.reset_active_high is not None:
        return cfg.reset_active_high
    low = reset.name.lower()
    # Names ending in _n / n are conventionally active-low.
    return not (low.endswith("_n") or low.endswith("n"))


def _decl(kind: str, port: Port) -> str:
    rng = f"[{port.width - 1}:0] " if port.width > 1 else ""
    return f"  {kind} {rng}{port.name};"


def render_testbench(top: str, ports: list[Port], cfg: BringUpConfig) -> str:
    """Render a virtual-board testbench for `top`.

    Raises ValueError if no clock port can be found (a cycle-based bring-up
    needs a clock; purely combinational designs are out of scope here).
    """
    clk = detect_clock(ports, cfg.clock)
    if clk is None:
        raise ValueError(
            "no clock port detected; specify one via BringUpConfig.clock"
        )
    rst = detect_reset(ports, cfg.reset)

    inputs = [p for p in ports if p.direction == "input"]
    outputs = [p for p in ports if p.direction in ("output", "inout")]
    driven = [p for p in inputs if p.name not in {clk.name, getattr(rst, "name", None)}]

    active_high = _reset_is_active_high(rst, cfg) if rst else True
    assert_val = "1'b1" if active_high else "1'b0"
    deassert_val = "1'b0" if active_high else "1'b1"

    lines: list[str] = []
    lines.append("`timescale 1ns/1ps")
    lines.append("// Auto-generated virtual-board harness for fpgaforge bring-up.")
    lines.append("module tb;")

    # Signal declarations.
    lines.append(f"  reg {clk.name};")
    if rst:
        lines.append(f"  reg {rst.name};")
    for p in driven:
        lines.append(_decl("reg", p))
    for p in outputs:
        lines.append(_decl("wire", p))

    # DUT instantiation (named port connections).
    conns = ", ".join(f".{p.name}({p.name})" for p in ports)
    lines.append("")
    lines.append(f"  {top} {cfg.dut_instance} ({conns});")

    # Clock generation.
    lines.append("")
    lines.append(f"  initial {clk.name} = 1'b0;")
    lines.append(
        f"  always #{cfg.half_period_ns:g} {clk.name} = ~{clk.name};"
    )

    # Deterministic stimulus for the remaining inputs.
    if driven:
        lines.append("")
        run_cond = f"({rst.name} == {deassert_val})" if rst else "1'b1"
        lines.append(f"  always @(posedge {clk.name}) begin")
        lines.append(f"    if ({run_cond}) begin")
        for p in driven:
            if cfg.stimulus == "counter":
                incr = "1'b1" if p.width == 1 else f"{p.width}'d1"
                lines.append(f"      {p.name} <= {p.name} + {incr};")
            else:  # "zero"
                lines.append(f"      {p.name} <= {p.name};")
        lines.append("    end")
        lines.append("  end")

    # Main sequence: reset, run, report.
    lines.append("")
    lines.append("  integer _i;")
    lines.append("  initial begin")
    lines.append(f'    $dumpfile("{cfg.vcd_path}");')
    lines.append("    $dumpvars(0, tb);")
    for p in driven:
        lines.append(f"    {p.name} = 0;")
    if rst:
        lines.append(f"    {rst.name} = {assert_val};")
        lines.append(f"    repeat ({cfg.reset_cycles}) @(posedge {clk.name});")
        lines.append(f"    {rst.name} = {deassert_val};")
    lines.append(f"    repeat ({cfg.cycles}) @(posedge {clk.name});")
    lines.append('    $display("VFPGA_DONE cycles=%0d", ' + str(cfg.cycles) + ");")
    if cfg.extra_display:
        for p in outputs:
            lines.append(
                f'    $display("VFPGA_OUT {p.name}=%0d (0x%0h)", {p.name}, {p.name});'
            )
    lines.append("    $finish;")
    lines.append("  end")

    # Safety watchdog so a hung design cannot block forever.
    watchdog = int((cfg.cycles + cfg.reset_cycles + 10) * cfg.half_period_ns * 2 * 4)
    lines.append("")
    lines.append("  initial begin")
    lines.append(f"    #{watchdog};")
    lines.append('    $display("VFPGA_TIMEOUT");')
    lines.append("    $finish;")
    lines.append("  end")

    lines.append("endmodule")
    return "\n".join(lines) + "\n"
