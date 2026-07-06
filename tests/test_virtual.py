import shutil

import pytest

from fpgaforge.virtual.board import (
    BringUpConfig,
    Port,
    detect_clock,
    detect_reset,
    render_testbench,
)
from fpgaforge.virtual.vfpga import VirtualFPGA, _parse_outputs, bringup

COUNTER_PORTS = [
    Port("clk", "input", 1),
    Port("rst", "input", 1),
    Port("count", "output", 8),
]

MAC_PORTS = [
    Port("clk", "input", 1),
    Port("rst", "input", 1),
    Port("a", "input", 8),
    Port("b", "input", 8),
    Port("c", "input", 16),
    Port("y", "output", 16),
]


def test_detect_clock_and_reset():
    assert detect_clock(COUNTER_PORTS).name == "clk"
    assert detect_reset(COUNTER_PORTS).name == "rst"


def test_detect_reset_active_low_by_name():
    ports = [Port("clk", "input", 1), Port("rst_n", "input", 1)]
    tb = render_testbench("m", ports, BringUpConfig(cycles=4))
    # Active-low reset asserts with 0 and deasserts with 1.
    assert "rst_n = 1'b0;" in tb
    assert "rst_n = 1'b1;" in tb


def test_render_testbench_structure():
    tb = render_testbench("counter", COUNTER_PORTS, BringUpConfig(cycles=16))
    assert "module tb;" in tb
    assert "counter dut (.clk(clk), .rst(rst), .count(count));" in tb
    assert "always #5 clk = ~clk;" in tb
    assert "$dumpfile" in tb
    assert "repeat (16) @(posedge clk);" in tb
    assert "VFPGA_DONE" in tb
    assert "VFPGA_OUT count" in tb


def test_render_testbench_drives_data_inputs():
    tb = render_testbench("mac", MAC_PORTS, BringUpConfig(cycles=8))
    # Data inputs get deterministic stimulus; clk/rst do not.
    assert "a <= a + 8'd1;" in tb
    assert "c <= c + 16'd1;" in tb
    assert "clk <= clk" not in tb


def test_render_testbench_requires_clock():
    ports = [Port("a", "input", 8), Port("y", "output", 8)]
    with pytest.raises(ValueError):
        render_testbench("comb", ports, BringUpConfig())


def test_parse_outputs():
    log = "VFPGA_DONE cycles=20\nVFPGA_OUT count=19 (0x13)\nVFPGA_OUT y=42 (0x2a)\n"
    outs = _parse_outputs(log)
    assert outs["count"] == "19 (0x13)"
    assert outs["y"] == "42 (0x2a)"


_TOOLS = all(shutil.which(t) for t in ("yosys", "iverilog", "vvp"))


@pytest.mark.skipif(not _TOOLS, reason="requires yosys, iverilog, vvp")
def test_bringup_counter_end_to_end(tmp_path):
    rtl = tmp_path / "counter.v"
    rtl.write_text(
        "module counter(input clk, input rst, output reg [7:0] count);\n"
        "  always @(posedge clk) if (rst) count <= 0; else count <= count + 1;\n"
        "endmodule\n"
    )
    result = bringup(
        rtl=str(rtl), top="counter", cycles=20, workdir=tmp_path / "bu"
    )
    assert result.synthesized
    assert result.compiled
    assert result.ran
    assert result.success
    assert not result.timed_out
    assert result.vcd_path is not None
    assert "count" in result.outputs
