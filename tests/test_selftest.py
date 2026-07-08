"""Tests for the on-FPGA self-test (BIST) generator.

The self-test is the platform's answer to the one gap no simulator can close:
silicon defects and environment. We validate that the generated harness (a)
reproduces the simulator-predicted golden signature, and (b) actually catches
a modeled silicon defect (stuck output bit) by dropping test_pass.
"""

import re
import shutil
import subprocess

import pytest

from fpgaforge.selftest import generate_selftest

_TOOLS = all(shutil.which(t) for t in ("yosys", "iverilog", "vvp"))
pytestmark = pytest.mark.skipif(not _TOOLS,
                                reason="requires yosys + iverilog + vvp")


def test_generates_and_validates_harness(tmp_path):
    r = generate_selftest("examples/counter.v", "counter", cycles=128,
                          workdir=tmp_path)
    assert r.error is None, r.error
    assert r.validated is True
    assert re.fullmatch(r"[0-9a-f]{16}", r.golden_signature)
    assert r.clock_port == "clk" and r.reset_port == "rst"
    text = (tmp_path / "counter_selftest.v").read_text()
    assert f"64'h{r.golden_signature}" in text          # golden baked in
    assert "test_pass" in text and "test_done" in text


def test_defective_silicon_is_caught(tmp_path):
    r = generate_selftest("examples/counter.v", "counter", cycles=128,
                          workdir=tmp_path)
    assert r.validated

    # Model a silicon defect: output bit 3 stuck at 0. Same harness, same
    # golden signature -> the self-test must fail.
    defect = tmp_path / "counter_defect.v"
    defect.write_text(
        "module counter(input clk, input rst, output wire [7:0] count);\n"
        "  reg [7:0] c;\n"
        "  always @(posedge clk) if (rst) c <= 0; else c <= c + 1;\n"
        "  assign count = c & 8'hF7;  // stuck-at-0 defect on bit 3\n"
        "endmodule\n"
    )
    vvp = tmp_path / "defect.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), str(defect),
         str(tmp_path / "counter_selftest.v"), str(tmp_path / "selftest_tb.v")],
        check=True, capture_output=True, text=True,
    )
    out = subprocess.run(["vvp", str(vvp)], capture_output=True,
                         text=True).stdout
    assert re.search(r"PASS\s+0", out), out


def test_unclocked_design_is_rejected(tmp_path):
    comb = tmp_path / "comb.v"
    comb.write_text("module comb(input a, input b, output y);\n"
                    "  assign y = a & b;\nendmodule\n")
    r = generate_selftest(str(comb), "comb", workdir=tmp_path / "w")
    assert r.error is not None and "clock" in r.error


def test_uninitialized_state_is_rejected(tmp_path):
    # State that reset never initializes -> X in the golden signature -> the
    # generator must refuse rather than emit an unsound self-test.
    bad = tmp_path / "bad.v"
    bad.write_text(
        "module bad(input clk, input rst, output reg [3:0] q);\n"
        "  reg [3:0] hidden;  // never reset\n"
        "  always @(posedge clk) begin\n"
        "    hidden <= hidden + 1;\n"
        "    q <= hidden;\n"
        "  end\nendmodule\n"
    )
    r = generate_selftest(str(bad), "bad", cycles=32, workdir=tmp_path / "w")
    assert r.error is not None
    assert "X" in r.error or "reset" in r.error
