import shutil

import pytest

from fpgaforge.timing import parse_critical_paths, signoff

SAMPLE_LOG = """\
Info: Critical path report for clock 'clk$SB_IO_IN_$glb_clk' (posedge -> posedge):
Info:       type curr  total name
Info:   clk-to-q  1.39  1.39 Source count_SB_DFFSR_Q_7_D_SB_LUT4_O_LC.O
Info:    routing  1.76  3.15 Net count[0]$SB_IO_OUT (16,1) -> (15,1)
Info:                          Sink $nextpnr_ICESTORM_LC_0.I1
Info:      logic  0.68  3.83 Source $nextpnr_ICESTORM_LC_0.COUT
Info:    routing  0.00  3.83 Net $nextpnr_ICESTORM_LC_0$O (15,1) -> (15,1)
Info:      logic  0.28  4.10 Source count_SB_DFFSR_Q_6_D_SB_LUT4_O_LC.COUT
Info:      setup  0.34  4.44 Setup count_SB_DFFSR_Q_6_D_SB_LUT4_O_LC.I3
Info: Max frequency for clock 'clk$SB_IO_IN_$glb_clk': 143.27 MHz (PASS at 100.00 MHz)
"""


def test_parse_critical_paths_basic():
    paths = parse_critical_paths(SAMPLE_LOG)
    assert len(paths) == 1
    p = paths[0]
    assert p.clock.startswith("clk")
    assert len(p.stages) == 6
    assert abs(p.total_ns - 4.44) < 1e-6


def test_logic_vs_routing_breakdown():
    p = parse_critical_paths(SAMPLE_LOG)[0]
    # clk-to-q 1.39 + logic 0.68 + logic 0.28 = 2.35 logic; routing 1.76 + 0 = 1.76
    assert abs(p.logic_ns - 2.35) < 1e-6
    assert abs(p.routing_ns - 1.76) < 1e-6
    assert p.n_logic_stages == 3


def test_parse_handles_no_report():
    assert parse_critical_paths("no timing here\n") == []


def test_parse_multiple_paths():
    log = SAMPLE_LOG + (
        "Info: Critical path report for cross-domain path 'a' -> '<async>':\n"
        "Info:   clk-to-q  0.54  0.54 Source foo.O\n"
        "Info:    routing  3.11  3.65 Net bar\n"
    )
    paths = parse_critical_paths(log)
    assert len(paths) == 2
    assert "cross-domain" in paths[1].clock


_TOOLS = all(shutil.which(t) for t in ("yosys", "nextpnr-ice40"))


@pytest.mark.skipif(not _TOOLS, reason="requires yosys + nextpnr-ice40")
def test_signoff_end_to_end(tmp_path):
    rtl = tmp_path / "counter.v"
    rtl.write_text(
        "module counter(input clk, input rst, output reg [7:0] count);\n"
        "  always @(posedge clk) if (rst) count <= 0; else count <= count + 1;\n"
        "endmodule\n"
    )
    report = signoff(str(rtl), top="counter", clock_ns=10, workdir=tmp_path / "t")
    assert report.routed_ok
    assert report.fmax_mhz > 0
    assert report.meets_timing        # counter easily beats 100 MHz
    assert report.worst_path is not None
    assert report.worst_path.total_ns > 0
    assert report.sdf_path is not None  # SDF artifact emitted
