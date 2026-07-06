import shutil

import pytest

from fpgaforge.readiness import (
    AT_RISK,
    BLOCKED,
    READY,
    assess,
    count_clock_domains,
    evaluate,
    score_and_verdict,
)


def _facts(**overrides):
    base = dict(
        synthesized=True,
        routed_ok=True,
        fmax_mhz=200.0,
        target_mhz=100.0,
        luts=100,
        lut_capacity=5280,
        io_count=10,
        io_capacity=39,
        bringup_status="up",
        synth_log="",
        rtl_text="module m(input clk); always @(posedge clk) begin end endmodule",
    )
    base.update(overrides)
    return base


def _status(checks, name):
    return next(c.status for c in checks if c.name == name)


def test_clean_design_is_ready():
    checks = evaluate(**_facts())
    score, verdict = score_and_verdict(checks)
    assert verdict == READY
    assert score == 100
    assert all(c.status == "pass" for c in checks)


def test_missed_timing_blocks():
    checks = evaluate(**_facts(fmax_mhz=80.0, target_mhz=100.0))
    assert _status(checks, "timing") == "fail"
    score, verdict = score_and_verdict(checks)
    assert verdict == BLOCKED
    assert score <= 40


def test_tight_timing_is_at_risk():
    checks = evaluate(**_facts(fmax_mhz=105.0, target_mhz=100.0))
    assert _status(checks, "timing") == "warn"
    _, verdict = score_and_verdict(checks)
    assert verdict == AT_RISK


def test_synthesis_failure_short_circuits():
    checks = evaluate(**_facts(synthesized=False))
    assert len(checks) == 1
    assert checks[0].status == "fail"
    _, verdict = score_and_verdict(checks)
    assert verdict == BLOCKED


def test_routing_failure_blocks_and_recommends():
    checks = evaluate(**_facts(routed_ok=False, fmax_mhz=0.0))
    assert _status(checks, "place_and_route") == "fail"
    pr = next(c for c in checks if c.name == "place_and_route")
    assert pr.recommendation is not None


def test_high_utilization_warns():
    checks = evaluate(**_facts(luts=5000, lut_capacity=5280))
    assert _status(checks, "resource_headroom") == "warn"


def test_io_overflow_fails():
    checks = evaluate(**_facts(io_count=98, io_capacity=39))
    assert _status(checks, "io_fit") == "fail"


def test_equivalence_skipped_by_default():
    checks = evaluate(**_facts())
    assert not any(c.name == "bitstream_equivalence" for c in checks)


def test_equivalence_proved_all_is_pass():
    checks = evaluate(**_facts(bitstream_equivalence="proved_all"))
    eq = next(c for c in checks if c.name == "bitstream_equivalence")
    assert eq.status == "pass" and "all time" in eq.message
    assert score_and_verdict(checks)[1] == READY


def test_equivalence_differ_blocks():
    checks = evaluate(**_facts(bitstream_equivalence="differ", equivalence_detail="in_a=5"))
    eq = next(c for c in checks if c.name == "bitstream_equivalence")
    assert eq.status == "fail" and "in_a=5" in eq.message
    assert score_and_verdict(checks)[1] == BLOCKED


def test_equivalence_inconclusive_warns():
    checks = evaluate(**_facts(bitstream_equivalence="inconclusive"))
    assert _status(checks, "bitstream_equivalence") == "warn"


def test_equivalence_verified_is_pass():
    checks = evaluate(**_facts(bitstream_equivalence="verified",
                               equivalence_detail="over 64 cycles (stimulus)"))
    eq = next(c for c in checks if c.name == "bitstream_equivalence")
    assert eq.status == "pass" and "64 cycles" in eq.message


def test_latch_and_loop_detection():
    checks = evaluate(**_facts(synth_log="Warning: inferring latch for signal q"))
    assert _status(checks, "latch_free") == "warn"
    checks = evaluate(**_facts(synth_log="ERROR: found logic loop in module"))
    assert _status(checks, "no_comb_loops") == "fail"


def test_bringup_down_fails():
    checks = evaluate(**_facts(bringup_status="down"))
    assert _status(checks, "functional_bringup") == "fail"


def test_bringup_skipped_warns():
    checks = evaluate(**_facts(bringup_status="skipped"))
    assert _status(checks, "functional_bringup") == "warn"


def test_count_clock_domains():
    single = "always @(posedge clk) q<=d; always @(posedge clk or negedge rst_n) x<=y;"
    assert count_clock_domains(single) == 1
    multi = "always @(posedge clk_a) q<=d; always @(posedge clk_b) x<=y;"
    assert count_clock_domains(multi) == 2


def test_multi_clock_warns():
    rtl = "always @(posedge clk_a) q<=d; always @(posedge clk_b) x<=y;"
    checks = evaluate(**_facts(rtl_text=rtl))
    assert _status(checks, "clock_domains") == "warn"


_TOOLS = all(shutil.which(t) for t in ("yosys", "nextpnr-ice40", "iverilog", "vvp"))


@pytest.mark.skipif(not _TOOLS, reason="requires full open-source flow")
def test_assess_end_to_end_counter(tmp_path):
    rtl = tmp_path / "counter.v"
    rtl.write_text(
        "module counter(input clk, input rst, output reg [7:0] count);\n"
        "  always @(posedge clk) if (rst) count <= 0; else count <= count + 1;\n"
        "endmodule\n"
    )
    report = assess(str(rtl), top="counter", clock_ns=10, cycles=20, iterations=3)
    assert report.verdict in (READY, AT_RISK)
    assert report.score > 0
    assert any(c.name == "functional_bringup" and c.status == "pass" for c in report.checks)
