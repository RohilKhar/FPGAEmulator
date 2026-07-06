from fpgaforge.diagnostics import Diagnostic, errors, extract, warnings


def test_extract_yosys_error_with_tool_attribution():
    log = (
        "$ yosys -q -s synth.ys\n"
        "ERROR: Module `\\bad' referenced in module `\\top' is not part of the design.\n"
        "[exit 1]\n"
    )
    diags = extract(log)
    assert len(diags) == 1
    d = diags[0]
    assert d.severity == "error"
    assert d.tool == "yosys"
    assert "not part of the design" in d.message


def test_extract_iverilog_fileline():
    log = (
        "$ iverilog -g2012 -o sim.vvp mapped.v tb.v\n"
        "tb.v:12: error: Unknown module type: counterX\n"
        "tb.v:5: warning: implicit definition of wire 'foo'.\n"
    )
    diags = extract(log)
    errs = [d for d in diags if d.severity == "error"]
    warns = [d for d in diags if d.severity == "warning"]
    assert errs[0].location == "tb.v:12"
    assert errs[0].tool == "iverilog"
    assert "Unknown module type" in errs[0].message
    assert warns[0].location == "tb.v:5"


def test_nextpnr_timing_error_reclassified_as_warning():
    log = (
        "$ nextpnr-ice40 --hx8k\n"
        "ERROR: Max frequency for clock 'clk': 80.00 MHz (FAIL at 100.00 MHz)\n"
    )
    diags = extract(log)
    assert len(diags) == 1
    assert diags[0].severity == "warning"  # timing miss, not a crash


def test_dedup_and_helpers():
    log = "ERROR: same problem\nERROR: same problem\nWarning: heads up\n"
    assert len(extract(log)) == 2
    assert len(errors(log)) == 1
    assert len(warnings(log)) == 1


def test_format_includes_tool_and_location():
    d = Diagnostic("error", "syntax error", tool="iverilog", location="tb.v:9")
    assert d.format() == "[error] iverilog: tb.v:9: syntax error"


def test_no_false_positives_on_clean_log():
    log = "$ yosys -q -s synth.ys\nInfo: all good\n[exit 0]\n"
    assert extract(log) == []
