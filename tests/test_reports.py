from pathlib import Path

from fpgaforge import reports

DATA = Path(__file__).parent / "data"


def test_parse_nextpnr_log():
    log = (DATA / "nextpnr_sample.log").read_text()
    parsed = reports.parse_nextpnr_log(log)
    assert parsed["fmax_mhz"] == 132.45
    assert parsed["target_freq_mhz"] == 100.0
    assert parsed["routed_ok"] is True
    assert parsed["util"]["ICESTORM_LC"] == 42
    assert parsed["util"]["ICESTORM_DSP"] == 1


def test_parse_nextpnr_log_failure():
    log = "ERROR: Failed to route\n"
    parsed = reports.parse_nextpnr_log(log)
    assert parsed["routed_ok"] is False
    assert parsed["fmax_mhz"] == 0.0


def test_parse_yosys_stat_text():
    text = (DATA / "yosys_stat_sample.txt").read_text()
    counts = reports.parse_yosys_stat_text(text)
    assert counts["SB_LUT4"] == 40
    assert counts["SB_DFF"] == 32
    assert counts["SB_CARRY"] == 16
    assert counts["SB_MAC16"] == 1


def test_parse_yosys_stat_json():
    stat = {"design": {"num_cells_by_type": {"SB_LUT4": 12, "SB_DFF": 8}}}
    counts = reports.parse_yosys_stat_json(stat)
    assert counts == {"SB_LUT4": 12, "SB_DFF": 8}


def test_build_metrics_combines_sources():
    log = (DATA / "nextpnr_sample.log").read_text()
    counts = reports.parse_yosys_stat_text((DATA / "yosys_stat_sample.txt").read_text())
    m = reports.build_metrics(nextpnr_log=log, yosys_stat=counts)
    assert m.fmax_mhz == 132.45
    assert m.target_freq_mhz == 100.0
    assert m.luts == 42  # from nextpnr utilisation
    assert m.ffs == 32   # from yosys stat
    assert m.carries == 16
    assert m.dsp == 1
    assert m.routed_ok is True
    assert m.meets_timing is True
    assert abs(m.crit_path_ns - 1000.0 / 132.45) < 1e-6
