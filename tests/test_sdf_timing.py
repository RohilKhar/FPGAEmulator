"""Tests for the SDF static-timing engine and timing-accurate emulation."""

import shutil

import pytest

from fpgaforge.sdf_timing import parse_sdf, longest_paths, analyze_sdf
from fpgaforge.emulator.timing_emu import timing_emulate


_MINI_SDF = """
(DELAYFILE
  (TIMESCALE 1ps)
  (CELL (CELLTYPE "top") (INSTANCE )
    (DELAY (ABSOLUTE
      (INTERCONNECT ff0/O lut/I0 (500:500:500) (500:500:500))
      (INTERCONNECT lut/O ff1/I2 (400:400:400) (400:400:400))
    )))
  (CELL (CELLTYPE "ICESTORM_LC") (INSTANCE ff0)
    (DELAY (ABSOLUTE
      (IOPATH CLK O (1000:1000:1000) (1000:1000:1000))
    )))
  (CELL (CELLTYPE "ICESTORM_LC") (INSTANCE lut)
    (DELAY (ABSOLUTE
      (IOPATH I0 O (600:600:600) (600:600:600))
    )))
  (CELL (CELLTYPE "ICESTORM_LC") (INSTANCE ff1)
    (DELAY (ABSOLUTE
      (IOPATH CLK O (1000:1000:1000) (1000:1000:1000))
    ))
    (TIMINGCHECK
      (SETUPHOLD (posedge I2) (posedge CLK) (300:300:300) (0:0:0))
    ))
)
"""


def test_parse_sdf_extracts_delays():
    sdf = parse_sdf(_MINI_SDF)
    assert sdf.timescale_ps == 1.0
    assert len(sdf.interconnects) == 2
    ff1 = [c for c in sdf.cells if c.instance == "ff1"][0]
    assert ff1.setup["I2"] == 300.0
    lut = [c for c in sdf.cells if c.instance == "lut"][0]
    assert ("I0", "O", 600.0) in lut.iopaths


def test_longest_path_period_and_endpoint():
    # clk-to-Q 1000 + route 500 + LUT 600 + route 400 = 2500 data, + 300 setup = 2800 ps
    r = longest_paths(parse_sdf(_MINI_SDF))
    assert r.min_period_ps == pytest.approx(2800.0)
    assert r.fmax_mhz == pytest.approx(1e6 / 2800.0, rel=1e-3)
    assert r.worst.endpoint == "ff1/I2"
    assert r.n_launch == 2          # ff0 and ff1 both have clk-to-Q
    assert "ff0/O" in r.worst.path


def test_slack_and_settling_at_clock():
    r = longest_paths(parse_sdf(_MINI_SDF))
    fmax = r.fmax_mhz
    assert r.settles_at(fmax * 0.9)          # slower clock: settles
    assert not r.settles_at(fmax * 1.1)      # faster clock: fails setup
    assert r.slack_ns_at(fmax * 0.5) > 0


def test_clk_to_q_launch_not_traversed_as_data():
    # A flop's own clk-to-Q must be a launch source, not counted as a data edge
    # into it (which would create a false loop / inflate the period).
    r = longest_paths(parse_sdf(_MINI_SDF))
    # Data arrival at the endpoint excludes a second clk-to-Q hop.
    assert r.worst.data_arrival_ps == pytest.approx(2500.0)


def test_single_clock_has_one_domain():
    r = longest_paths(parse_sdf(_MINI_SDF))
    assert not r.multi_clock
    assert len(r.domains) == 1
    assert r.worst.domain == "clk"
    assert not r.cross_domain


# Two clock domains (clkA feeds ff0/ff1, clkB feeds ff2/ff3), an intra-domain
# path in each, and one path that crosses from clkA into clkB.
_MULTI_SDF = """
(DELAYFILE
  (TIMESCALE 1ps)
  (CELL (CELLTYPE "top") (INSTANCE )
    (DELAY (ABSOLUTE
      (INTERCONNECT clkA ff0/CLK (0:0:0) (0:0:0))
      (INTERCONNECT clkA ff1/CLK (0:0:0) (0:0:0))
      (INTERCONNECT clkB ff2/CLK (0:0:0) (0:0:0))
      (INTERCONNECT clkB ff3/CLK (0:0:0) (0:0:0))
      (INTERCONNECT ff0/O ff1/I2 (500:500:500) (500:500:500))
      (INTERCONNECT ff2/O ff3/I2 (900:900:900) (900:900:900))
      (INTERCONNECT ff0/O ff3/I3 (700:700:700) (700:700:700))
    )))
  (CELL (CELLTYPE "ICESTORM_LC") (INSTANCE ff0)
    (DELAY (ABSOLUTE (IOPATH CLK O (1000:1000:1000) (1000:1000:1000)))))
  (CELL (CELLTYPE "ICESTORM_LC") (INSTANCE ff2)
    (DELAY (ABSOLUTE (IOPATH CLK O (1000:1000:1000) (1000:1000:1000)))))
  (CELL (CELLTYPE "ICESTORM_LC") (INSTANCE ff1)
    (DELAY (ABSOLUTE (IOPATH CLK O (1000:1000:1000) (1000:1000:1000))))
    (TIMINGCHECK (SETUPHOLD (posedge I2) (posedge CLK) (300:300:300) (0:0:0))))
  (CELL (CELLTYPE "ICESTORM_LC") (INSTANCE ff3)
    (DELAY (ABSOLUTE (IOPATH CLK O (1000:1000:1000) (1000:1000:1000))))
    (TIMINGCHECK
      (SETUPHOLD (posedge I2) (posedge CLK) (300:300:300) (0:0:0))
      (SETUPHOLD (posedge I3) (posedge CLK) (300:300:300) (0:0:0))))
)
"""


def test_multi_clock_domains_separated():
    r = longest_paths(parse_sdf(_MULTI_SDF))
    assert r.multi_clock
    assert set(r.domains) == {"clkA", "clkB"}

    # Domain A: 1000 clk-to-Q + 500 route + 300 setup = 1800 ps.
    assert r.domains["clkA"].min_period_ps == pytest.approx(1800.0)
    # Domain B: 1000 + 900 + 300 = 2200 ps -> the binding (slowest) domain.
    assert r.domains["clkB"].min_period_ps == pytest.approx(2200.0)
    assert r.min_period_ps == pytest.approx(2200.0)
    assert r.worst.domain == "clkB"


def test_cross_domain_path_reported_not_in_fmax():
    r = longest_paths(parse_sdf(_MULTI_SDF))
    # ff0 (clkA) -> ff3/I3 (clkB) is a crossing, not a clkB setup constraint.
    assert any(
        x.launch_domain == "clkA" and x.capture_domain == "clkB"
        and x.endpoint == "ff3/I3"
        for x in r.cross_domain
    )
    # clkB Fmax must come from the intra-B path (ff3/I2), not the crossing.
    assert r.domains["clkB"].worst.endpoint == "ff3/I2"


# ------------------------- tool-gated integration ------------------- #
_TOOLS = ["yosys", "nextpnr-ice40", "icepack", "iceunpack", "icebox_vlog",
          "iverilog", "vvp"]


@pytest.mark.skipif(
    any(shutil.which(t) is None for t in _TOOLS),
    reason="requires the full open-source iCE40 toolchain",
)
def test_timing_emulate_pass_and_setup_violation():
    slow = timing_emulate("examples/counter.v", "counter", clock_mhz=40.0,
                          cycles=32, workdir=".runs/test_temu_slow")
    assert slow.error is None, slow.error
    assert slow.emulated_fmax_mhz > 0
    assert slow.settles
    assert slow.functional_match
    assert slow.verdict == "TIMING-ACCURATE PASS"

    fast = timing_emulate("examples/counter.v", "counter",
                          clock_mhz=slow.emulated_fmax_mhz * 1.5,
                          cycles=32, workdir=".runs/test_temu_fast")
    assert fast.error is None, fast.error
    assert not fast.settles
    assert fast.verdict == "SETUP VIOLATION"
    assert fast.worst_endpoint
