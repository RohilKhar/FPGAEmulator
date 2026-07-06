"""Tests for the ECP5 backend, target routing, and Trellis metric parsing."""

import shutil
from pathlib import Path

import pytest

from fpgaforge.backends.ecp5 import Ecp5Backend
from fpgaforge.backends.base import Design, FlowOptions
from fpgaforge.backends.mock import MockBackend
from fpgaforge.optimizer import backend_for_target
from fpgaforge.reports import build_metrics


_ECP5_LOG = """
Info: Max frequency for clock 'clk': 85.32 MHz (PASS at 50.00 MHz)
Info: Device utilisation:
Info:            TRELLIS_COMB:   420/24288     1%
Info:              TRELLIS_FF:   200/24288     0%
Info:                  DP16KD:     2/   56     3%
Info:              MULT18X18D:     1/   28     3%
Info: Program finished normally.
"""


def test_build_metrics_parses_trellis_resources():
    m = build_metrics(nextpnr_log=_ECP5_LOG, target_freq_mhz=50.0)
    assert m.fmax_mhz == pytest.approx(85.32)
    assert m.routed_ok
    assert m.luts == 420
    assert m.ffs == 200
    assert m.bram == 2
    assert m.dsp == 1
    assert m.meets_timing


def test_backend_for_target_routes_by_family():
    # iCE40 targets never route to the ECP5 backend.
    assert backend_for_target("ice40_up5k").name in ("ice40", "mock")
    # ECP5 target: real backend if tools present, else mock fallback.
    b = backend_for_target("ecp5_45k")
    assert b.name in ("ecp5", "mock")


def test_ecp5_backend_metadata_and_unsupported_target():
    be = Ecp5Backend()
    assert be.name == "ecp5"
    d = Design(rtl_files=("examples/counter.v",), top="counter", target="ice40_up5k")
    res = be.run(d, FlowOptions(), Path(".runs/test_ecp5_bad"))
    assert res.success is False
    assert "unsupported ecp5 target" in (res.error or "")


@pytest.mark.skipif(shutil.which("yosys") is None, reason="requires yosys")
def test_ecp5_synthesis_runs_even_without_pnr(tmp_path):
    # yosys synth_ecp5 should map the design and populate features, regardless of
    # whether nextpnr-ecp5 is installed (P&R then fails cleanly).
    be = Ecp5Backend()
    d = Design(rtl_files=("examples/counter.v",), top="counter", target="ecp5_45k")
    res = be.run(d, FlowOptions(), tmp_path / "run")
    # Synthesis populated features from the mapped netlist.
    assert res.features.get("num_luts", 0) > 0 or res.features.get("num_ffs", 0) > 0
    if not shutil.which("nextpnr-ecp5"):
        assert res.success is False
