"""Unit tests for stimulus coverage + divergence reporting (no tools needed)."""

from fpgaforge.emulator.emulator import (
    VerificationResult,
    _divergence_window,
    _input_coverage,
)
from fpgaforge.virtual.board import Port


def _port(name, width, direction="input"):
    return Port(name=name, direction=direction, width=width)


def test_input_coverage_full_toggle():
    driven = [_port("a", 4)]
    # 0 then 15 -> every bit seen at both 0 and 1.
    stim = [["STIM 0 a=0", "STIM 1 a=15"]]
    cov, toggled, total = _input_coverage(stim, driven)
    assert total == 4
    assert toggled == 4
    assert cov == 1.0


def test_input_coverage_partial():
    driven = [_port("a", 4)]
    # Only ever 0 and 1 -> only bit 0 toggles.
    stim = [["STIM 0 a=0", "STIM 1 a=1"]]
    cov, toggled, total = _input_coverage(stim, driven)
    assert toggled == 1
    assert total == 4
    assert cov == 0.25


def test_input_coverage_no_driven_inputs_is_full():
    cov, toggled, total = _input_coverage([], [])
    assert cov == 1.0
    assert total == 0


def test_confidence_penalizes_poor_stimulus():
    common = dict(
        design_id="d",
        matches=True,
        total_compared=1000,
        seeds=[1, 2, 3, 4],
        toggle_coverage=1.0,
        toggled_bits=8,
        total_bits=8,
    )
    well_driven = VerificationResult(
        driven_inputs=2, input_coverage=1.0, **common
    )
    poorly_driven = VerificationResult(
        driven_inputs=2, input_coverage=0.1, **common
    )
    assert well_driven.confidence > poorly_driven.confidence


def test_confidence_not_penalized_without_inputs():
    # A pure sequential design (no driven inputs) is not docked for stimulus.
    res = VerificationResult(
        design_id="d",
        matches=True,
        total_compared=1000,
        seeds=[1, 2, 3, 4],
        toggle_coverage=1.0,
        toggled_bits=8,
        total_bits=8,
        driven_inputs=0,
        input_coverage=0.0,  # ignored because nothing was driven
    )
    assert res.confidence > 0.8


def test_divergence_window_marks_failing_cycle():
    rtl = [f"CYC {i} q={i}" for i in range(10)]
    bit = [f"CYC {i} q={i}" for i in range(10)]
    bit[5] = "CYC 5 q=99"  # inject a divergence at cycle 5
    stim = [f"STIM {i} a={i}" for i in range(10)]
    window = _divergence_window(rtl, bit, stim, cycle=5, radius=2)
    text = "\n".join(window)
    assert "DIVERGES" in text
    assert "q=99" in text
    # Includes the diverging field callout and the stimulus for that cycle.
    assert "dif: q" in text
    assert "in : a=5" in text
