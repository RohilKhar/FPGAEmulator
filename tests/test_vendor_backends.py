"""Tests for the AMD (Vivado) and Intel (Quartus) vendor backends.

The tools aren't installed in CI, so we test the pure report parsers (with
representative report text) and the graceful behaviour when the tools are
missing. The reconstructor abstraction and its capability gating are covered too.
"""

from pathlib import Path

from fpgaforge.backends.base import Design, FlowOptions
from fpgaforge.backends.vivado import (
    VivadoBackend, parse_utilization, parse_wns_ns, build_vivado_metrics,
)
from fpgaforge.backends.quartus import (
    QuartusBackend, parse_fit_resources, parse_sta_fmax_mhz,
    parse_total_power_mw, build_quartus_metrics,
)
from fpgaforge.emulator.reconstruct import (
    reconstructor_for, IceStormReconstructor, NoReconstruction,
)


_VIVADO_UTIL = """
+-------------------------+------+-------+-----------+-------+
|        Site Type        | Used | Fixed | Available | Util% |
+-------------------------+------+-------+-----------+-------+
| Slice LUTs*             |  312 |     0 |     20800 |  1.50 |
| Slice Registers         |  205 |     0 |     41600 |  0.49 |
| Block RAM Tile          |    2 |     0 |        50 |  4.00 |
| DSPs                    |    4 |     0 |        90 |  4.44 |
"""

_VIVADO_TIMING = """
Design Timing Summary
---------------------
    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints
    -------      -------  ---------------------  -------------------
      2.000        0.000                      0                  512
"""


def test_vivado_parsers():
    u = parse_utilization(_VIVADO_UTIL)
    assert u == {"luts": 312, "ffs": 205, "bram": 2, "dsp": 4}
    assert parse_wns_ns(_VIVADO_TIMING) == 2.0
    assert parse_wns_ns("no timing here") is None


def test_vivado_metrics_fmax_from_wns():
    m = build_vivado_metrics(_VIVADO_UTIL, _VIVADO_TIMING,
                             target_period_ns=10.0, target_freq_mhz=100.0)
    # achieved period = 10 - 2 = 8 ns -> 125 MHz
    assert round(m.fmax_mhz, 1) == 125.0
    assert m.luts == 312 and m.dsp == 4
    assert m.routed_ok is True


def test_vivado_failing_timing_is_slower():
    timing = "WNS(ns)\n-------\n  -1.000"
    m = build_vivado_metrics(_VIVADO_UTIL, timing, 10.0, 100.0)
    # negative slack -> achieved period 11 ns -> ~90.9 MHz, below target
    assert m.fmax_mhz < 100.0
    assert m.meets_timing is False


def test_vivado_unavailable_returns_clean_error(tmp_path):
    be = VivadoBackend(vivado="definitely-not-vivado")
    assert be.is_available() is False
    d = Design(rtl_files=("examples/counter.v",), top="counter", target="xc7a35t")
    r = be.run(d, FlowOptions(), tmp_path)
    assert r.success is False
    assert "vivado" in (r.error or "").lower()


def test_vivado_rejects_non_amd_target(tmp_path):
    be = VivadoBackend()
    d = Design(rtl_files=("examples/counter.v",), top="counter", target="ice40_up5k")
    r = be.run(d, FlowOptions(), tmp_path)
    assert r.success is False
    assert "not an AMD" in (r.error or "")


_QUARTUS_FIT = """
; Total logic elements ; 1,234 ; ; 24,624 ;
; Total registers      ; 567   ;
; Total DSP Blocks     ; 4     ;
; Total RAM Blocks     ; 3     ;
"""

_QUARTUS_STA = """
; Fmax Summary ;
; 175.44 MHz ; 160.20 MHz ; clk  ; ;
; 250.00 MHz ; 245.00 MHz ; clk2 ; ;
"""

_QUARTUS_POW = "; Total Thermal Power Dissipation ; 123.4 mW ;"


def test_quartus_parsers():
    r = parse_fit_resources(_QUARTUS_FIT)
    assert r == {"luts": 1234, "ffs": 567, "bram": 3, "dsp": 4}
    # min restricted Fmax across clocks
    assert parse_sta_fmax_mhz(_QUARTUS_STA) == 160.2
    assert parse_total_power_mw(_QUARTUS_POW) == 123.4


def test_quartus_metrics():
    m = build_quartus_metrics(_QUARTUS_FIT, _QUARTUS_STA, target_freq_mhz=100.0)
    assert m.fmax_mhz == 160.2
    assert m.luts == 1234 and m.ffs == 567
    assert m.routed_ok is True


def test_quartus_unavailable_returns_clean_error(tmp_path):
    be = QuartusBackend(quartus_map="definitely-not-quartus")
    assert be.is_available() is False
    d = Design(rtl_files=("examples/counter.v",), top="counter",
               target="cyclonev_5csema5")
    r = be.run(d, FlowOptions(), tmp_path)
    assert r.success is False
    assert "quartus" in (r.error or "").lower()


# ------------------------- reconstructor abstraction --------------------- #
def test_reconstructor_selection():
    assert isinstance(reconstructor_for("ice40_up5k"), IceStormReconstructor)
    assert reconstructor_for("ice40_up5k").available is True
    # Truly locked silicon (encrypted/undocumented bitstream) -> NoReconstruction.
    for target in ("cyclonev_5csema5", "ecp5_85k", "xczu3eg"):
        r = reconstructor_for(target)
        assert isinstance(r, NoReconstruction)
        assert r.available is False
        assert "proprietary" in r.why_unavailable(target).lower() \
            or "no open bitstream" in r.why_unavailable(target).lower() \
            or "impossible" in r.why_unavailable(target).lower()


def test_gowin_backend_registry_and_gating(tmp_path):
    from fpgaforge.backends.gowin import _DEVICES as GOWIN_DEVICES, GowinBackend
    from fpgaforge import devices as dv

    assert set(GOWIN_DEVICES) == {d.target for d in dv.by_backend("gowin")}
    be = GowinBackend(nextpnr="definitely-not-nextpnr")
    assert be.is_available() is False
    d = Design(rtl_files=("examples/counter.v",), top="counter",
               target="gowin_gw1n9")
    r = be.run(d, FlowOptions(), tmp_path)
    # yosys synth_gowin may succeed, but P&R must fail cleanly without nextpnr.
    assert r.success is False
    assert r.error is not None


def test_prove_on_locked_vendor_target_degrades_gracefully(tmp_path):
    from fpgaforge.emulator.emulator import Emulator

    em = Emulator()
    d = Design(rtl_files=("examples/counter.v",), top="counter", target="xczu3eg")
    r = em.prove_equivalence(d, workdir=tmp_path)
    assert r.equivalent is None
    assert "proprietary" in (r.error or "").lower()
