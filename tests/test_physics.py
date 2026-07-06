"""Unit tests for the physics sign-off models (pure math, no tools)."""

import math

from fpgaforge.physics import (
    ICE40_SG48,
    ICE40_UP5K_PVT,
    InterfaceBudget,
    Net,
    PackageModel,
    analyze_interface,
    analyze_net,
    derate_fmax,
    physical_signoff,
)
from fpgaforge.physics.pvt import PVTConfig
from fpgaforge.physics.signal_integrity import spice_deck


# ------------------------------- PVT -------------------------------- #
def test_pvt_slow_corner_is_guaranteed():
    # If the STA number is already the slow corner, the guaranteed Fmax equals
    # it and every other corner is faster.
    r = derate_fmax(100.0, 50.0, ICE40_UP5K_PVT, sta_corner="slow")
    assert math.isclose(r.guaranteed_fmax_mhz, 100.0, rel_tol=0.02)
    assert r.best_fmax_mhz > r.guaranteed_fmax_mhz
    assert r.meets_worst_case is True
    assert r.worst_corner.process == "slow"
    # Corners span a realistic spread (fast corner well above slow).
    assert r.best_fmax_mhz > 1.5 * r.guaranteed_fmax_mhz


def test_pvt_typical_corner_derates_down():
    # If STA was reported at the typical corner, the guaranteed (slow) number is
    # lower than the STA figure.
    r = derate_fmax(100.0, 50.0, ICE40_UP5K_PVT, sta_corner="typ")
    assert r.guaranteed_fmax_mhz < 100.0
    assert r.guaranteed_fmax_mhz >= 50.0        # still meets target here
    assert r.meets_worst_case is True


def test_pvt_delay_factor_monotonic():
    cfg = PVTConfig()
    slow_hot = cfg.delay_factor("slow", cfg.vdd_min, cfg.temp_max_c)
    typ = cfg.delay_factor("typ", cfg.vdd_nom, cfg.temp_nom_c)
    fast_cold = cfg.delay_factor("fast", cfg.vdd_max, cfg.temp_min_c)
    assert slow_hot > typ > fast_cold           # slower corner -> bigger delay
    # Lower voltage is slower; higher temp is slower.
    assert cfg.delay_factor("typ", cfg.vdd_min, 25) > cfg.delay_factor("typ", cfg.vdd_max, 25)
    assert cfg.delay_factor("typ", 1.2, 85) > cfg.delay_factor("typ", 1.2, 0)


def test_pvt_fails_when_target_above_worst_case():
    r = derate_fmax(60.0, 90.0, ICE40_UP5K_PVT, sta_corner="slow")
    assert r.meets_worst_case is False
    assert r.margin_pct() < 0


# --------------------------- signal integrity ----------------------- #
def test_si_matched_short_net_is_clean():
    net = Net(name="led", drive_impedance_ohm=50, trace_z0_ohm=50,
              trace_len_mm=20, load_pf=8)
    r = analyze_net(net, ICE40_SG48, use_spice=False)
    assert r.overshoot_frac == 0.0              # matched source, no reflection
    assert not r.needs_termination
    assert r.risks == []


def test_si_underterminated_long_net_overshoots():
    net = Net(name="dq", drive_impedance_ohm=20, trace_z0_ohm=50,
              driver_rise_ns=0.3, trace_len_mm=150, load_pf=5)
    r = analyze_net(net, ICE40_SG48, use_spice=False)
    assert r.electrically_long
    # Reflection overshoot = (Z0-Zout)/(Z0+Zout) = 30/70 ~ 0.4286.
    assert math.isclose(r.reflection_overshoot_frac, 30.0 / 70.0, rel_tol=1e-6)
    assert r.needs_termination
    assert any("overshoot" in risk for risk in r.risks)
    assert r.peak_voltage > net.vdd


def test_si_ssn_scales_with_simultaneous_outputs():
    base = Net(name="b", n_simultaneous=1, driver_rise_ns=0.4, load_pf=10)
    many = Net(name="m", n_simultaneous=16, driver_rise_ns=0.4, load_pf=10)
    r1 = analyze_net(base, ICE40_SG48, use_spice=False)
    r16 = analyze_net(many, ICE40_SG48, use_spice=False)
    assert math.isclose(r16.ssn_volts, 16 * r1.ssn_volts, rel_tol=1e-6)
    assert r16.ssn_volts > r1.ssn_volts


def test_si_rise_time_is_rss_of_driver_and_rc():
    net = Net(driver_rise_ns=1.0, drive_impedance_ohm=50, load_pf=10)
    r = analyze_net(net, ICE40_SG48, use_spice=False)
    rc = 2.2 * 50 * (10e-12) * 1e9
    assert math.isclose(r.rise_time_ns, math.hypot(1.0, rc), rel_tol=1e-6)


def test_spice_deck_has_line_and_load():
    deck = spice_deck(Net(name="dq", drive_impedance_ohm=30, trace_z0_ohm=50,
                          trace_len_mm=100, load_pf=5))
    assert "T1 drv 0 load 0 Z0=50" in deck
    assert "Rout src drv 30" in deck
    assert "Cload load 0" in deck
    assert ".tran" in deck


# ------------------------------ interfaces -------------------------- #
def test_interface_closes_with_margin():
    b = InterfaceBudget(name="mem", clock_mhz=100, ddr=True, tco_max_ns=3,
                        tco_min_ns=1, setup_req_ns=0.5, hold_req_ns=0.5,
                        board_skew_ns=0.2, clock_jitter_ns=0.15)
    r = analyze_interface(b)
    assert math.isclose(r.ui_ns, 5.0)           # DDR halves the 10 ns period
    assert r.passes
    assert r.margin_ns > 0
    assert math.isclose(r.setup_margin_ns, r.hold_margin_ns)


def test_interface_fails_when_rate_too_high():
    b = InterfaceBudget(name="fast", clock_mhz=500, ddr=True, tco_max_ns=2,
                        tco_min_ns=0.5, setup_req_ns=0.3, hold_req_ns=0.3,
                        board_skew_ns=0.3, clock_jitter_ns=0.2)
    r = analyze_interface(b)
    assert not r.passes
    assert r.risks


def test_interface_no_eye_when_uncertainty_exceeds_ui():
    b = InterfaceBudget(name="broken", clock_mhz=1000, ddr=True, tco_max_ns=5,
                        tco_min_ns=0, setup_req_ns=0.1, hold_req_ns=0.1)
    r = analyze_interface(b)
    assert r.eye_ns <= 0
    assert any("no eye" in risk for risk in r.risks)


# -------------------------------- IBIS ------------------------------ #
def test_ibis_number_parsing():
    from fpgaforge.physics.ibis import _num, _ramp_slew

    assert math.isclose(_num("3.5nH"), 3.5e-9, rel_tol=1e-9)
    assert math.isclose(_num("0.6pF"), 0.6e-12, rel_tol=1e-9)
    assert _num("50") == 50.0
    assert math.isclose(_num("16.0mA"), 16.0e-3, rel_tol=1e-9)
    assert _num("NA") is None
    # ramp "dV/time" -> V/ns
    assert math.isclose(_ramp_slew("1.98/1.20n"), 1.98 / 1.2, rel_tol=1e-6)


def test_ibis_load_example_model():
    from fpgaforge.physics import load_ibis

    m = load_ibis("examples/ice40_lvcmos33.ibs", "LVCMOS33_8mA")
    assert m.vdd == 3.3
    assert math.isclose(m.l_pkg_nh, 3.5, rel_tol=1e-6)   # typ column from [Package]
    assert math.isclose(m.c_pkg_pf, 0.6, rel_tol=1e-6)
    assert math.isclose(m.r_pkg, 0.2, rel_tol=1e-6)
    # 8 mA-class buffer -> tens of ohms output impedance.
    assert 20.0 < m.output_impedance() < 150.0
    assert math.isclose(m.rise_time_ns(), 0.8 * 3.3 / (1.98 / 1.2), rel_tol=1e-3)


def test_net_from_ibis_uses_measured_driver():
    from fpgaforge.physics import load_ibis, net_from_ibis

    m = load_ibis("examples/ice40_lvcmos33.ibs", "LVCMOS33_8mA")
    net = net_from_ibis(m, name="dq", trace_len_mm=120, load_pf=8)
    assert net.vdd == 3.3
    assert net.drive_impedance_ohm == m.output_impedance()
    assert net.driver_rise_ns == m.rise_time_ns()
    assert net.name == "dq"


# ------------------------------ crosstalk --------------------------- #
def test_crosstalk_next_saturates_and_scales():
    from fpgaforge.physics import CrosstalkPair, analyze_crosstalk

    # Long coupled run (2*Td >> tr) -> NEXT saturates at Kb*Vdd.
    p = CrosstalkPair(victim="v", vdd=3.3, aggressor_rise_ns=0.2,
                      coupling_len_mm=100, k_c=0.06, k_l=0.10, n_aggressors=1)
    r = analyze_crosstalk(p)
    kb = (0.06 + 0.10) / 4.0
    assert math.isclose(r.next_v, kb * 3.3, rel_tol=1e-6)
    # Two aggressors double the injected noise.
    r2 = analyze_crosstalk(CrosstalkPair(**{**p.__dict__, "n_aggressors": 2}))
    assert math.isclose(r2.next_v, 2 * r.next_v, rel_tol=1e-6)


def test_crosstalk_flags_when_over_margin():
    from fpgaforge.physics import CrosstalkPair, analyze_crosstalk

    p = CrosstalkPair(victim="clk", vdd=3.3, aggressor_rise_ns=0.15,
                      coupling_len_mm=150, k_c=0.20, k_l=0.30, n_aggressors=4,
                      noise_margin_v=0.5)
    r = analyze_crosstalk(p)
    assert not r.ok
    assert r.worst_v > r.noise_margin_v


# ---------------------------- SPICE helpers ------------------------- #
def test_measure_transient_settling_and_peak():
    from fpgaforge.physics.signal_integrity import _measure_transient, _EDGE_START_NS

    # Synthetic waveform: edge at 1 ns, overshoots to 4.0, rings, settles to 3.3.
    # 5%-of-Vdd band = 0.165 V; the 5 ns sample (3.6) is out of band, 8 ns is in.
    samples = [
        (0.0, 0.0), (1e-9, 0.0), (2e-9, 4.0), (3e-9, 2.9),
        (5e-9, 3.6), (8e-9, 3.30), (12e-9, 3.30),
    ]
    peak, settle_ns = _measure_transient(samples, vdd=3.3, tol_frac=0.05)
    assert peak == 4.0
    # Last out-of-band sample is at 5 ns -> settle = 5 - 1 = 4 ns.
    assert math.isclose(settle_ns, 4.0, rel_tol=1e-6)


def test_parse_wrdata():
    from fpgaforge.physics.signal_integrity import _parse_wrdata

    text = "0.000000e+00 0.0\n1.000000e-09 3.3\n2.0e-9 4.1\n# junk line\n"
    samples = _parse_wrdata(text)
    assert samples == [(0.0, 0.0), (1e-9, 3.3), (2e-9, 4.1)]


# ------------------------------- fusion ----------------------------- #
def test_physical_signoff_verdict_fusion():
    # Clean design + clean board -> PASS.
    ok = physical_signoff(100.0, 50.0, design_id="d",
                          nets=[Net(drive_impedance_ohm=50, trace_z0_ohm=50, trace_len_mm=10)])
    assert ok.verdict == "PASS"

    # An SI risk -> AT_RISK (does not block, but flagged).
    risky = physical_signoff(
        100.0, 50.0, design_id="d",
        nets=[Net(drive_impedance_ohm=15, trace_z0_ohm=50, driver_rise_ns=0.3,
                  trace_len_mm=200, load_pf=5)],
    )
    assert risky.verdict == "AT_RISK"

    # Missing worst-case timing -> BLOCKED.
    blocked = physical_signoff(60.0, 120.0, design_id="d")
    assert blocked.verdict == "BLOCKED"


# ------------------------------ power ------------------------------- #
def test_power_scales_with_activity_and_frequency():
    from fpgaforge.physics.power import estimate_power, PowerConfig

    res = {"luts": 2000, "ffs": 1500, "bram": 4, "dsp": 2}
    base = estimate_power(res, freq_mhz=50, io_count=20)
    hot = estimate_power(res, freq_mhz=100, io_count=20)
    busy = estimate_power(res, freq_mhz=50, io_count=20, activity=0.30)
    assert hot.dynamic_core_mw > base.dynamic_core_mw       # 2x freq -> ~2x dynamic
    assert busy.dynamic_core_mw > base.dynamic_core_mw       # more activity -> more power
    assert base.total_mw > 0


def test_low_power_ice40_stays_near_ambient():
    from fpgaforge.physics.power import estimate_power

    # A small iCE40 design dissipates little -> junction ~ ambient (self-heating
    # is real but tiny; the model should reflect that honestly).
    res = {"luts": 500, "ffs": 300}
    r = estimate_power(res, freq_mhz=48, io_count=8)
    assert r.junction_temp_c >= r.ambient_c
    assert r.junction_temp_c < r.ambient_c + 5
    assert r.within_thermal_limit


def test_thermal_runaway_is_flagged_over_limit():
    from fpgaforge.physics.power import estimate_power, PowerConfig

    # Force a hot scenario: huge switched cap, bad thermal path.
    cfg = PowerConfig(c_lut_pf=5.0, theta_ja_c_per_w=200.0, static_mw_nom=50.0)
    r = estimate_power({"luts": 5000, "ffs": 5000}, freq_mhz=200, io_count=100, cfg=cfg)
    assert r.junction_temp_c > r.tj_max_c
    assert not r.within_thermal_limit
    assert r.notes


def test_power_feeds_pvt_and_derates_when_hot():
    from fpgaforge.physics.power import PowerConfig

    # Cool design: guaranteed Fmax unchanged (self-heating negligible).
    cool = physical_signoff(100.0, 50.0, design_id="d",
                            resources={"luts": 500, "ffs": 300}, io_count=8)
    assert cool.power is not None
    assert math.isclose(cool.pvt.guaranteed_fmax_mhz, 100.0, rel_tol=0.02)

    # Hot design (bad thermal path) pushes Tj above the 85 C spec, which must
    # derate the guaranteed Fmax below the STA number.
    hotcfg = PowerConfig(c_lut_pf=4.0, theta_ja_c_per_w=150.0, static_mw_nom=40.0)
    hot = physical_signoff(100.0, 50.0, design_id="d",
                           resources={"luts": 5000, "ffs": 5000}, io_count=80,
                           power_cfg=hotcfg, activity=0.4)
    assert hot.power.junction_temp_c > 85.0
    assert hot.pvt.guaranteed_fmax_mhz < 100.0     # thermal feedback bit
    assert hot.verdict == "BLOCKED"                # over junction-temp spec


# --------------------------- field solver --------------------------- #
def test_microstrip_hits_50_ohm_for_typical_geometry():
    from fpgaforge.physics.fieldsolver import StackupGeometry, microstrip_line

    # ~50-ohm FR-4 microstrip: w=0.30mm over h=0.17mm, er=4.3.
    g = StackupGeometry(kind="microstrip", trace_w_mm=0.30, height_mm=0.17, er=4.3)
    sol = microstrip_line(g)
    assert 45.0 <= sol.z0_ohm <= 60.0
    # FR-4 microstrip is inhomogeneous: 1 < er_eff < er.
    assert 1.0 < sol.er_eff < 4.3
    # Velocity ~0.5-0.6 c (299.79 mm/ns).
    assert 150.0 < sol.velocity_mm_ns < 200.0


def test_stripline_velocity_matches_homogeneous_dielectric():
    from fpgaforge.physics.fieldsolver import StackupGeometry, stripline_line

    g = StackupGeometry(kind="stripline", trace_w_mm=0.15, plane_sep_mm=0.5,
                        thickness_mm=0.035, er=4.3)
    sol = stripline_line(g)
    # Homogeneous medium -> er_eff == er, v = c/sqrt(er).
    assert math.isclose(sol.er_eff, 4.3, rel_tol=1e-9)
    assert math.isclose(sol.velocity_mm_ns, 299.792458 / math.sqrt(4.3), rel_tol=1e-3)


def test_coupling_decreases_monotonically_with_spacing():
    from fpgaforge.physics.fieldsolver import StackupGeometry, coupled_microstrip

    prev = 1.0
    for spacing in (0.15, 0.3, 0.6, 1.2):
        g = StackupGeometry(kind="microstrip", trace_w_mm=0.30, height_mm=0.17,
                            er=4.3, spacing_mm=spacing)
        sol = coupled_microstrip(g)
        # Even mode > single-line > odd mode is the physical ordering.
        assert sol.z0_even > sol.z0_odd
        assert 0.0 < sol.k_backward < prev  # strictly decreasing with spacing
        prev = sol.k_backward
        assert sol.k_l > 0.0 and sol.k_c > 0.0
        # Microstrip is inductively dominated -> k_l > k_c (non-zero FEXT) for
        # meaningful coupling; at very wide spacing both collapse to model noise.
        if sol.k_backward > 0.03:
            assert sol.k_l > sol.k_c


def test_net_and_crosstalk_from_geometry_are_consistent():
    from fpgaforge.physics.fieldsolver import (
        StackupGeometry, net_from_geometry, crosstalk_from_geometry, coupled_microstrip,
    )
    from fpgaforge.physics.crosstalk import analyze_crosstalk

    g = StackupGeometry(kind="microstrip", trace_w_mm=0.30, height_mm=0.17,
                        er=4.3, spacing_mm=0.2)
    sol = coupled_microstrip(g)

    net = net_from_geometry(g, name="io", trace_len_mm=90.0, load_pf=8.0)
    assert math.isclose(net.trace_z0_ohm, sol.z0_ohm, rel_tol=1e-9)
    assert math.isclose(net.velocity_mm_ns, sol.velocity_mm_ns, rel_tol=1e-9)

    pair = crosstalk_from_geometry(g, coupling_len_mm=60.0, n_aggressors=2)
    assert math.isclose(pair.k_l, abs(sol.k_l), rel_tol=1e-9)
    assert math.isclose(pair.k_c, abs(sol.k_c), rel_tol=1e-9)
    res = analyze_crosstalk(pair)
    assert res.next_v > 0.0  # some near-end coupling


# ------------------------------- PDN -------------------------------- #
def test_pdn_target_impedance_math():
    from fpgaforge.physics.pdn import PDNConfig

    cfg = PDNConfig(vdd=1.2, ripple_fraction=0.05, transient_current_a=0.5)
    # Z_target = Vdd * ripple / I = 1.2 * 0.05 / 0.5 = 0.12 ohm.
    assert math.isclose(cfg.z_target_ohm, 0.12, rel_tol=1e-9)


def test_pdn_good_decoupling_beats_target_over_band():
    from fpgaforge.physics.pdn import PDNConfig, DecouplingCap, analyze_pdn

    # Generous decoupling + modest transient, capped at the mid band.
    cfg = PDNConfig(
        vdd=1.2, ripple_fraction=0.10, transient_current_a=0.2,
        vrm_resistance_mohm=5.0, vrm_inductance_nh=2.0, mount_inductance_nh=0.2,
        f_max_hz=2e7,
        caps=[
            DecouplingCap(47, 10, 1.0, 2, "bulk"),
            DecouplingCap(1.0, 8, 0.6, 4, "mid"),
            DecouplingCap(0.1, 20, 0.4, 8, "hf"),
        ],
    )
    res = analyze_pdn(cfg)
    assert res.meets_target
    assert not res.risks
    # Self-resonant frequencies are reported per bank.
    assert len(res.resonances) == 3


def test_pdn_flags_undersized_network():
    from fpgaforge.physics.pdn import PDNConfig, DecouplingCap, analyze_pdn

    # One bulk cap, aggressive transient -> impedance blows past target at HF.
    cfg = PDNConfig(vdd=1.2, ripple_fraction=0.05, transient_current_a=3.0,
                    mount_inductance_nh=1.0, caps=[DecouplingCap(10, 20, 1.2, 1, "bulk")])
    res = analyze_pdn(cfg)
    assert not res.meets_target
    assert res.risks
    assert res.worst_z_ohm > cfg.z_target_ohm


def test_pdn_feeds_signoff_at_risk():
    from fpgaforge.physics.pdn import PDNConfig, DecouplingCap

    bad_pdn = PDNConfig(vdd=1.2, ripple_fraction=0.05, transient_current_a=3.0,
                        mount_inductance_nh=1.0, caps=[DecouplingCap(10, 20, 1.2, 1)])
    report = physical_signoff(100.0, 50.0, design_id="d", pdn=bad_pdn)
    assert report.pdn is not None and not report.pdn.meets_target
    assert report.verdict == "AT_RISK"
