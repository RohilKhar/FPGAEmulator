"""Unit tests for the RL reward function (pure shaping + mock integration)."""

from fpgaforge.readiness import ReadinessReport, READY, AT_RISK, BLOCKED
from fpgaforge.reward import (
    DesignReward,
    score_report,
    score_design,
    _timing_score,
    _util_score,
    _equivalence_score,
)
from fpgaforge.backends.mock import MockBackend


def _report(verdict=READY, **metrics):
    base = {
        "synthesized": True, "routed_ok": True, "fmax_mhz": 100.0,
        "target_mhz": 50.0, "luts": 100, "lut_capacity": 5280,
        "io_count": 10, "io_capacity": 39, "bringup_status": "up",
        "equivalence_status": "proved_all", "equivalence_confidence": 1.0,
        "has_comb_loop": False, "has_latch": False, "clock_domains": 1,
    }
    base.update(metrics)
    return ReadinessReport(design_id="d", verdict=verdict, score=90, metrics=base,
                           fmax_mhz=base["fmax_mhz"], target_mhz=base["target_mhz"])


# --------------------------- shaping pieces ------------------------- #
def test_timing_score_is_dense_and_monotone():
    # Missing target still earns partial, climbing credit.
    assert _timing_score(0, 50) == 0.0
    assert 0.0 < _timing_score(25, 50) < _timing_score(45, 50) < 0.8
    # At target -> 0.8, comfortable margin -> 1.0, saturates above.
    assert abs(_timing_score(50, 50) - 0.8) < 1e-9
    assert abs(_timing_score(57.5, 50) - 1.0) < 1e-9
    assert _timing_score(200, 50) == 1.0


def test_util_score_penalizes_over_capacity():
    assert _util_score(100, 5280) == 1.0            # plenty of headroom
    assert _util_score(0, 5280) == 1.0              # unknown -> no penalty
    assert 0.0 <= _util_score(5000, 5280) < 1.0     # high util
    assert _util_score(6000, 5280) < _util_score(5300, 5280)  # over cap worse


def test_equivalence_score_ladder():
    assert _equivalence_score("proved_all", 1.0) == 1.0
    assert _equivalence_score("proved_bounded", 1.0) == 0.95
    assert _equivalence_score("differ", 0.0) == 0.0
    assert _equivalence_score("skipped", None) == 0.5
    # Verified scales with confidence.
    assert _equivalence_score("verified", 1.0) > _equivalence_score("verified", 0.0)


# --------------------------- full scoring --------------------------- #
def test_clean_design_scores_near_one_and_ready():
    r = score_report(_report())
    assert r.ready and r.verdict == READY
    assert r.reward > 0.95
    assert r.issues == []


def test_reward_climbs_as_timing_improves():
    slow = score_report(_report(verdict=AT_RISK, fmax_mhz=40.0)).reward
    tight = score_report(_report(verdict=AT_RISK, fmax_mhz=52.0)).reward
    comfy = score_report(_report(fmax_mhz=80.0)).reward
    assert slow < tight < comfy


def test_fatal_faults_cap_reward_and_emit_issues():
    # Did not synthesize -> fatal, reward capped at 0.5, issue present.
    r = score_report(_report(verdict=BLOCKED, synthesized=False, routed_ok=False,
                             fmax_mhz=0.0, luts=0))
    assert r.reward <= 0.5
    cats = {i.category for i in r.issues}
    assert "synthesis" in cats
    assert any(i.severity == "fatal" for i in r.issues)


def test_over_capacity_is_flagged_fatal():
    r = score_report(_report(verdict=BLOCKED, luts=6000))
    assert any(i.category == "resources" and i.severity == "fatal" for i in r.issues)
    assert r.reward <= 0.5


def test_timing_miss_produces_actionable_issue_with_metric():
    r = score_report(_report(verdict=BLOCKED, fmax_mhz=30.0))
    timing = [i for i in r.issues if i.category == "timing"][0]
    assert timing.metric == 30.0 and timing.target == 50.0
    assert timing.fix


def test_cdc_issue_is_tagged_modeled():
    r = score_report(_report(clock_domains=2))
    cdc = [i for i in r.issues if "clock domains" in i.message][0]
    assert cdc.evidence == "modeled"


def test_timing_issue_carries_critical_path_and_depth_aware_fix():
    crit = {"clock": "clk", "total_ns": 7.0, "logic_ns": 3.75, "routing_ns": 2.4,
            "logic_stages": 8, "from": "regA.Q", "to": "regB.D"}
    r = score_report(_report(verdict=BLOCKED, fmax_mhz=143.0, target_mhz=500.0,
                             critical_path=crit))
    t = [i for i in r.issues if i.category == "timing"][0]
    assert t.details and "8 cell stage" in t.details
    assert "regA.Q" in t.details and "regB.D" in t.details
    assert "pipeline register" in t.fix          # depth-aware advice


def test_routing_bound_timing_gets_routing_advice():
    crit = {"clock": "clk", "total_ns": 7.0, "logic_ns": 1.0, "routing_ns": 5.0,
            "logic_stages": 2, "from": "a", "to": "b"}
    r = score_report(_report(verdict=BLOCKED, fmax_mhz=143.0, target_mhz=500.0,
                             critical_path=crit))
    t = [i for i in r.issues if i.category == "timing"][0]
    assert "outing dominates" in t.fix or "fanout" in t.fix


def test_synthesis_issue_surfaces_located_tool_error():
    errs = ["error: examples/bad.v:12: syntax error near ';'"]
    r = score_report(_report(verdict=BLOCKED, synthesized=False, routed_ok=False,
                             fmax_mhz=0.0, luts=0, tool_errors=errs))
    syn = [i for i in r.issues if i.category == "synthesis"][0]
    assert syn.details and "syntax error" in syn.details
    assert syn.location == "examples/bad.v:12"


def test_equivalence_differ_includes_counterexample():
    r = score_report(_report(
        verdict=BLOCKED, equivalence_status="differ", equivalence_confidence=0.0,
        equivalence_detail="mismatch at cycle 7: out=1 expected 0"))
    eq = [i for i in r.issues if i.category == "equivalence"][0]
    assert eq.details and "cycle 7" in eq.details


def test_issues_are_sorted_most_blocking_first():
    r = score_report(_report(verdict=BLOCKED, synthesized=False, routed_ok=False,
                             fmax_mhz=0.0, luts=0, clock_domains=2))
    # A fatal (synthesis) must precede a warning (CDC).
    sev = [i.severity for i in r.issues]
    assert sev == sorted(sev, key=lambda s: {"fatal": 0, "error": 1, "warning": 2}[s])
    assert r.issues[0].severity == "fatal"


def test_unsynchronized_cdc_penalizes_drc_and_emits_issue():
    r = score_report(_report(cdc_worst="unsynchronized", cdc_unsynchronized=1,
                             cdc_detail="clk_a -> clk_b: unsynchronized"))
    assert r.components["drc"] < 0.5
    cdc = [i for i in r.issues if i.category == "cdc"][0]
    assert cdc.severity == "error" and "synchronizer" in cdc.fix


def test_structural_cdc_suppresses_name_heuristic():
    # When real CDC ran, we must not also emit the crude name-count issue.
    r = score_report(_report(cdc_worst="synchronized", clock_domains=2))
    heuristic = [i for i in r.issues if "name heuristic" in (i.message or "")]
    assert heuristic == []


def test_physics_pvt_margin_emits_modeled_issue():
    # Fmax just meets target nominally, but the slow/hot/low-V corner is slower,
    # so PVT should flag a modeled timing-margin risk.
    r = score_report(_report(routed_ok=True, fmax_mhz=52.0, target_mhz=50.0,
                             luts=500, synthesized=True))
    pvt = [i for i in r.issues if i.category == "pvt"]
    assert pvt and pvt[0].evidence == "modeled"
    assert pvt[0].severity == "warning"


def test_physics_can_be_disabled():
    r = score_report(_report(routed_ok=True, fmax_mhz=52.0, target_mhz=50.0,
                             luts=500, synthesized=True), physics=False)
    assert not any(i.category in ("pvt", "thermal") for i in r.issues)


def test_physics_penalty_never_exceeds_proven_reward():
    # A comfortably-fast design: physics penalty is at most a small factor.
    with_phys = score_report(_report(routed_ok=True, fmax_mhz=200.0, target_mhz=50.0,
                                     luts=500, synthesized=True), physics=True)
    without = score_report(_report(routed_ok=True, fmax_mhz=200.0, target_mhz=50.0,
                                   luts=500, synthesized=True), physics=False)
    assert with_phys.reward <= without.reward
    assert with_phys.reward >= without.reward * 0.9


def test_to_dict_is_json_serializable():
    import json

    r = score_report(_report(verdict=BLOCKED, fmax_mhz=30.0))
    payload = json.dumps(r.to_dict())
    back = json.loads(payload)
    assert set(back) == {"reward", "verdict", "ready", "components", "issues", "metrics"}
    assert isinstance(back["issues"], list) and back["issues"]


def test_weights_are_overridable():
    rep = _report(verdict=AT_RISK, fmax_mhz=40.0)
    base = score_report(rep).reward
    heavy = score_report(rep, weights={"timing": 0.8}).reward
    assert heavy != base  # re-weighting changes the scalar


# --------------------------- integration ---------------------------- #
def test_score_design_end_to_end_with_mock_backend():
    r = score_design("examples/counter.v", top="counter", clock_ns=20.0,
                     quick=True, backend=MockBackend())
    assert isinstance(r, DesignReward)
    assert 0.0 <= r.reward <= 1.0
    assert set(r.components) >= {"synthesis", "routing", "timing", "equivalence"}
