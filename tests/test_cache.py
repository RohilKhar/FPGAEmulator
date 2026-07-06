"""Tests for reward caching and parallel batch scoring."""

from fpgaforge.cache import RewardCache, reward_key
from fpgaforge.reward import score_design, score_batch, DesignReward
from fpgaforge.backends.mock import MockBackend


def test_reward_key_is_stable_and_sensitive():
    a = reward_key(["examples/counter.v"], "counter", "ice40_up5k", 10.0, 64,
                   False, False, True)
    b = reward_key(["examples/counter.v"], "counter", "ice40_up5k", 10.0, 64,
                   False, False, True)
    c = reward_key(["examples/counter.v"], "counter", "ice40_up5k", 5.0, 64,
                   False, False, True)      # different clock
    assert a == b
    assert a != c


def test_cache_hit_avoids_recompute(tmp_path):
    rc = RewardCache(root=tmp_path / "rc")
    kwargs = dict(rtl="examples/counter.v", top="counter", clock_ns=10.0,
                  cycles=8, quick=True, backend=MockBackend())
    first = score_design(cache=rc, **kwargs)
    assert rc.misses == 1 and rc.hits == 0
    second = score_design(cache=rc, **kwargs)
    assert rc.hits == 1
    assert second.reward == first.reward
    assert second.verdict == first.verdict
    assert isinstance(second, DesignReward)


def test_cache_reconstructs_issues(tmp_path):
    rc = RewardCache(root=tmp_path / "rc")
    kwargs = dict(rtl="examples/counter.v", top="counter", clock_ns=10.0,
                  cycles=8, quick=True, backend=MockBackend())
    first = score_design(cache=rc, **kwargs)
    second = score_design(cache=rc, **kwargs)
    assert [i.to_dict() for i in second.issues] == [i.to_dict() for i in first.issues]


def test_from_dict_roundtrip():
    rc = RewardCache
    d = {
        "reward": 0.7, "verdict": "AT_RISK", "ready": False,
        "components": {"timing": 0.5},
        "issues": [{"category": "timing", "severity": "warning", "message": "x"}],
        "metrics": {"fmax_mhz": 42.0},
    }
    r = DesignReward.from_dict(d)
    assert r.reward == 0.7
    assert r.issues[0].category == "timing"
    assert r.report is None


def test_score_batch_parallel(tmp_path):
    specs = [
        dict(rtl="examples/counter.v", top="counter", clock_ns=10.0, cycles=8,
             quick=True, backend=MockBackend()),
        dict(rtl="examples/counter.v", top="counter", clock_ns=20.0, cycles=8,
             quick=True, backend=MockBackend()),
    ]
    results = score_batch(specs, max_workers=2, cache=RewardCache(root=tmp_path / "rc"))
    assert len(results) == 2
    assert all(isinstance(r, DesignReward) for r in results)
