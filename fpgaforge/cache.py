"""Content-addressed caching for the RL reward.

Running the real flow per rollout is the throughput bottleneck when the reward
is an RL environment. But the reward is a pure function of (RTL content, top,
target, clock, and the scoring flags): identical inputs must give an identical
DesignReward. So we cache by a content hash and return instantly on a repeat --
which is common in RL (revisiting states, replaying, hyperparameter sweeps).

The cache stores the JSON-serializable reward (scalar, components, issues,
metrics); the heavyweight ReadinessReport object is not restored (RL loops only
need the reward + issues). Disk-backed so it survives across processes/runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def reward_key(
    rtl_files: Iterable[str],
    top: str,
    target_fpga: str,
    clock_ns: float,
    cycles: int,
    quick: bool,
    optimize: bool,
    physics: bool,
    weights: dict | None = None,
) -> str:
    """Stable content hash over everything that affects the reward."""
    h = hashlib.sha256()
    for p in rtl_files:
        path = Path(p)
        h.update(b"\x00")
        h.update(path.read_bytes() if path.exists() else str(p).encode())
    payload = json.dumps(
        {
            "top": top, "target": target_fpga, "clock_ns": clock_ns,
            "cycles": cycles, "quick": quick, "optimize": optimize,
            "physics": physics, "weights": weights or {},
        },
        sort_keys=True,
    )
    h.update(payload.encode())
    return h.hexdigest()[:32]


class RewardCache:
    """A tiny disk-backed cache of ``DesignReward.to_dict()`` payloads."""

    def __init__(self, root: str | Path = ".cache/reward") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                self.hits += 1
                return data
            except (OSError, json.JSONDecodeError):
                return None
        self.misses += 1
        return None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        try:
            self._path(key).write_text(json.dumps(payload))
        except OSError:
            pass

    def clear(self) -> None:
        for f in self.root.glob("*.json"):
            f.unlink(missing_ok=True)
        self.hits = self.misses = 0
