"""A reward function over RTL designs, for training an RL policy that emits RTL.

Instead of hooking a policy up to a real FPGA (slow, and you cannot afford a
failed build per rollout), this turns the readiness gate into a dense,
machine-readable reward:

* a shaped scalar :attr:`DesignReward.reward` in ``[0, 1]`` that increases
  *monotonically* as the design gets closer to being flashable -- so the policy
  gets gradient long before it fully closes (partial credit for synthesizing,
  routing, approaching the timing target, fitting resources, and proving
  bitstream equivalence);
* per-stage :attr:`DesignReward.components` so you can weight objectives or do
  reward decomposition;
* a structured :attr:`DesignReward.issues` list -- each with a category,
  severity, the offending metric vs. its target, a suggested fix, and an
  *evidence tier* (``proven`` vs ``modeled``) -- so the agent can learn *what*
  to fix, not just that it failed.

Design notes for RL use:

* ``optimize=False`` by default: the reward scores *the RTL the policy emitted*,
  not what the tool's knob-search could massage it into. (Normal synthesis
  optimization still runs -- that is compilation, not a design change.)
* Deterministic given the same design/tools, and JSON-serializable via
  :meth:`DesignReward.to_dict` for logging.
* ``quick=True`` skips the expensive bitstream-equivalence step for cheap early
  training; turn it off for the final, high-fidelity signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .readiness import ReadinessReport, assess

# Category -> weight in the shaped scalar reward. Weights sum to 1.0; timing is
# the heaviest continuous lever (aligns with the maximize-Fmax objective).
DEFAULT_WEIGHTS: dict[str, float] = {
    "synthesis": 0.12,
    "routing": 0.12,
    "timing": 0.24,
    "resources": 0.12,
    "io": 0.08,
    "drc": 0.08,
    "functional": 0.10,
    "equivalence": 0.14,
}

# Timing margin (fmax/target) at/above which timing scores a perfect 1.0.
_COMFORTABLE_MARGIN = 1.15
_HIGH_UTIL = 0.80


# Severity ordering so an agent sees the blocker first.
_SEVERITY_RANK = {"fatal": 0, "error": 1, "warning": 2}


@dataclass
class Issue:
    """One machine-readable problem with the design.

    Rich enough for an agent to make the *next edit*: not just the category and a
    generic fix, but the offending metric vs. target, the exact tool message /
    location where available, and concrete context (e.g. the timing critical
    path's endpoints, or an equivalence counterexample).
    """

    category: str          # synthesis|routing|timing|resources|io|drc|functional|equivalence
    severity: str          # "fatal" | "error" | "warning"
    message: str
    metric: float | None = None   # the achieved value (e.g. 42.0 MHz)
    target: float | None = None   # the value it needed to hit (e.g. 50.0 MHz)
    fix: str | None = None
    evidence: str = "proven"       # "proven" (tool fact) | "modeled" (physics estimate)
    details: str | None = None     # concrete context: crit path, counterexample, ...
    location: str | None = None    # file:line for a specific RTL fix, if known

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DesignReward:
    reward: float                              # shaped scalar in [0, 1]
    verdict: str                               # READY | AT_RISK | BLOCKED
    ready: bool                                # verdict == READY (terminal success)
    components: dict[str, float] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    report: ReadinessReport | None = None

    def to_dict(self) -> dict:
        """JSON-serializable view for logging / an RL replay buffer."""
        return {
            "reward": self.reward,
            "verdict": self.verdict,
            "ready": self.ready,
            "components": self.components,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DesignReward":
        """Rebuild a reward from :meth:`to_dict` (e.g. a cache hit); no report."""
        return cls(
            reward=d.get("reward", 0.0),
            verdict=d.get("verdict", ""),
            ready=d.get("ready", False),
            components=d.get("components", {}),
            issues=[Issue(**i) for i in d.get("issues", [])],
            metrics=d.get("metrics", {}),
            report=None,
        )

    def summary(self) -> str:
        lines = [
            f"reward : {self.reward:.3f}   verdict: {self.verdict}",
            "components:",
        ]
        for k, v in self.components.items():
            lines.append(f"  {k:12}: {v:.2f}")
        if self.issues:
            lines.append("issues (most blocking first):")
            for i in self.issues:
                mt = ""
                if i.metric is not None and i.target is not None:
                    mt = f" ({i.metric:g} vs {i.target:g})"
                loc = f" @ {i.location}" if i.location else ""
                lines.append(f"  [{i.severity}/{i.evidence}] {i.category}: {i.message}{mt}{loc}")
                if i.details:
                    for dl in i.details.splitlines():
                        lines.append(f"      {dl}")
                if i.fix:
                    lines.append(f"      fix: {i.fix}")
        return "\n".join(lines)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _timing_score(fmax: float, target: float) -> float:
    """Continuous: 0 when far below target, 0.8 at target, 1.0 at comfortable margin."""
    if target <= 0 or fmax <= 0:
        return 0.0
    ratio = fmax / target
    if ratio < 1.0:
        return 0.8 * ratio                       # dense credit for approaching target
    return _clamp(0.8 + 0.2 * (ratio - 1.0) / (_COMFORTABLE_MARGIN - 1.0))


def _util_score(count: int, capacity: int) -> float:
    if capacity <= 0 or count <= 0:
        return 1.0                               # nothing to penalize / unknown
    util = count / capacity
    if util <= _HIGH_UTIL:
        return 1.0
    if util <= 1.0:
        return _clamp(1.0 - 0.5 * (util - _HIGH_UTIL) / (1.0 - _HIGH_UTIL))
    return _clamp(0.5 - (util - 1.0))            # over capacity: drops fast


def _equivalence_score(status: str, confidence: float | None) -> float:
    return {
        "proved_all": 1.0,
        "proved_bounded": 0.95,
        "verified": 0.6 + 0.4 * (confidence if confidence is not None else 0.5),
        "inconclusive": 0.4,
        "differ": 0.0,
        "skipped": 0.5,                          # no evidence gathered -> neutral
    }.get(status, 0.5)


def _first_location(tool_errors: list[str]) -> str | None:
    """Pull a `file:line` out of a formatted diagnostic, if present."""
    import re

    for e in tool_errors:
        m = re.search(r"([\w./-]+\.s?v):(\d+)", e)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
    return None


def _grep_warnings(tool_warnings: list[str], needle: str) -> str | None:
    hits = [w for w in tool_warnings if needle.lower() in w.lower()]
    return "\n".join(hits[:5]) or None


def _resource_context(m: dict) -> str | None:
    luts, cap = m.get("luts", 0), m.get("lut_capacity", 0)
    io, iocap = m.get("io_count", 0), m.get("io_capacity", 0)
    parts = []
    if cap:
        parts.append(f"LUTs {luts}/{cap}")
    if iocap:
        parts.append(f"I/O {io}/{iocap}")
    return ("utilization: " + ", ".join(parts)) if parts else None


def _crit_path_text(crit: dict) -> str | None:
    """Human/agent-readable critical path so the fix targets the right spot."""
    if not crit:
        return None
    txt = (
        f"critical path on clock '{crit.get('clock','?')}': "
        f"{crit.get('total_ns',0):.2f} ns "
        f"= {crit.get('logic_ns',0):.2f} ns logic across "
        f"{crit.get('logic_stages',0)} cell stage(s) "
        f"+ {crit.get('routing_ns',0):.2f} ns routing"
    )
    if crit.get("from"):
        txt += f"\n  from: {crit['from']}"
    if crit.get("to"):
        txt += f"\n  to  : {crit['to']}"
    return txt


def _timing_fix(crit: dict) -> str:
    """Point the agent at the dominant cost (logic depth vs. routing)."""
    stages = crit.get("logic_stages", 0)
    logic = crit.get("logic_ns", 0.0)
    routing = crit.get("routing_ns", 0.0)
    if stages and stages >= 4:
        return (f"Logic depth is {stages} cell stages ({logic:.2f} ns); insert a "
                f"pipeline register to split this path (retiming helps too).")
    if routing > logic and (logic or routing):
        return ("Routing dominates the path; reduce fanout/congestion (replicate "
                "high-fanout drivers) or lower utilization.")
    return "Pipeline the critical path, enable retiming, or relax the target clock."


def _physics_issues(m: dict) -> tuple[list[Issue], float]:
    """Modeled-tier physics feedback (PVT worst-case timing + power/thermal).

    Closed-form only (no tools), so it is safe to run in the pure scorer. These
    are ``modeled`` estimates -- they never cap a proven reward, but they nudge
    the policy toward designs with real-world margin and flag physical risks the
    logic/timing signal cannot see. Returns ``(issues, penalty_factor in (0,1])``.
    """
    issues: list[Issue] = []
    penalty = 1.0
    fmax = float(m.get("fmax_mhz", 0.0) or 0.0)
    target = float(m.get("target_mhz", 0.0) or 0.0)
    if not m.get("routed_ok") or fmax <= 0 or target <= 0:
        return issues, penalty

    try:
        from .physics.pvt import derate_fmax
        from .physics.power import estimate_power
    except Exception:  # noqa: BLE001 - physics optional
        return issues, penalty

    # PVT: the STA Fmax is the slow corner, so the *headroom* over target is what
    # absorbs board-temperature / voltage / aging variation the tool can't see.
    pvt = derate_fmax(fmax, target)
    guaranteed = pvt.guaranteed_fmax_mhz
    if 0 < guaranteed < target * _COMFORTABLE_MARGIN:
        penalty *= 0.95
        worst = pvt.worst_corner
        issues.append(Issue(
            "pvt", "warning", "thin worst-case P/V/T timing margin",
            metric=round(guaranteed, 2), target=round(target, 2),
            fix="Add ~15% timing headroom so slow-process / hot / low-voltage "
                "operation (and aging) still meets the clock.",
            evidence="modeled",
            details=(f"worst corner {worst.name}: {guaranteed:.1f} MHz "
                     f"({pvt.margin_pct():+.0f}% vs target)") if worst else None,
        ))

    # Power/thermal: estimate junction temp from resources at the target clock.
    resources = {"luts": int(m.get("luts", 0)), "ffs": int(m.get("ffs", 0)),
                 "bram": int(m.get("bram", 0)), "dsp": int(m.get("dsp", 0))}
    if any(resources.values()):
        pw = estimate_power(resources, freq_mhz=target,
                            io_count=int(m.get("io_count", 0)))
        if not pw.within_thermal_limit:
            penalty *= 0.9
            issues.append(Issue(
                "thermal", "error", "estimated junction temperature over spec",
                metric=round(pw.junction_temp_c, 1), target=round(pw.tj_max_c, 1),
                fix="Lower activity/frequency, cut power, or improve cooling "
                    "(smaller theta_JA / bigger package).",
                evidence="modeled",
                details=f"total {pw.total_mw:.1f} mW -> Tj {pw.junction_temp_c:.1f} C "
                        f"(ambient {pw.ambient_c:.0f} C)",
            ))
    return issues, penalty


def score_report(
    report: ReadinessReport, weights: dict[str, float] | None = None,
    physics: bool = True,
) -> DesignReward:
    """Turn a :class:`ReadinessReport` into a shaped reward + structured issues.

    Pure: no tools run here, so it is fully unit-testable and lets you re-score a
    stored report with different weights. With ``physics=True`` (default) the
    reward also folds in ``modeled``-tier PVT and power/thermal margins so an
    agent iterating on a design also sees physical risk, not just logic/timing.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    m = report.metrics or {}
    issues: list[Issue] = []

    synthesized = bool(m.get("synthesized"))
    routed = bool(m.get("routed_ok"))
    fmax = float(m.get("fmax_mhz", report.fmax_mhz) or 0.0)
    target = float(m.get("target_mhz", report.target_mhz) or 0.0)
    luts = int(m.get("luts", 0))
    lut_cap = int(m.get("lut_capacity", 0))
    io_count = int(m.get("io_count", 0))
    io_cap = int(m.get("io_capacity", 0))
    bringup = m.get("bringup_status", "skipped")
    eq_status = m.get("equivalence_status", "skipped")
    eq_conf = m.get("equivalence_confidence")
    eq_detail = m.get("equivalence_detail", "")
    crit = m.get("critical_path") or {}
    tool_errors = m.get("tool_errors") or []
    tool_warnings = m.get("tool_warnings") or []

    comp: dict[str, float] = {}
    comp["synthesis"] = 1.0 if synthesized else 0.0
    comp["routing"] = 1.0 if routed else 0.0
    comp["timing"] = _timing_score(fmax, target) if routed else 0.0
    comp["resources"] = _util_score(luts, lut_cap)
    comp["io"] = _util_score(io_count, io_cap)

    drc = 1.0
    if m.get("has_comb_loop"):
        drc = 0.0
    elif m.get("has_latch"):
        drc *= 0.7
    cdc_worst = m.get("cdc_worst", "unknown")
    if cdc_worst == "unsynchronized":
        drc *= 0.3          # a real, dangerous CDC bug
    elif cdc_worst == "single_flop":
        drc *= 0.7
    comp["drc"] = drc

    comp["functional"] = {"up": 1.0, "skipped": 0.5, "down": 0.0}.get(bringup, 0.5)
    comp["equivalence"] = _equivalence_score(eq_status, eq_conf)

    # ---- structured issues (what to fix) ----
    # Real, located tool messages are the most actionable feedback for an agent.
    tool_err_text = "\n".join(tool_errors[:8]) or None
    first_loc = _first_location(tool_errors)

    if not synthesized:
        issues.append(Issue("synthesis", "fatal", "RTL did not synthesize",
                            fix="Fix the syntax/elaboration errors reported by the synthesizer.",
                            details=tool_err_text, location=first_loc))
    if synthesized and not routed:
        issues.append(Issue("routing", "fatal", "did not place & route",
                            fix="Reduce resource/I/O usage or target a larger device.",
                            details=tool_err_text or _resource_context(m)))
    if routed and target > 0 and fmax < target:
        issues.append(Issue("timing", "error", "misses the timing target",
                            metric=round(fmax, 2), target=round(target, 2),
                            fix=_timing_fix(crit),
                            details=_crit_path_text(crit)))
    elif routed and target > 0 and fmax < target * _COMFORTABLE_MARGIN:
        issues.append(Issue("timing", "warning", "tight timing margin",
                            metric=round(fmax, 2), target=round(target, 2),
                            fix="Add timing headroom for PVT/board variation.",
                            details=_crit_path_text(crit)))
    if lut_cap and luts / lut_cap > 1.0:
        issues.append(Issue("resources", "fatal", "over LUT capacity",
                            metric=luts, target=lut_cap,
                            fix="Reduce logic or use a larger device."))
    elif lut_cap and luts / lut_cap > _HIGH_UTIL:
        issues.append(Issue("resources", "warning", "high LUT utilization",
                            metric=luts, target=lut_cap,
                            fix="High utilization raises congestion/timing risk."))
    if io_cap and io_count > io_cap:
        issues.append(Issue("io", "fatal", "too many top-level I/O for the package",
                            metric=io_count, target=io_cap,
                            fix="Serialize I/O or choose a larger package."))
    if m.get("has_comb_loop"):
        issues.append(Issue("drc", "fatal", "combinational loop detected",
                            fix="Break the combinational feedback in the RTL."))
    if m.get("has_latch"):
        issues.append(Issue("drc", "warning", "inferred latch(es)",
                            fix="Complete if/else or add default assignments.",
                            details=_grep_warnings(tool_warnings, "latch")))
    if m.get("cdc_unsynchronized"):
        issues.append(Issue("cdc", "error",
                            f"{m['cdc_unsynchronized']} unsynchronized clock-domain crossing(s)",
                            fix="Insert a two-flop synchronizer (async FIFO for buses); "
                                "never pass a raw cross-domain signal through combinational logic.",
                            details=m.get("cdc_detail") or None))
    elif m.get("cdc_single_flop"):
        issues.append(Issue("cdc", "warning",
                            f"{m['cdc_single_flop']} single-flop clock-domain crossing(s)",
                            fix="Use a two-flop synchronizer to cut metastability risk.",
                            details=m.get("cdc_detail") or None))
    if bringup == "down":
        issues.append(Issue("functional", "error", "virtual bring-up misbehaved",
                            fix="Fix the functional bug found in the virtual fabric."))
    if eq_status == "differ":
        issues.append(Issue("equivalence", "fatal",
                            "flashed bitstream differs from RTL",
                            fix="Toolchain miscompile; do not flash. Reproduce with the counterexample.",
                            details=eq_detail or None))
    # Only fall back to the name-count heuristic when structural CDC didn't run.
    if (m.get("cdc_worst", "unknown") == "unknown"
            and (m.get("clock_domains") or 0) > 1):
        issues.append(Issue("drc", "warning",
                            f"{m['clock_domains']} clock domains (CDC risk, name heuristic)",
                            fix="Ensure proper synchronizers; sim cannot fully prove CDC.",
                            evidence="modeled"))

    # Modeled-tier physics feedback (PVT worst-case + power/thermal).
    physics_penalty = 1.0
    if physics:
        phys_issues, physics_penalty = _physics_issues(m)
        issues.extend(phys_issues)

    reward = sum(w.get(k, 0.0) * v for k, v in comp.items())
    # Physics is modeled (not proven), so it only mildly scales the reward.
    reward *= physics_penalty
    # If a fatal fault exists, cap the reward so the policy cannot farm partial
    # credit around a design that fundamentally cannot ship.
    if any(i.severity == "fatal" for i in issues):
        reward = min(reward, 0.5)

    # Most-blocking issue first, so an agent knows what to fix next.
    issues.sort(key=lambda i: _SEVERITY_RANK.get(i.severity, 9))

    return DesignReward(
        reward=round(_clamp(reward), 4),
        verdict=report.verdict,
        ready=report.verdict == "READY",
        components={k: round(v, 4) for k, v in comp.items()},
        issues=issues,
        metrics=m,
        report=report,
    )


def score_design(
    rtl: str | Sequence[str],
    top: str,
    target_fpga: str = "ice40_up5k",
    clock_ns: float = 10.0,
    cycles: int = 64,
    quick: bool = False,
    optimize: bool = False,
    weights: dict[str, float] | None = None,
    physics: bool = True,
    cache=None,
    backend=None,
) -> DesignReward:
    """Score an RTL design as an RL reward (runs the real flow, then shapes it).

    Set ``quick=True`` to skip bitstream-equivalence for cheap rollouts. Keep
    ``optimize=False`` so the reward reflects the emitted RTL, not a tool-tuned
    variant. Pass ``cache=True`` (or a :class:`~fpgaforge.cache.RewardCache`) to
    return instantly on a repeat of the same design+flags -- the common case in
    an RL loop.
    """
    from .cache import RewardCache, reward_key

    rc = None
    if cache is True:
        rc = RewardCache()
    elif isinstance(cache, RewardCache):
        rc = cache

    rtl_files = [rtl] if isinstance(rtl, str) else list(rtl)
    key = None
    if rc is not None:
        key = reward_key(rtl_files, top, target_fpga, clock_ns, cycles,
                         quick, optimize, physics, weights)
        cached = rc.get(key)
        if cached is not None:
            return DesignReward.from_dict(cached)

    report = assess(
        rtl=rtl, top=top, target_fpga=target_fpga, clock_ns=clock_ns,
        cycles=cycles, optimize=optimize, prove_equivalence=not quick,
        backend=backend,
    )
    result = score_report(report, weights=weights, physics=physics)
    if rc is not None and key is not None:
        rc.put(key, result.to_dict())
    return result


def score_batch(
    designs: Sequence[dict], max_workers: int = 4, cache=None,
) -> list[DesignReward]:
    """Score many designs in parallel (for RL batch rollouts / sweeps).

    ``designs`` is a list of kwargs dicts for :func:`score_design`. The real flow
    is subprocess-bound, so threads give real speedup; per-design workdirs are
    keyed by design id so parallel runs of *distinct* designs never collide. A
    shared ``cache`` (True or a :class:`RewardCache`) is honored across workers.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .cache import RewardCache

    rc = RewardCache() if cache is True else cache  # share one instance
    results: list[DesignReward] = [None] * len(designs)  # type: ignore[list-item]

    def _run(idx_spec):
        idx, spec = idx_spec
        return idx, score_design(cache=rc, **spec)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, res in ex.map(_run, list(enumerate(designs))):
            results[idx] = res
    return results
