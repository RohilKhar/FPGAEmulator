"""The AI-native optimization loop.

`optimize()` is the headline API. It replaces the manual edit/compile/read-report
cycle with: propose knob candidates, rank them with the predictive model, run the
most promising ones through a real (or mock) backend, log everything to the
corpus, and return the best implementation found.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from . import features as feat
from .backends.base import Backend, Design, FlowOptions, RunResult
from .backends.ice40 import Ice40Backend
from .backends.ecp5 import Ecp5Backend
from .backends.mock import MockBackend
from .corpus import Corpus
from .model import FmaxPredictor

OBJECTIVES = ("maximize_fmax", "minimize_luts")


@dataclass
class OptimizationResult:
    design_id: str
    objective: str
    backend: str
    best: RunResult | None = None
    baseline: RunResult | None = None
    history: list[RunResult] = field(default_factory=list)

    @property
    def improvement_pct(self) -> float:
        if not (self.best and self.baseline):
            return 0.0
        b = self.baseline.metrics.fmax_mhz
        if b <= 0:
            return 0.0
        return 100.0 * (self.best.metrics.fmax_mhz - b) / b

    def summary(self) -> str:
        lines = [
            f"design    : {self.design_id}",
            f"objective : {self.objective}",
            f"backend   : {self.backend}",
            f"runs      : {len(self.history)}",
        ]
        if self.baseline:
            m = self.baseline.metrics
            lines.append(
                f"baseline  : {m.fmax_mhz:.2f} MHz, {m.luts} LUTs "
                f"(knobs {self.baseline.options.to_dict()})"
            )
        if self.best:
            m = self.best.metrics
            lines.append(
                f"best      : {m.fmax_mhz:.2f} MHz, {m.luts} LUTs "
                f"(knobs {self.best.options.to_dict()})"
            )
            lines.append(f"improvement: {self.improvement_pct:+.1f}% Fmax")
            if self.best.bitstream_path:
                lines.append(f"bitstream : {self.best.bitstream_path}")
        # If nothing succeeded, surface the actual tool errors.
        if not any(r.success for r in self.history):
            probe = self.best or (self.history[0] if self.history else None)
            if probe is not None:
                if probe.error:
                    lines.append(f"error     : {probe.error}")
                for d in probe.errors():
                    lines.append(f"  {d.format()}")
        return "\n".join(lines)


def _backend_instance(name: str) -> Backend | None:
    """Construct a backend by its registry name, or None if we don't ship it."""
    if name == "ice40":
        return Ice40Backend()
    if name == "ecp5":
        return Ecp5Backend()
    if name == "vivado":
        from .backends.vivado import VivadoBackend

        return VivadoBackend()
    if name == "quartus":
        from .backends.quartus import QuartusBackend

        return QuartusBackend()
    if name == "gowin":
        from .backends.gowin import GowinBackend

        return GowinBackend()
    return None


def backend_for_target(target: str | None = None) -> Backend:
    """Pick the backend matching a device target (falling back to mock).

    The device registry (fpgaforge/devices.py) names which backend implements a
    target -- iCE40/ECP5 (open source), Vivado (AMD), or Quartus (Intel). If the
    matching tools are not installed, the offline MockBackend is returned so
    callers still get a (mock) result.
    """
    from .devices import get as _dev_get

    dev = _dev_get(target)
    name = dev.backend if dev is not None else "ice40"
    b = _backend_instance(name)
    if b is not None and b.is_available():
        return b
    return MockBackend()


def default_backend(target: str | None = None) -> Backend:
    """Prefer the real flow for ``target`` if its tools are installed, else mock."""
    return backend_for_target(target)


def candidate_space(seeds: Sequence[int] = (1, 2, 3)) -> list[FlowOptions]:
    """Small, explicit knob search space for the first milestone."""
    grid = list(
        itertools.product(
            (False, True),   # abc9
            (False, True),   # retime
            (False, True),   # pipeline_output
            (True,),         # dsp (keep multipliers hardened)
            ("heap",),       # placer
            seeds,
        )
    )
    seen: set[str] = set()
    candidates: list[FlowOptions] = []
    for abc9, retime, pipe, dsp, placer, seed in grid:
        opt = FlowOptions(
            abc9=abc9,
            retime=retime,
            pipeline_output=pipe,
            dsp=dsp,
            placer=placer,
            seed=seed,
        )
        if opt.key() not in seen:
            seen.add(opt.key())
            candidates.append(opt)
    return candidates


def _score(result: RunResult, objective: str) -> float:
    m = result.metrics
    if not (result.success and m.routed_ok):
        return float("-inf")
    if objective == "minimize_luts":
        return -float(m.luts)
    return float(m.fmax_mhz)  # maximize_fmax (default)


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def optimize(
    rtl: str | Sequence[str],
    top: str,
    target_fpga: str = "ice40_up5k",
    objective: str = "maximize_fmax",
    iterations: int = 8,
    clock_ns: float = 10.0,
    backend: Backend | None = None,
    model: FmaxPredictor | None = None,
    corpus: Corpus | None = None,
    run_dir: str | Path = ".runs",
    seeds: Sequence[int] = (1, 2, 3),
    pcf: str | Path | None = None,
) -> OptimizationResult:
    """Optimize a design's implementation toward `objective`.

    Args:
        rtl: one RTL file path, or a sequence of them (also accepts inline RTL).
        top: top module name.
        target_fpga: device target, e.g. "ice40_up5k".
        objective: one of OBJECTIVES.
        iterations: max number of backend runs (including the baseline).
        clock_ns: target clock period constraint.
        backend: override the backend (defaults to auto-detected).
        model: override the Fmax predictor (defaults to loaded/heuristic).
        corpus: override the corpus store.
        run_dir: root directory for per-run artifacts.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")

    rtl_files = [rtl] if isinstance(rtl, str) else list(rtl)
    design = Design(
        rtl_files=tuple(rtl_files), top=top, target=target_fpga, clock_ns=clock_ns,
        pcf=str(pcf) if pcf else None,
    )
    if backend is None:
        backend = default_backend(design.target)
    if model is None:
        model = FmaxPredictor.load()
    if corpus is None:
        corpus = Corpus()
    run_root = Path(run_dir) / _sanitize(design.design_id())

    # Rank candidates using cheap RTL features (same design, knobs differ).
    rank_features = feat.from_rtl_files(design.rtl_files)
    baseline_opts = FlowOptions()  # defaults
    others = [o for o in candidate_space(seeds) if o.key() != baseline_opts.key()]
    others.sort(key=lambda o: model.predict(rank_features, o), reverse=True)

    ordered = [baseline_opts] + others
    to_run = ordered[: max(1, iterations)]

    result = OptimizationResult(
        design_id=design.design_id(), objective=objective, backend=backend.name
    )

    for opt in to_run:
        workdir = run_root / opt.key()
        run = backend.run(design, opt, workdir)
        corpus.append(run, extra={"objective": objective, "target": target_fpga})
        result.history.append(run)
        if result.baseline is None and opt.key() == baseline_opts.key():
            result.baseline = run

    # Best = argmax of the objective over successful runs (fall back to first).
    successful = [r for r in result.history if _score(r, objective) != float("-inf")]
    if successful:
        result.best = max(successful, key=lambda r: _score(r, objective))
    elif result.history:
        result.best = result.history[0]

    return result
