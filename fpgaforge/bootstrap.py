"""Corpus bootstrapping: turn a set of designs into training data.

The Fmax predictor is only as good as its corpus, and a fresh install has an
empty one. `bootstrap_corpus` sweeps each design across the knob search space
(and, optionally, several target devices / clock constraints), runs every point
through a backend, and appends the outcome to the corpus. The result is a
ready-to-train dataset produced in one command instead of waiting for organic
optimizer runs to accumulate.

It is deliberately backend-agnostic: pass a MockBackend for an offline dry run,
or let it auto-select the real flow per target when the tools are installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .backends.base import Backend, Design, FlowOptions
from .corpus import Corpus
from .optimizer import _sanitize, candidate_space, default_backend


@dataclass
class BootstrapSpec:
    """One design to sweep, optionally across multiple targets/clocks."""

    rtl_files: tuple[str, ...]
    top: str
    targets: tuple[str, ...] = ("ice40_up5k",)
    clocks_ns: tuple[float, ...] = (10.0,)
    name: str | None = None

    def designs(self) -> list[Design]:
        out: list[Design] = []
        for target in self.targets:
            for clock_ns in self.clocks_ns:
                out.append(
                    Design(
                        rtl_files=tuple(self.rtl_files),
                        top=self.top,
                        target=target,
                        clock_ns=clock_ns,
                        name=self.name,
                    )
                )
        return out


@dataclass
class BootstrapReport:
    designs: int = 0
    points: int = 0
    runs: int = 0
    successes: int = 0
    routed: int = 0
    corpus_size: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        rate = (self.routed / self.runs * 100.0) if self.runs else 0.0
        lines = [
            f"bootstrap: {self.designs} design/target/clock combos, {self.runs} runs "
            f"in {self.elapsed_s:.1f}s",
            f"  routed OK : {self.routed}/{self.runs} ({rate:.0f}%)   "
            f"successes: {self.successes}",
            f"  corpus now: {self.corpus_size} rows",
        ]
        if self.errors:
            lines.append(f"  {len(self.errors)} run(s) errored; first: {self.errors[0]}")
        return "\n".join(lines)


def bootstrap_corpus(
    specs: Sequence[BootstrapSpec],
    seeds: Sequence[int] = (1, 2),
    backend: Backend | None = None,
    corpus: Corpus | None = None,
    run_dir: str | Path = ".runs/bootstrap",
    max_per_design: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> BootstrapReport:
    """Sweep designs x knobs and append every run to the corpus.

    Args:
        specs: designs to sweep (each may fan out to several targets/clocks).
        seeds: placement seeds to include in the knob space.
        backend: force a backend; default auto-selects per design target.
        corpus: corpus store to append to (defaults to the shared corpus).
        run_dir: root for per-run artifacts.
        max_per_design: cap the number of knob points per design (None = all).
        progress: optional callback invoked with a status line per run.
    """
    # NB: Corpus defines __len__, so an empty corpus is falsy -- test explicitly.
    corpus = corpus if corpus is not None else Corpus()
    run_root = Path(run_dir)
    report = BootstrapReport()
    start = time.time()

    for spec in specs:
        for design in spec.designs():
            report.designs += 1
            be = backend or default_backend(design.target)
            candidates = candidate_space(seeds)
            if max_per_design is not None:
                candidates = candidates[:max_per_design]
            report.points += len(candidates)

            base = run_root / _sanitize(design.design_id())
            for opt in candidates:
                workdir = base / opt.key()
                try:
                    run = be.run(design, opt, workdir)
                except Exception as exc:  # backends should not raise, but be safe
                    report.errors.append(f"{design.name}/{opt.key()}: {exc}")
                    continue
                corpus.append(
                    run, extra={"target": design.target, "source": "bootstrap"}
                )
                report.runs += 1
                if run.success:
                    report.successes += 1
                if run.metrics.routed_ok:
                    report.routed += 1
                if run.error:
                    report.errors.append(f"{design.name}/{opt.key()}: {run.error}")
                if progress is not None:
                    progress(
                        f"{design.name} [{design.target}] {opt.key()}: "
                        f"fmax={run.metrics.fmax_mhz:.1f} routed={run.metrics.routed_ok}"
                    )

    report.elapsed_s = time.time() - start
    report.corpus_size = len(corpus)
    return report
