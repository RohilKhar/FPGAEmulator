"""MockBackend: a deterministic, tool-free backend.

It produces plausible, reproducible metrics from heuristic RTL features and the
chosen knobs, so the API, predictive model, optimizer, CLI, and tests all run
with zero tools installed. Knob effects are modeled to reward the same choices
that help in reality (retiming, abc9, DSP mapping, pipelining), which lets the
optimizer demonstrably improve Fmax offline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .. import features as feat
from .base import Backend, Design, FlowOptions, RunMetrics, RunResult

# Rough iCE40 UP5K capacity for a mock congestion check.
_LUT_CAPACITY = 5280


class MockBackend(Backend):
    name = "mock"

    def is_available(self) -> bool:
        return True

    def run(self, design: Design, options: FlowOptions, workdir: Path) -> RunResult:
        workdir.mkdir(parents=True, exist_ok=True)
        f = feat.from_rtl_files(design.rtl_files)

        carries = f["num_carries"]
        max_fanout = f["max_fanout"]
        luts = f["num_luts"]
        ffs = f["num_ffs"]
        dsp_avail = f["num_dsp"]

        # Baseline Fmax degrades with arithmetic depth and high fanout.
        base = 260.0
        base /= 1.0 + carries / 60.0
        base /= 1.0 + max_fanout / 45.0

        mult = 1.0
        if options.abc9:
            mult *= 1.08
        if options.retime:
            mult *= 1.12
        if options.pipeline_output:
            mult *= 1.20  # shorter critical path
        if options.dsp and dsp_avail > 0:
            mult *= 1.15  # multiplies hardened into DSP blocks
        if options.placer == "heap":
            mult *= 1.03

        fmax = base * mult * self._seed_jitter(design, options)

        est_luts = int(luts)
        est_ffs = int(ffs)
        est_dsp = int(dsp_avail) if options.dsp else 0
        if options.dsp and dsp_avail > 0:
            # DSP absorbs some LUT/carry logic.
            est_luts = int(est_luts * 0.7)
        if options.pipeline_output:
            est_ffs += int(f["num_outputs"]) or 8

        routed_ok = est_luts <= _LUT_CAPACITY

        metrics = RunMetrics(
            fmax_mhz=round(fmax, 3) if routed_ok else 0.0,
            target_freq_mhz=design.target_freq_mhz,
            crit_path_ns=round(1000.0 / fmax, 4) if (routed_ok and fmax > 0) else 0.0,
            luts=est_luts,
            ffs=est_ffs,
            bram=int(f["num_mem_bits"] // 4096),
            dsp=est_dsp,
            carries=int(carries),
            routed_ok=routed_ok,
        )

        return RunResult(
            design_id=design.design_id(),
            options=options,
            metrics=metrics,
            features=f,
            success=routed_ok,
            backend=self.name,
            workdir=str(workdir),
            error=None if routed_ok else "mock: design exceeds LUT capacity",
        )

    def _seed_jitter(self, design: Design, options: FlowOptions) -> float:
        """Deterministic +/-3% jitter as a function of seed and design."""
        h = hashlib.sha1(f"{design.rtl_hash()}:{options.seed}".encode()).digest()
        frac = h[0] / 255.0  # 0..1
        return 0.97 + 0.06 * frac
