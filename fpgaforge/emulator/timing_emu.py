"""Timing-accurate (delay-aware) emulation.

Functional emulation (``verify_bitstream``) proves the flashed fabric computes
the right values with *zero* delay. But real silicon has delay: at a given clock
a flop's data must arrive before its setup deadline, or it latches a stale value
and the design silently misbehaves -- something a zero-delay run cannot see.

This fuses two views into one verdict:

1. **functional** -- does the reconstructed bitstream match the design cycle by
   cycle (the existing X-aware verify)?
2. **timing** -- do the *real routed delays* (from the SDF nextpnr emits) let
   every capture flop meet setup at the chosen clock? Computed by an independent
   longest-path solver over the SDF (see :mod:`fpgaforge.sdf_timing`), which also
   pinpoints the failing endpoint.

Only when the fabric is both functionally correct *and* settles at the target
clock is it "timing-accurate PASS" -- i.e. it will behave on hardware at speed.
Flop power-up state is modeled explicitly (iCE40 flops come up at 0), which is
the concrete refinement of the design's uninitialized ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class TimingEmulationResult:
    design_id: str
    clock_mhz: float = 0.0
    functional_match: bool = False
    functional_confidence: float = 0.0
    emulated_fmax_mhz: float = 0.0
    settles: bool = False
    slack_ns: float = 0.0
    worst_endpoint: str = ""
    worst_path: list[str] = field(default_factory=list)
    n_launch: int = 0
    n_endpoints: int = 0
    ff_init_state: str = "0 (iCE40 power-up)"
    sdf_path: str | None = None
    bitstream_path: str | None = None
    workdir: str | None = None
    log: str = ""
    error: str | None = None

    @property
    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        if not self.functional_match:
            return "FUNCTIONAL MISMATCH"
        if not self.settles:
            return "SETUP VIOLATION"
        return "TIMING-ACCURATE PASS"

    def summary(self) -> str:
        lines = [
            f"timing-accurate emulation: {self.verdict}",
            f"design    : {self.design_id}",
            f"clock     : {self.clock_mhz:.1f} MHz "
            f"(emulated Fmax {self.emulated_fmax_mhz:.1f} MHz from routed delays)",
        ]
        if self.functional_match:
            lines.append(
                f"functional: MATCH (empirical confidence {self.functional_confidence:.0%}), "
                f"flops init to {self.ff_init_state}"
            )
        else:
            lines.append("functional: MISMATCH (zero-delay behavior already differs)")
        lines.append(
            f"timing    : {'settles' if self.settles else 'DOES NOT settle'} at "
            f"{self.clock_mhz:.1f} MHz, slack {self.slack_ns:+.3f} ns"
        )
        if not self.settles and self.worst_endpoint:
            lines.append(f"  failing endpoint: {self.worst_endpoint}")
            if self.worst_path:
                head = " -> ".join(self.worst_path[:4])
                lines.append(f"  path: {head}{' -> ...' if len(self.worst_path) > 4 else ''}")
            lines.append(
                "  fix : this flop misses setup -> pipeline the path, reduce logic "
                "depth, or lower the clock"
            )
        if self.error:
            lines.append(f"error     : {self.error}")
        return "\n".join(lines)


def timing_emulate(
    rtl: str | Sequence[str],
    top: str,
    target_fpga: str = "ice40_up5k",
    clock_mhz: float = 50.0,
    cycles: int = 48,
    stimulus: str = "counter",
    seeds: Sequence[int] | None = None,
    workdir: str | Path = ".runs/timing_emu",
    engine=None,
) -> TimingEmulationResult:
    """Run delay-aware emulation: functional match + real-delay setup at ``clock_mhz``.

    Builds the bitstream once (with SDF), verifies functional equivalence, and
    checks every capture flop meets setup at the chosen clock using the routed
    delays. Reports a fused verdict and, on a setup violation, the failing
    endpoint and path.
    """
    from ..timing import signoff as timing_signoff
    from ..sdf_timing import analyze_sdf
    from .emulator import Emulator, verify_bitstream

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    rtl_files = [rtl] if isinstance(rtl, str) else list(rtl)
    from ..backends.base import Design

    design = Design(rtl_files=tuple(rtl_files), top=top, target=target_fpga)
    res = TimingEmulationResult(design_id=design.design_id(), clock_mhz=clock_mhz,
                                workdir=str(workdir))

    # 1. Route with SDF so we have the real routed delays.
    clock_ns = 1000.0 / clock_mhz if clock_mhz > 0 else 10.0
    trep = timing_signoff(rtl_files, top, target_fpga=target_fpga, clock_ns=clock_ns,
                          workdir=str(workdir / "route"))
    if trep.error or not trep.sdf_path:
        res.error = trep.error or "no SDF produced (need nextpnr with timing artifacts)"
        return res
    res.sdf_path = trep.sdf_path

    # 2. Independent longest-path timing from the SDF.
    sta = analyze_sdf(trep.sdf_path)
    res.emulated_fmax_mhz = sta.fmax_mhz
    res.n_launch = sta.n_launch
    res.n_endpoints = sta.n_endpoints
    res.slack_ns = sta.slack_ns_at(clock_mhz)
    res.settles = sta.settles_at(clock_mhz)
    if sta.worst:
        res.worst_endpoint = sta.worst.endpoint
        res.worst_path = sta.worst.path

    # 3. Functional match (zero-delay cycle-accurate verify).
    engine = engine or Emulator()
    vres = verify_bitstream(
        rtl_files, top, target_fpga=target_fpga, cycles=cycles,
        clock_mhz=min(clock_mhz, sta.fmax_mhz or clock_mhz),
        stimulus=stimulus, seeds=seeds, workdir=str(workdir / "verify"),
        engine=engine,
    )
    res.log += vres.log
    res.bitstream_path = vres.bitstream_path
    if vres.error:
        res.error = vres.error
        return res
    res.functional_match = vres.matches
    res.functional_confidence = vres.confidence
    return res
