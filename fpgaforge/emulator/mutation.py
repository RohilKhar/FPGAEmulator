"""Mutation testing: validate the *verifier itself*.

``verify_bitstream`` reports an empirical confidence from coverage saturation --
but that only measures how much stimulus was applied, not whether the comparison
would actually *catch* a corrupted bitstream. Mutation testing closes that gap:
take the known-good bitstream, flip a configuration bit (a LUT truth-table /
tile bit), reconstruct that mutated fabric, and check the verifier flags it as
different from the RTL.

The **kill rate** -- fraction of mutants detected -- is a far more defensible
confidence signal than coverage alone: it demonstrates the comparison has teeth.
Surviving mutants are expected (a flipped bit on unused/redundant config changes
nothing observable), but a *high* survival rate means the stimulus is too weak
to distinguish behaviors, which is exactly what you want to know.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# `assign n27 = /* LUT   22  1  0 */ 1'b0;`  (icebox_vlog LUT expression)
_LUT_ASSIGN_RE = re.compile(r"^(\s*assign\s+\S+\s*=\s*)(/\*\s*LUT[^*]*\*/)(\s*)(.+?)(;\s*)$")


def flip_netlist_lut(verilog: str, rng: random.Random) -> tuple[str, str]:
    """Invert one reconstructed LUT expression -> a valid, functionally-different
    netlist. Unlike a raw bitstream bit flip (which usually corrupts routing and
    fails to reconstruct), this always compiles, so it genuinely exercises the
    cycle-accurate comparator. Returns ``(mutated_verilog, description)``.
    """
    lines = verilog.split("\n")
    lut_lines = [i for i, ln in enumerate(lines) if _LUT_ASSIGN_RE.match(ln)]
    if not lut_lines:
        raise ValueError("no reconstructed LUT expressions to mutate")
    idx = rng.choice(lut_lines)
    m = _LUT_ASSIGN_RE.match(lines[idx])
    head, tag, sp, expr, tail = m.groups()
    # Logical inversion keeps the result 1-bit and always changes the function.
    lines[idx] = f"{head}{tag} /*MUT*/{sp}!({expr}){tail}"
    return "\n".join(lines), f"line {idx}: LUT {expr.strip()} -> !(...)"


def flip_logic_bit(asc_text: str, rng: random.Random, n_bits: int = 1) -> tuple[str, str]:
    """Flip ``n_bits`` random configuration bit(s) inside ``.logic_tile`` blocks.

    Pure and testable: returns ``(mutated_asc, description)``. Raises if the
    bitstream has no logic-tile content to mutate.
    """
    lines = asc_text.split("\n")
    in_logic = False
    candidates: list[int] = []
    for i, ln in enumerate(lines):
        if ln.startswith("."):
            in_logic = ln.startswith(".logic_tile")
            continue
        if in_logic and ln and all(c in "01" for c in ln):
            candidates.append(i)
    if not candidates:
        raise ValueError("no logic-tile configuration bits to mutate")

    notes = []
    for _ in range(max(1, n_bits)):
        idx = rng.choice(candidates)
        row = list(lines[idx])
        pos = rng.randrange(len(row))
        old = row[pos]
        row[pos] = "1" if old == "0" else "0"
        lines[idx] = "".join(row)
        notes.append(f"line {idx} col {pos}: {old}->{row[pos]}")
    return "\n".join(lines), "; ".join(notes)


@dataclass
class MutationResult:
    design_id: str
    n_mutants: int = 0
    killed_functional: int = 0     # reconstructed, simulated, and caught as different
    malformed: int = 0             # mutation broke reconstruction/compile (detected)
    survived: int = 0              # simulated but behaved identically (verifier blind)
    details: list[str] = field(default_factory=list)
    bitstream_path: str | None = None
    workdir: str | None = None
    log: str = ""
    error: str | None = None

    @property
    def killed(self) -> int:
        return self.killed_functional + self.malformed

    @property
    def kill_rate(self) -> float:
        return self.killed / self.n_mutants if self.n_mutants else 0.0

    def summary(self) -> str:
        if self.error:
            return f"mutation testing: ERROR - {self.error}"
        lines = [
            "mutation testing (verifier self-validation)",
            f"design    : {self.design_id}",
            f"mutants   : {self.n_mutants}",
            f"killed    : {self.killed} ({self.killed_functional} functional, "
            f"{self.malformed} malformed)",
            f"survived  : {self.survived}",
            f"kill rate : {self.kill_rate:.0%}",
        ]
        if self.survived:
            lines.append(
                "note      : survivors are bit flips with no observable effect "
                "(unused/redundant config) or stimulus too weak to distinguish"
            )
        return "\n".join(lines)


def mutation_test(
    design,
    n_mutants: int = 8,
    cycles: int = 64,
    clock_mhz: float = 50.0,
    seeds: Sequence[int] | None = None,
    base_seed: int = 0xC0FFEE,
    n_bits: int = 1,
    strategy: str = "netlist",
    config=None,
    workdir: str | Path = ".runs/mutation",
    engine=None,
) -> MutationResult:
    """Run a mutation campaign to validate the verifier's sensitivity.

    Builds the golden bitstream once, then for each mutant injects a fault and
    checks the cycle-accurate comparison catches it. Two strategies:

    * ``"netlist"`` (default): invert a reconstructed LUT expression -> a valid,
      simulatable, functionally-different fabric. This directly tests whether the
      comparator has teeth (killed vs survived).
    * ``"bitstream"``: flip raw ``.logic_tile`` bit(s). More realistic corruption,
      but most flips hit routing and fail to reconstruct (counted as detected).

    Returns the kill rate.
    """
    from ..virtual.board import BringUpConfig
    from .emulator import Emulator, _cycle_trace, _compare_traces_x, render_compare_tb
    from . import netlist as nl

    engine = engine or Emulator()
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    res = MutationResult(design_id=design.design_id(), workdir=str(workdir))

    cfg = config or BringUpConfig(cycles=cycles, stimulus="random")
    cfg.cycles = cycles
    art, err, log = engine._prepare_artifacts(design, cfg, workdir, clock_mhz)
    res.log += log
    if err is not None:
        res.error = err
        return res
    res.bitstream_path = art.bitstream_path

    golden_asc_path = workdir / "roundtrip.asc"
    if not golden_asc_path.exists():
        res.error = "golden bitstream .asc not found for mutation"
        return res
    golden_asc = golden_asc_path.read_text()
    golden_recon = Path(art.recon_v).read_text() if Path(art.recon_v).exists() else ""

    # Compile the RTL (reference) DUT once and capture its golden traces.
    tb_v = workdir / "mut_tb.v"
    tb_v.write_text(render_compare_tb(design.top, art.ports, cfg))
    run_seeds = list(seeds) if seeds else [1, 1337, 424242]

    rtl_vvp = workdir / "rtl.vvp"
    ok, clog = engine._compile(rtl_vvp, [tb_v, art.mapped_v, art.cells_sim], workdir)
    res.log += "\n" + clog
    if not ok:
        res.error = "failed to compile reference RTL design"
        return res
    rtl_traces: dict[int, list[str]] = {}
    for s in run_seeds:
        _, out = engine._run_vvp(rtl_vvp, workdir, [f"+SEED={s}"])
        rtl_traces[s] = _cycle_trace(out)

    rng = random.Random(base_seed)
    for m in range(n_mutants):
        res.n_mutants += 1
        recon_v = workdir / f"mutant_{m}.v"
        if strategy == "netlist":
            try:
                mut_v, desc = flip_netlist_lut(golden_recon, rng)
            except ValueError as exc:
                res.error = str(exc)
                return res
            recon_v.write_text(mut_v)
        else:
            mut_asc, desc = flip_logic_bit(golden_asc, rng, n_bits=n_bits)
            mut_asc_path = workdir / f"mutant_{m}.asc"
            mut_asc_path.write_text(mut_asc)
            try:
                nl.reconstruct(mut_asc_path, recon_v, module="recon",
                               icebox_vlog=engine.icebox_vlog)
            except Exception as exc:  # noqa: BLE001 - malformed mutant is "detected"
                res.malformed += 1
                res.details.append(f"mutant {m} [{desc}]: malformed ({exc})")
                continue

        bit_vvp = workdir / f"mutant_{m}.vvp"
        ok, clog = engine._compile(
            bit_vvp, [tb_v, recon_v, art.wrapper_v, art.cells_sim], workdir
        )
        if not ok:
            res.malformed += 1
            res.details.append(f"mutant {m} [{desc}]: did not compile")
            continue

        killed = False
        for s in run_seeds:
            _, out = engine._run_vvp(bit_vvp, workdir, [f"+SEED={s}"])
            stats = _compare_traces_x(rtl_traces[s], _cycle_trace(out))
            if not stats.matches:
                killed = True
                res.details.append(f"mutant {m} [{desc}]: killed ({stats.first_mismatch})")
                break
        if killed:
            res.killed_functional += 1
        else:
            res.survived += 1
            res.details.append(f"mutant {m} [{desc}]: survived")

    return res
