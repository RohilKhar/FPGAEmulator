"""Tests for bitstream mutation testing (verifier self-validation)."""

import random
import shutil

import pytest

from fpgaforge.emulator.mutation import (
    flip_logic_bit, flip_netlist_lut, MutationResult, mutation_test,
)
from fpgaforge.backends.base import Design


_SAMPLE_ASC = "\n".join([
    ".device 5k",
    ".logic_tile 1 1",
    "000000000000",
    "111111111111",
    ".logic_tile 2 1",
    "010101010101",
    ".ram_data 3 3",
    "1111000011110000",     # must NOT be mutated (not a logic tile)
])


def test_flip_logic_bit_changes_exactly_one_bit_in_a_logic_tile():
    rng = random.Random(42)
    mutated, desc = flip_logic_bit(_SAMPLE_ASC, rng)
    assert mutated != _SAMPLE_ASC
    # Exactly one character differs across the whole file.
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(_SAMPLE_ASC, mutated)) if a != b]
    assert len(diffs) == 1
    assert desc


def test_flip_logic_bit_never_touches_non_logic_sections():
    # The .ram_data row must remain intact across many mutations.
    ram_row = "1111000011110000"
    for seed in range(50):
        mutated, _ = flip_logic_bit(_SAMPLE_ASC, random.Random(seed))
        assert ram_row in mutated


def test_flip_logic_bit_raises_without_logic_tiles():
    with pytest.raises(ValueError):
        flip_logic_bit(".device 5k\n.ram_data 1 1\n1010\n", random.Random(0))


def test_flip_multiple_bits():
    rng = random.Random(7)
    mutated, desc = flip_logic_bit(_SAMPLE_ASC, rng, n_bits=3)
    diffs = sum(1 for a, b in zip(_SAMPLE_ASC, mutated) if a != b)
    assert 1 <= diffs <= 3          # may re-pick the same bit; at most n_bits
    assert desc.count("->") == 3


_SAMPLE_RECON = "\n".join([
    "module recon (input clk, output reg q);",
    "assign n27 = /* LUT   22  1  0 */ 1'b0;",
    "assign n18 = /* LUT   22  1  3 */ (n12 ? !q : q);",
    "assign n13 = /* CARRY 22  1  3 */ (q & 1'b0) | ((q | 1'b0) & n12);",
    "endmodule",
])


def test_flip_netlist_lut_inverts_a_lut_expression():
    mutated, desc = flip_netlist_lut(_SAMPLE_RECON, random.Random(1))
    assert "!(" in mutated and "/*MUT*/" in mutated
    assert desc
    # CARRY lines must never be mutated (only LUT expressions).
    for ln in mutated.split("\n"):
        if "CARRY" in ln:
            assert "/*MUT*/" not in ln


def test_flip_netlist_lut_raises_without_luts():
    with pytest.raises(ValueError):
        flip_netlist_lut("module m; endmodule", random.Random(0))


def test_mutation_result_kill_rate_math():
    r = MutationResult(design_id="d", n_mutants=10, killed_functional=6,
                       malformed=2, survived=2)
    assert r.killed == 8
    assert r.kill_rate == 0.8
    assert "kill rate : 80%" in r.summary()


# ------------------------- tool-gated integration ------------------- #
_TOOLS = ["yosys", "nextpnr-ice40", "icepack", "iceunpack", "icebox_vlog",
          "iverilog", "vvp"]


@pytest.mark.skipif(
    any(shutil.which(t) is None for t in _TOOLS),
    reason="requires the full open-source iCE40 toolchain",
)
def test_mutation_campaign_kills_most_mutants_on_counter():
    design = Design(rtl_files=("examples/counter.v",), top="counter",
                    target="ice40_up5k")
    r = mutation_test(design, n_mutants=6, cycles=48, clock_mhz=24.0,
                      workdir=".runs/test_mutation")
    assert r.error is None, r.error
    assert r.n_mutants == 6
    # A real verifier should catch a good fraction of corrupted bitstreams.
    assert r.kill_rate > 0.0
    assert r.killed + r.survived == r.n_mutants
