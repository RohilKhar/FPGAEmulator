"""Tests for the SMT (memory-array) formal proof engine and engine selection."""

import shutil
from pathlib import Path

import pytest

from fpgaforge.backends.base import Design
from fpgaforge.emulator.emulator import Emulator, _Artifacts


def _design(rtl):
    return Design(rtl_files=(rtl,), top="x")


# ------------------------- engine selection (no tools) ------------------- #
def test_detects_memory_array_declaration():
    em = Emulator()
    assert em._design_has_memory(_design("examples/ram_sync.v"), None) is True
    assert em._design_has_memory(_design("examples/counter.v"), None) is False


def test_select_engine_prefers_smt_for_memory(monkeypatch):
    em = Emulator()
    monkeypatch.setattr(em, "_smt_available", lambda: True)
    mem = _design("examples/ram_sync.v")
    logic = _design("examples/counter.v")
    assert em._select_engine("auto", mem, None) == "smt"
    assert em._select_engine("auto", logic, None) == "sat"
    # Explicit choices are honored regardless of design.
    assert em._select_engine("sat", mem, None) == "sat"
    assert em._select_engine("smt", logic, None) == "smt"


def test_select_engine_falls_back_to_sat_without_smt_tools(monkeypatch):
    em = Emulator()
    monkeypatch.setattr(em, "_smt_available", lambda: False)
    assert em._select_engine("auto", _design("examples/ram_sync.v"), None) == "sat"


def test_smt_script_keeps_memory_as_arrays():
    em = Emulator()
    art = _Artifacts(
        ports=[], rtl_files=["examples/ram_sync.v"],
        mapped_v=Path("m.v"), recon_v=Path("recon.v"), wrapper_v=Path("wrap.v"),
        cells_sim=Path("cells_sim.v"), bitstream_path="out.bin",
    )
    script = em._smt_equiv_script("ram_sync", art, Path("miter.smt2"))
    assert "write_smt2" in script
    assert "setundef -params -zero" in script      # zero-init memory contents
    assert "memory_map" not in script              # never bit-blast the memory
    assert "\nsat " not in script                  # SMT path, not the SAT solver


# ------------------------- tool-gated integration ------------------------ #
_TOOLS = ["yosys", "nextpnr-ice40", "icepack", "iceunpack", "icebox_vlog",
          "yosys-smtbmc"]
_HAVE_SOLVER = any(shutil.which(s) for s in ("z3", "yices-smt2", "boolector"))


@pytest.mark.skipif(
    any(shutil.which(t) is None for t in _TOOLS) or not _HAVE_SOLVER,
    reason="requires the iCE40 toolchain + yosys-smtbmc + an SMT solver",
)
def test_smt_proves_memory_design_equivalent():
    from fpgaforge.emulator.emulator import prove

    # A shallow bounded proof: the SAT engine cannot resolve the $mem at all,
    # while the SMT engine proves equivalence over a few cycles from reset.
    r = prove("examples/ram_sync.v", "ram_sync", depth=2, strategy="smt",
              workdir=".runs/test_prove_smt")
    assert r.engine == "smt"
    assert r.error is None, r.error
    assert r.equivalent is True
    assert r.method == "bmc"
