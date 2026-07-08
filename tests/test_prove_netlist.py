"""Tests for netlist-level equivalence (RTL == vendor post-impl netlist).

A Vivado ``write_verilog -mode funcsim`` netlist uses Xilinx UNISIM primitives
(LUT*, FD*, CARRY4, ...). yosys ships behavioral models of exactly those, so we
can prove RTL == netlist without any vendor tools. We validate the whole
pipeline by having yosys itself emit a Xilinx-primitive netlist via
``synth_xilinx`` -- the same primitives Vivado's funcsim netlist carries.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from fpgaforge import devices
from fpgaforge.backends.base import Design
from fpgaforge.emulator.emulator import Emulator


# ------------------------- registry / sim-lib mapping -------------------- #
def test_vendor_families_have_a_sim_lib():
    for target in ("xc7a35t", "xc7z020", "xczu3eg", "ecp5_45k",
                   "cyclone10lp_10cl025", "max10_10m50"):
        assert devices.get(target).sim_lib, f"{target} missing sim_lib"


def test_ice40_needs_no_sim_lib():
    # iCE40 is proved at the *bitstream* tier, not via a Verilog sim library.
    assert devices.get("ice40_up5k").sim_lib == ""


def test_netlist_script_reads_rtl_and_netlist_behaviorally():
    em = Emulator()
    d = Design(rtl_files=("examples/counter.v",), top="counter", target="xc7a35t")
    s = em._netlist_equiv_script(d, "net.v", "+/xilinx/cells_sim.v", "induct", 20)
    assert "read_verilog +/xilinx/cells_sim.v" in s   # behavioral, not -lib
    assert "miter -equiv -flatten -make_assert gold gate miter" in s
    assert "tempinduct" in s                            # unbounded proof mode
    # RTL and netlist are resolved to absolute paths (yosys runs in a workdir).
    assert str(Path("examples/counter.v").resolve()) in s
    assert str(Path("net.v").resolve()) in s


def test_missing_netlist_is_a_clean_error():
    em = Emulator()
    d = Design(rtl_files=("examples/counter.v",), top="counter", target="xc7a35t")
    r = em.prove_netlist_equivalence(d, "does/not/exist.v")
    assert r.equivalent is None
    assert "not found" in (r.error or "")
    assert r.against == "netlist"


def test_unknown_family_without_sim_lib_errors(tmp_path):
    em = Emulator()
    nl = tmp_path / "n.v"
    nl.write_text("module counter(); endmodule\n")
    d = Design(rtl_files=("examples/counter.v",), top="counter", target="ice40_up5k")
    r = em.prove_netlist_equivalence(d, nl)
    assert r.equivalent is None
    assert "no yosys cell library" in (r.error or "")


# ------------------------- tool-gated integration ------------------------ #
def _have_xilinx_lib() -> bool:
    if shutil.which("yosys") is None:
        return False
    p = subprocess.run(["yosys", "-p", "read_verilog -lib +/xilinx/cells_sim.v"],
                       capture_output=True, text=True)
    return p.returncode == 0


@pytest.mark.skipif(not _have_xilinx_lib(),
                    reason="requires yosys with bundled xilinx cells_sim")
def test_proves_rtl_equals_xilinx_netlist(tmp_path):
    net = tmp_path / "net_xil.v"
    subprocess.run(
        ["yosys", "-q", "-p",
         f"read_verilog examples/counter.v; synth_xilinx -top counter; "
         f"write_verilog -noattr {net}"],
        check=True, capture_output=True, text=True,
    )
    assert net.exists()

    r = Emulator().prove_netlist_equivalence(
        Design(rtl_files=("examples/counter.v",), top="counter", target="xc7a35t"),
        net, strategy="sat", workdir=tmp_path / "prove",
    )
    assert r.error is None, r.error
    assert r.equivalent is True
    assert r.against == "netlist"
    assert r.unbounded is True          # proved for all inputs, all time
    assert r.method == "induction"
    assert "post-implementation netlist" in r.summary()


@pytest.mark.skipif(not _have_xilinx_lib(),
                    reason="requires yosys with bundled xilinx cells_sim")
def test_prove_public_api_routes_to_netlist(tmp_path):
    from fpgaforge.emulator.emulator import prove

    net = tmp_path / "net_xil.v"
    subprocess.run(
        ["yosys", "-q", "-p",
         f"read_verilog examples/counter.v; synth_xilinx -top counter; "
         f"write_verilog -noattr {net}"],
        check=True, capture_output=True, text=True,
    )
    r = prove("examples/counter.v", "counter", target_fpga="xc7a35t",
              netlist=str(net), strategy="sat", workdir=tmp_path / "prove")
    assert r.against == "netlist"
    assert r.equivalent is True
