import shutil

import pytest

from fpgaforge.emulator import bitstream as bsmod
from fpgaforge.emulator import fabric as fabmod
from fpgaforge.emulator import netlist as nl
from fpgaforge.emulator.emulator import _compare_traces
from fpgaforge.virtual.board import Port


# --------------------------- pure: bitstream parse --------------------------- #
def _blank_logic_rows():
    return ["0" * 54 for _ in range(16)]


def _set(rows, r, c, v="1"):
    rows[r] = rows[r][:c] + v + rows[r][c + 1 :]


def test_parse_asc_tiles_and_device():
    rows = _blank_logic_rows()
    text = ".device 5k\n.logic_tile 1 1\n" + "\n".join(rows) + "\n"
    text += ".io_tile 1 0\n" + "\n".join("0" * 18 for _ in range(16)) + "\n"
    bs = bsmod.parse_asc(text)
    assert bs.device == "5k"
    assert len(bs.logic_tiles()) == 1
    assert len(bs.tiles_of("io")) == 1
    assert bs.logic_tiles()[0].rows[0] == "0" * 54


# --------------------------- pure: fabric decode ---------------------------- #
def test_decode_lut_bit0():
    # LUT truth-table bit 0 comes from raw index _LUT_PERM[0] == 4 -> row 0, col 40.
    rows = _blank_logic_rows()
    _set(rows, 0, 36 + 4)  # cell 0, raw bit 4
    tile = bsmod.Tile(kind="logic", x=1, y=1, rows=rows)
    cell0 = fabmod.decode_logic_tile(tile)[0]
    assert cell0.lut_init == 0x0001
    assert cell0.truth_table()[0] == 1


def test_decode_dff_and_carry_enable():
    rows = _blank_logic_rows()
    _set(rows, 0, 36 + 9)   # cell 0 DffEnable  (raw idx 9)
    _set(rows, 0, 36 + 8)   # cell 0 CarryEnable(raw idx 8)
    tile = bsmod.Tile(kind="logic", x=1, y=1, rows=rows)
    cell0 = fabmod.decode_logic_tile(tile)[0]
    assert cell0.dff_enable is True
    assert cell0.carry_enable is True


def test_decode_cell_index_uses_correct_rows():
    # Cell 3 uses rows 6 and 7; setting a bit in cell 0's rows must not leak.
    rows = _blank_logic_rows()
    _set(rows, 6, 36 + 4)  # cell 3, raw bit 4 -> lut bit 0
    tile = bsmod.Tile(kind="logic", x=1, y=1, rows=rows)
    cells = fabmod.decode_logic_tile(tile)
    assert cells[0].lut_init == 0
    assert cells[3].lut_init == 0x0001


def test_lut_equation_constants():
    c = fabmod.LogicCell(0, 0, 0, lut_init=0xFFFF, carry_enable=False,
                         dff_enable=False, set_noreset=False, async_setreset=False)
    assert c.lut_equation() == "1"
    c0 = fabmod.LogicCell(0, 0, 0, lut_init=0, carry_enable=False,
                          dff_enable=False, set_noreset=False, async_setreset=False)
    assert c0.lut_equation() == "0"


# --------------------------- pure: netlist helpers -------------------------- #
def test_normalize_netlist_output_reg_and_port_wire():
    src = (
        "module m (input clk, output \\count[0] , input \\a[0] );\n"
        "wire \\a[0] ;\n"
        "reg \\count[0] = 0 ;\n"
        "endmodule\n"
    )
    out = nl.normalize_netlist(src)
    assert "output reg \\count[0]" in out
    assert "wire \\a[0] ;" not in out          # redundant port wire dropped
    assert "initial \\count[0] = 0;" in out     # reg init -> initial


def test_expand_bits_and_generate_pcf():
    ports = [Port("clk", "input", 1), Port("y", "output", 2)]
    assert nl.expand_bits(ports) == ["clk", "y[0]", "y[1]"]
    pcf = nl.generate_pcf(ports, ["35", "34", "2"])
    assert pcf == "set_io clk 35\nset_io y[0] 34\nset_io y[1] 2\n"


def test_generate_pcf_overflow():
    ports = [Port("a", "input", 8)]
    with pytest.raises(RuntimeError):
        nl.generate_pcf(ports, ["1", "2"])


def test_make_rebus_wrapper():
    ports = [Port("clk", "input", 1), Port("rst", "input", 1), Port("count", "output", 8)]
    w = nl.make_rebus_wrapper("recon", "counter", ports)
    assert "module counter (clk, rst, count);" in w
    assert "output [7:0] count;" in w
    assert ".clk(clk)" in w
    assert ".\\count[0] (count[0])" in w
    assert ".\\count[7] (count[7])" in w


def test_compare_traces():
    a = ["CYC 0 y=1", "CYC 1 y=2"]
    assert _compare_traces(a, list(a)) == (True, None)
    ok, msg = _compare_traces(a, ["CYC 0 y=1", "CYC 1 y=9"])
    assert ok is False and "cycle 1" in msg


def test_compare_traces_x_aware_dontcare():
    """Unknown (x/z) design values are don't-cares the bitstream may refine."""
    from fpgaforge.emulator.emulator import _compare_traces_x

    # RTL reads uninitialized memory as x; bitstream powers up to 0.
    rtl = ["CYC 0 dout=x", "CYC 1 dout=x", "CYC 2 dout=5"]
    bit = ["CYC 0 dout=0", "CYC 1 dout=0", "CYC 2 dout=5"]
    stats = _compare_traces_x(rtl, bit)
    assert stats.matches is True
    assert stats.compared == 1          # only the concrete cycle counts
    assert stats.indeterminate == 2     # the two x cycles are skipped

    # A real divergence on a concrete cycle must still fail.
    bad = ["CYC 0 dout=0", "CYC 1 dout=0", "CYC 2 dout=7"]
    stats2 = _compare_traces_x(rtl, bad)
    assert stats2.matches is False
    assert "dout design=5 bitstream=7" in stats2.first_mismatch


def test_toggle_coverage_metric():
    from fpgaforge.emulator.emulator import _toggle_coverage

    outputs = [Port("y", "output", 4)]
    # Bit0 toggles (0,1); bit1 toggles (0,2); bits 2-3 never set -> 2/4 = 50%.
    traces = [["CYC 0 y=0", "CYC 1 y=1", "CYC 2 y=2", "CYC 3 y=x"]]
    cov, toggled, total, distinct = _toggle_coverage(traces, outputs)
    assert total == 4
    assert toggled == 2
    assert cov == 0.5
    assert distinct == 3      # x cycle excluded from distinct vectors


def test_verification_confidence_blend():
    from fpgaforge.emulator.emulator import VerificationResult

    weak = VerificationResult(design_id="d", matches=True, total_compared=10,
                              seeds=[1], toggle_coverage=0.2, total_bits=8)
    strong = VerificationResult(design_id="d", matches=True, total_compared=2000,
                                seeds=[1, 2, 3, 4], toggle_coverage=1.0, total_bits=8)
    assert 0.0 < weak.confidence < strong.confidence <= 1.0
    # A mismatch always has zero confidence.
    assert VerificationResult(design_id="d", matches=False,
                              total_compared=999).confidence == 0.0


def test_seed_pool_distinct_and_deterministic():
    from fpgaforge.emulator.emulator import _seed_pool, _DEFAULT_SEEDS

    pool = _seed_pool(12)
    assert len(pool) == 12
    assert len(set(pool)) == 12                      # all distinct
    assert pool[: len(_DEFAULT_SEEDS)] == _DEFAULT_SEEDS
    assert _seed_pool(12) == pool                    # reproducible


def test_saturation_boosts_confidence():
    from fpgaforge.emulator.emulator import VerificationResult

    base = dict(design_id="d", matches=True, total_compared=500,
                seeds=[1, 2, 3, 4, 5], toggle_coverage=0.5, total_bits=8)
    unsat = VerificationResult(**base)
    sat = VerificationResult(coverage_saturated=True, **base)
    assert sat.confidence > unsat.confidence
    assert "saturated" in sat.summary()


def test_render_compare_tb_stimulus_modes():
    from fpgaforge.emulator.emulator import render_compare_tb
    from fpgaforge.virtual.board import BringUpConfig

    ports = [Port("clk", "input", 1), Port("rst", "input", 1),
             Port("a", "input", 4), Port("y", "output", 8)]
    counter_tb = render_compare_tb("adder", ports, BringUpConfig(vcd_path="w.vcd"))
    assert "a <= a + 4'd1;" in counter_tb
    assert '$dumpfile("w.vcd")' in counter_tb    # waveform capture
    assert "CYC %0d" in counter_tb

    rand_tb = render_compare_tb("adder", ports, BringUpConfig(stimulus="random"))
    assert "a <= $random(_seed);" in rand_tb       # seeded stream
    assert "integer _seed;" in rand_tb
    assert '$value$plusargs("SEED=%d", _seed)' in rand_tb


def test_classify_proof():
    from fpgaforge.emulator.emulator import _classify_proof

    assert _classify_proof("Induction step proven: SUCCESS!\n") == "proved"
    assert _classify_proof("SAT proof finished - no model found: SUCCESS!\n") == "proved"
    assert _classify_proof("SAT proof finished - model found: FAIL!\n") == "counterexample"
    assert _classify_proof("Trying induction with length 3\ngave up\n") == "inconclusive"


def test_proof_result_summary():
    from fpgaforge.emulator.emulator import ProofResult

    unbounded = ProofResult(design_id="d", equivalent=True, unbounded=True, method="induction")
    assert "ALL time" in unbounded.summary()
    assert unbounded.proved is True

    bounded = ProofResult(design_id="d", equivalent=True, unbounded=False, depth=20, method="bmc")
    assert "20 cycles" in bounded.summary()

    ne = ProofResult(design_id="d", equivalent=False, method="bmc", counterexample="in_a=5")
    assert "NOT EQUIVALENT" in ne.summary()
    assert ne.proved is False


# --------------------------- tool-gated end-to-end -------------------------- #
_TOOLS = all(
    shutil.which(t)
    for t in ("yosys", "nextpnr-ice40", "icepack", "iceunpack", "icebox_vlog",
              "iverilog", "vvp")
)

_COUNTER = (
    "module counter(input clk, input rst, output reg [7:0] count);\n"
    "  always @(posedge clk) if (rst) count <= 0; else count <= count + 1;\n"
    "endmodule\n"
)


@pytest.mark.skipif(not _TOOLS, reason="requires the full iCE40 open-source flow")
def test_verify_bitstream_matches(tmp_path):
    from fpgaforge.emulator import verify_bitstream

    rtl = tmp_path / "counter.v"
    rtl.write_text(_COUNTER)
    res = verify_bitstream(str(rtl), top="counter", cycles=16,
                           workdir=tmp_path / "v")
    assert res.matches, res.first_mismatch or res.error
    assert res.fabric is not None
    assert res.fabric.dffs_used == 8       # 8-bit counter -> 8 flops
    assert res.bitstream_path is not None


@pytest.mark.skipif(not shutil.which("yosys") or not shutil.which("nextpnr-ice40"),
                    reason="requires yosys + nextpnr-ice40")
def test_multi_file_synthesis(tmp_path):
    # Regression: Ice40Backend must join multiple read_verilog on separate lines.
    from fpgaforge.backends.base import Design, FlowOptions
    from fpgaforge.backends.ice40 import Ice40Backend

    sub = tmp_path / "sub.v"
    sub.write_text(
        "module sub(input a, input b, output y); assign y = a & b; endmodule\n"
    )
    top = tmp_path / "top.v"
    top.write_text(
        "module top(input clk, input a, input b, output reg y);\n"
        "  wire w; sub u(.a(a), .b(b), .y(w));\n"
        "  always @(posedge clk) y <= w;\n"
        "endmodule\n"
    )
    design = Design(rtl_files=(str(top), str(sub)), top="top", target="ice40_up5k")
    res = Ice40Backend().run(design, FlowOptions(), tmp_path / "run")
    assert res.error != "synthesis failed"
    assert res.success


@pytest.mark.skipif(not _TOOLS, reason="requires the full iCE40 open-source flow")
def test_emulate_decodes_bitstream(tmp_path):
    from fpgaforge.emulator import emulate, verify_bitstream

    rtl = tmp_path / "counter.v"
    rtl.write_text(_COUNTER)
    v = verify_bitstream(str(rtl), top="counter", cycles=8, workdir=tmp_path / "v")
    assert v.bitstream_path is not None

    res = emulate(v.bitstream_path, workdir=tmp_path / "e")
    assert res.error is None
    assert res.device == "5k"
    assert res.fabric.dffs_used == 8
    assert res.reconstructed
    assert res.netlist_path is not None


@pytest.mark.skipif(not _TOOLS, reason="requires the full iCE40 open-source flow")
def test_prove_equivalence_unbounded(tmp_path):
    from fpgaforge.emulator import prove

    rtl = tmp_path / "counter.v"
    rtl.write_text(_COUNTER)
    r = prove(str(rtl), top="counter", depth=16, workdir=tmp_path / "p")
    assert r.equivalent is True, r.error or r.counterexample
    assert r.proved
    assert r.unbounded            # temporal induction converges for the counter
    assert r.method == "induction"


# ------------------------- pure: virtual board ------------------------- #
def test_classify_pins_roles():
    from fpgaforge.emulator import peripherals as P

    ports = [
        Port("clk", "input", 1), Port("resetn", "input", 1),
        Port("led", "output", 1), Port("tx", "output", 1),
        Port("rx", "input", 1), Port("btn", "input", 1),
        Port("sw", "input", 4), Port("gpio", "output", 8),
    ]
    roles = P.classify_pins(ports, P.BoardConfig())
    assert roles["clk"] == P.CLOCK
    assert roles["resetn"] == P.RESET
    assert roles["led"] == P.LED
    assert roles["tx"] == P.UART_TX
    assert roles["rx"] == P.UART_RX
    assert roles["btn"] == P.BUTTON
    assert roles["sw"] == P.SWITCH
    assert roles["gpio"] == P.GPIO_OUT
    # Explicit override wins.
    roles2 = P.classify_pins(ports, P.BoardConfig(pin_roles={"gpio": P.LED}))
    assert roles2["gpio"] == P.LED


def test_render_board_tb_wires_peripherals():
    from fpgaforge.emulator import peripherals as P

    ports = [Port("clk", "input", 1), Port("resetn", "input", 1),
             Port("led", "output", 1), Port("tx", "output", 1),
             Port("rx", "input", 1)]
    cfg = P.BoardConfig(clock_mhz=12, baud=1_000_000, uart_rx_bytes=[0x41])
    roles = P.classify_pins(ports, cfg)
    tb = P.render_board_tb("blinky", ports, cfg, roles)
    assert "blinky dut (" in tb
    assert "localparam real BIT_NS = 1000.0000;" in tb   # 1e9/1e6
    assert 'UARTCHAR tx' in tb                            # decode DUT tx
    assert "@(negedge tx)" in tb
    assert "rx = 1'b1;" in tb                             # rx idle-high + injected
    assert "BOARD_DONE" in tb
    # active-low reset asserted then released.
    assert "resetn = 1'b0;" in tb and "resetn = 1'b1;" in tb


def test_parse_board_log_and_capture():
    from fpgaforge.emulator import peripherals as P

    log = (
        "LED led 0 0\n"
        "UARTCHAR tx 72\nUARTCHAR tx 105\nUARTCHAR tx 10\n"
        "LED led 5000 1\nLED led 10000 0\n"
        "GPIO gpio 3000 7\n"
        "BOARD_DONE\n"
    )
    roles = {"clk": P.CLOCK, "led": P.LED, "tx": P.UART_TX, "gpio": P.GPIO_OUT}
    res = P.parse_board_log(log, roles)
    assert res.ran is True
    assert res.uart["tx"].text == "Hi\n"
    assert res.uart["tx"].bytes_ == [72, 105, 10]
    assert res.led_toggle_counts()["led"] == 2
    assert res.led_final()["led"] == 0
    assert "uart[tx]" in res.summary()


def test_uart_capture_nonprintable():
    from fpgaforge.emulator.peripherals import UartCapture

    cap = UartCapture(pin="tx", bytes_=[0x00, 0x48, 0xFF])
    assert cap.text == "\\x00H\\xff"


@pytest.mark.skipif(not _TOOLS, reason="requires the full iCE40 open-source flow")
def test_emulate_board_uart_and_led(tmp_path):
    """End-to-end: run the flashed bitstream against a virtual board."""
    from fpgaforge.emulator import BoardConfig, emulate_board

    rtl = "examples/blinky_uart.v"
    cfg = BoardConfig(clock_mhz=12, baud=1_000_000, duration_us=60)
    res = emulate_board(rtl, top="blinky_uart", target_fpga="ice40_up5k",
                        board=cfg, workdir=tmp_path / "b")
    assert res.error is None, res.error
    assert res.ran
    # The flashed image actually transmits "Hi\n" on its UART.
    assert "Hi" in res.uart["tx"].text
    # And blinks the LED.
    assert res.led_toggle_counts().get("led", 0) > 0
    assert res.bitstream_path is not None
