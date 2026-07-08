"""Tests for pin-constraint parsing and validation (the board-level gate)."""

from dataclasses import dataclass

import pytest

from fpgaforge.pins import check_pins, load_pins


@dataclass
class P:
    name: str
    width: int = 1
    direction: str = "input"


PORTS = [P("clk"), P("rst"), P("count", 8, "output")]


def _write(tmp_path, name, text):
    f = tmp_path / name
    f.write_text(text)
    return f


# ------------------------------- parsers --------------------------------- #
def test_parse_pcf(tmp_path):
    pc = load_pins(_write(tmp_path, "a.pcf",
                          "# comment\nset_io clk 35\nset_io --warn-no-port count[0] 11\n"))
    assert pc.fmt == "pcf"
    m = pc.by_port()
    assert m["clk"].pin == "35"
    assert m["count[0]"].pin == "11"


def test_parse_xdc(tmp_path):
    pc = load_pins(_write(tmp_path, "a.xdc", """
set_property PACKAGE_PIN E3 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
set_property -dict { PACKAGE_PIN H5 IOSTANDARD LVCMOS33 } [get_ports {count[0]}]
"""))
    m = pc.by_port()
    assert m["clk"].pin == "E3" and m["clk"].io_standard == "LVCMOS33"
    assert m["count[0]"].pin == "H5" and m["count[0]"].io_standard == "LVCMOS33"


def test_parse_qsf(tmp_path):
    pc = load_pins(_write(tmp_path, "a.qsf", """
set_location_assignment PIN_R8 -to clk
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to clk
set_location_assignment PIN_A2 -to count[0]
"""))
    m = pc.by_port()
    assert m["clk"].pin == "R8" and m["clk"].io_standard == "3.3-V LVTTL"
    assert m["count[0]"].pin == "A2"


def test_parse_lpf(tmp_path):
    pc = load_pins(_write(tmp_path, "a.lpf", """
LOCATE COMP "clk" SITE "G2";
IOBUF PORT "clk" IO_TYPE=LVCMOS33;
LOCATE COMP "count[0]" SITE "B2";
"""))
    m = pc.by_port()
    assert m["clk"].pin == "G2" and m["clk"].io_standard == "LVCMOS33"


def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError):
        load_pins(_write(tmp_path, "a.txt", "set_io clk 35\n"))


# ------------------------------ validation ------------------------------- #
def _full_pcf(tmp_path):
    lines = ["set_io clk 35", "set_io rst 10"]
    lines += [f"set_io count[{i}] {11 + i}" for i in range(8)]
    return load_pins(_write(tmp_path, "full.pcf", "\n".join(lines)))


def test_complete_pin_map_is_ok(tmp_path):
    rep = check_pins(_full_pcf(tmp_path), PORTS)
    assert rep.ok
    assert rep.constrained_ports == 10 and rep.total_port_bits == 10


def test_missing_port_is_an_error(tmp_path):
    pc = load_pins(_write(tmp_path, "p.pcf", "set_io clk 35\nset_io rst 10\n"))
    rep = check_pins(pc, PORTS)
    assert not rep.ok
    assert any("count" in e and "no pin assignment" in e for e in rep.errors)


def test_partial_bus_is_an_error(tmp_path):
    pc = load_pins(_write(tmp_path, "p.pcf",
                          "set_io clk 35\nset_io rst 10\nset_io count[0] 11\n"))
    rep = check_pins(pc, PORTS)
    assert any("bits" in e for e in rep.errors)


def test_double_booked_pin_is_an_error(tmp_path):
    pc = _full_pcf(tmp_path)
    pc.assignments[1].pin = "35"  # rst on the clk pin
    rep = check_pins(pc, PORTS)
    assert any("assigned to both" in e for e in rep.errors)


def test_unknown_port_and_bad_pin(tmp_path):
    pc = load_pins(_write(tmp_path, "p.pcf", "set_io nosuch 35\n"))
    rep = check_pins(pc, PORTS)
    assert any("unknown port" in e for e in rep.errors)

    rep = check_pins(_full_pcf(tmp_path), PORTS, valid_pins={"35", "10"})
    assert any("does not exist on this package" in e for e in rep.errors)


def test_board_clock_source_check(tmp_path):
    board = {"clock_sources": [{"pin": "35", "mhz": 12.0}]}
    rep = check_pins(_full_pcf(tmp_path), PORTS, board=board,
                     clock_port="clk", clock_ns=84.0)
    assert rep.ok  # 12 MHz source covers an ~11.9 MHz target

    # Clock pinned somewhere with no board clock -> hard error.
    pc = _full_pcf(tmp_path)
    pc.by_port()["clk"].pin = "44"
    rep = check_pins(pc, PORTS, board=board, clock_port="clk", clock_ns=84.0)
    assert any("no clock" in e for e in rep.errors)

    # Board clock too slow for the requested Fmax -> warning (PLL needed).
    rep = check_pins(_full_pcf(tmp_path), PORTS, board=board,
                     clock_port="clk", clock_ns=10.0)
    assert rep.ok and any("PLL" in w for w in rep.warnings)


def test_board_rail_check(tmp_path):
    pc = load_pins(_write(tmp_path, "a.xdc", """
set_property -dict { PACKAGE_PIN E3 IOSTANDARD LVCMOS33 } [get_ports clk]
"""))
    ports = [P("clk")]
    ok_board = {"rails": [{"volts": 3.3}]}
    assert check_pins(pc, ports, board=ok_board).ok
    bad_board = {"rails": [{"volts": 1.8}]}
    rep = check_pins(pc, ports, board=bad_board)
    assert any("no 3.3 V rail" in e for e in rep.errors)
