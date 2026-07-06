"""Unit tests for SDC constraint parsing and structural CDC analysis."""

from fpgaforge.constraints import parse_sdc
from fpgaforge.cdc import analyze_cdc


# ------------------------------ SDC --------------------------------- #
def test_parse_create_clock_multiclock():
    sdc = """
    create_clock -name sysclk -period 10.0 [get_ports clk]
    create_clock -name rxclk  -period 8.0  [get_ports rx_clk]
    """
    con = parse_sdc(sdc)
    assert con.is_multiclock
    assert set(con.clocks) == {"sysclk", "rxclk"}
    assert con.clocks["sysclk"].freq_mhz == 100.0
    assert con.fastest_clock().name == "rxclk"   # 125 MHz


def test_parse_false_and_multicycle_paths():
    sdc = """
    create_clock -name a -period 10 [get_ports ca]
    create_clock -name b -period 10 [get_ports cb]
    set_false_path -from [get_clocks a] -to [get_clocks b]
    set_multicycle_path 2 -from [get_clocks a] -to [get_clocks b]
    """
    con = parse_sdc(sdc)
    assert con.async_pair("a", "b")
    mc = [e for e in con.exceptions if e.kind == "multicycle"][0]
    assert mc.cycles == 2


def test_parse_io_delays():
    sdc = """
    set_input_delay 2.5 [get_ports din]
    set_output_delay 3.0 [get_ports dout]
    """
    con = parse_sdc(sdc)
    assert con.input_delays.get("din") == 2.5
    assert con.output_delays.get("dout") == 3.0


def test_unknown_commands_are_ignored():
    con = parse_sdc("set_clock_groups -asynchronous\n# a comment\ncreate_clock -period 5 [get_ports c]")
    assert len(con.clocks) == 1


# ------------------------------ CDC --------------------------------- #
def _ff(clk, d, q, typ="$dff"):
    return {"type": typ, "connections": {"CLK": [clk], "D": list(d) if isinstance(d, (list, tuple)) else [d],
                                          "Q": list(q) if isinstance(q, (list, tuple)) else [q]}}


def _netlist(cells, names):
    return {"modules": {"top": {
        "cells": cells,
        "netnames": {n: {"bits": [b]} for n, b in names.items()},
        "ports": {},
    }}}


def test_single_clock_has_no_crossings():
    cells = {
        "a": _ff(1, 10, 11),
        "b": _ff(1, 11, 12),
    }
    r = analyze_cdc(_netlist(cells, {"clk": 1}), top="top")
    assert r.n_domains == 1
    assert r.crossings == []


def test_two_flop_synchronizer_is_recognized_safe():
    # src(domain clka) -> sync1(clkb) -> sync2(clkb)
    cells = {
        "src":  _ff(1, 10, 3),      # domain A (clk net 1), Q=net3
        "sync1": _ff(2, 3, 4),      # domain B (clk net 2), D<-src.Q, Q=net4
        "sync2": _ff(2, 4, 5),      # domain B, D<-sync1.Q
    }
    r = analyze_cdc(_netlist(cells, {"clka": 1, "clkb": 2, "sig": 3}), top="top")
    assert r.n_domains == 2
    assert len(r.crossings) == 1
    assert r.crossings[0].classification == "synchronized"
    assert r.worst == "synchronized"
    assert not r.unsynchronized


def test_single_flop_crossing_flagged():
    cells = {
        "src": _ff(1, 10, 3),       # domain A
        "dst": _ff(2, 3, 7),        # domain B, direct capture, no 2nd stage
    }
    r = analyze_cdc(_netlist(cells, {"clka": 1, "clkb": 2}), top="top")
    assert r.worst == "single_flop"
    assert len(r.single_flop) == 1


def test_combinational_crossing_is_unsynchronized():
    # src(A).Q -> AND gate -> dst(B).D : comb logic on the crossing path
    cells = {
        "src": _ff(1, 10, 3),
        "gate": {"type": "$and", "connections": {"A": [3], "B": [11], "Y": [6]}},
        "dst": _ff(2, 6, 7),
    }
    r = analyze_cdc(_netlist(cells, {"clka": 1, "clkb": 2}), top="top")
    assert r.worst == "unsynchronized"
    assert len(r.unsynchronized) == 1
    assert r.unsynchronized[0].from_domain == "clka"
    assert r.unsynchronized[0].to_domain == "clkb"
