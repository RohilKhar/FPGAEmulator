"""Unit tests for VCD switching-activity measurement (no tools needed)."""

from fpgaforge.vcd import measure_activity, parse_vcd_activity


def _make_vcd(rows, clk_id="!", q_id='"', q_width=1):
    """Build a tiny VCD. `rows` is a list of (clk, q) at successive timestamps."""
    header = [
        "$timescale 1ns $end",
        "$scope module tb $end",
        f"$var wire 1 {clk_id} clk $end",
        f"$var wire {q_width} {q_id} q $end",
        "$upscope $end",
        "$enddefinitions $end",
    ]
    body = []
    t = 0
    for clk, q in rows:
        body.append(f"#{t}")
        body.append(f"{clk}{clk_id}")
        if q_width == 1:
            body.append(f"{q}{q_id}")
        else:
            body.append(f"b{q} {q_id}")
        t += 5
    return "\n".join(header + body) + "\n"


def test_static_signal_zero_activity():
    # clk toggles; q never changes -> activity 0.
    rows = [(0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0)]
    rep = parse_vcd_activity(_make_vcd(rows))
    assert rep.clock is not None
    assert rep.cycles == 3  # three rising edges
    assert rep.activity == 0.0


def test_toggling_signal_high_activity():
    # q flips every cycle -> activity approaches 1.0.
    rows = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 0), (1, 1), (0, 1), (1, 0)]
    rep = parse_vcd_activity(_make_vcd(rows))
    assert rep.cycles == 4
    assert rep.activity > 0.4  # q changes on most cycles


def test_clock_excluded_from_data_bits():
    rows = [(0, 0), (1, 0), (0, 0), (1, 0)]
    rep = parse_vcd_activity(_make_vcd(rows))
    # Only q (1 bit) counted as data; clk excluded.
    assert rep.n_bits == 1
    assert rep.n_signals == 1


def test_vector_bit_transitions():
    # 4-bit bus counting 0,1,2,3 -> bit0 flips a lot, bit1 less.
    rows = [
        (0, "0000"), (1, "0001"), (0, "0001"), (1, "0010"),
        (0, "0010"), (1, "0011"), (0, "0011"), (1, "0100"),
    ]
    rep = parse_vcd_activity(_make_vcd(rows, q_width=4))
    assert rep.cycles == 4
    assert rep.n_bits == 4
    assert 0.0 < rep.activity <= 1.0
    assert rep.total_transitions > 0


def test_clock_hint_selects_named_clock():
    rows = [(0, 1), (1, 0), (0, 1), (1, 0)]
    rep = parse_vcd_activity(_make_vcd(rows), clock_hint="clk")
    assert rep.clock and rep.clock.endswith("clk")


def test_measure_activity_from_file(tmp_path):
    rows = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 0), (1, 1)]
    p = tmp_path / "w.vcd"
    p.write_text(_make_vcd(rows))
    rep = measure_activity(p)
    assert rep.cycles == 3
    assert rep.activity > 0.0
