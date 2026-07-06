"""Measure real switching activity from a simulation VCD.

The power/thermal model needs an activity factor -- the fraction of nodes that
toggle per clock cycle. Assuming a global constant (e.g. 0.15) is a guess; but
the virtual bring-up / verify / board runs already dump a full VCD of the mapped
fabric, so we can *measure* it: count per-bit transitions of every net, divide by
the number of clock cycles, and average. That turns power from a datasheet
hand-wave into a number derived from how the design actually behaves under its
own stimulus.

The parser is intentionally small and dependency-free; it understands the subset
of VCD that Icarus Verilog emits (scalar + vector value changes, one clock).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_CLOCK_NAMES = {"clk", "clock", "clk_i", "sysclk", "sys_clk", "i_clk", "clock_i"}


@dataclass
class ActivityReport:
    activity: float = 0.0          # mean per-bit toggle rate per cycle (0..1+)
    cycles: int = 0
    n_signals: int = 0             # data signals measured (clock excluded)
    n_bits: int = 0                # total data bits measured
    total_transitions: int = 0
    clock: str | None = None
    busiest: list[tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "switching activity (measured)",
            f"  clock      : {self.clock or '?'} over {self.cycles} cycles",
            f"  signals    : {self.n_signals} nets, {self.n_bits} bits",
            f"  activity   : {self.activity:.1%} mean per-bit toggle/cycle "
            f"({self.total_transitions} transitions)",
        ]
        if self.busiest:
            top = ", ".join(f"{n}={r:.0%}" for n, r in self.busiest[:5])
            lines.append(f"  busiest    : {top}")
        return "\n".join(lines)


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1].split("[", 1)[0]


def parse_vcd_activity(text: str, clock_hint: str | None = None) -> ActivityReport:
    """Compute switching activity from VCD text.

    Args:
        text: full VCD contents.
        clock_hint: name (leaf or full) of the clock net; auto-detected if None.
    """
    id_name: dict[str, str] = {}
    id_width: dict[str, int] = {}
    scope: list[str] = []

    lines = text.splitlines()
    i = 0
    n = len(lines)
    # ---- header: variable definitions ----
    while i < n:
        line = lines[i].strip()
        i += 1
        if line.startswith("$scope"):
            toks = line.split()
            if len(toks) >= 3:
                scope.append(toks[2])
        elif line.startswith("$upscope"):
            if scope:
                scope.pop()
        elif line.startswith("$var"):
            toks = line.split()
            # $var <type> <width> <id> <name> [range] $end
            if len(toks) >= 5:
                width = int(toks[2])
                vid = toks[3]
                name = toks[4]
                full = ".".join(scope + [name]) if scope else name
                # A net can be aliased under many scopes (one clock feeds every
                # flop). They share one VCD id; keep the cleanest name (fewest
                # path components, then shortest) for display.
                prev = id_name.get(vid)
                if prev is None or (full.count("."), len(full)) < (prev.count("."), len(prev)):
                    id_name[vid] = full
                id_width[vid] = max(id_width.get(vid, 0), width)
        elif line.startswith("$enddefinitions"):
            break

    # ---- pick the clock ----
    clock_id = None
    if clock_hint:
        for vid, nm in id_name.items():
            if nm == clock_hint or _leaf(nm) == _leaf(clock_hint):
                clock_id = vid
                break
    if clock_id is None:
        for vid, nm in id_name.items():
            if id_width.get(vid, 1) == 1 and _leaf(nm).lower() in _CLOCK_NAMES:
                clock_id = vid
                break

    current: dict[str, str] = {}
    transitions: dict[str, int] = {}
    rising: dict[str, int] = {}

    def apply_scalar(vid: str, val: str) -> None:
        old = current.get(vid)
        current[vid] = val
        if old is None:
            return
        if old != val and old in "01" and val in "01":
            transitions[vid] = transitions.get(vid, 0) + 1
            if old == "0" and val == "1":
                rising[vid] = rising.get(vid, 0) + 1

    def apply_vector(vid: str, bits: str) -> None:
        old = current.get(vid)
        current[vid] = bits
        if old is None:
            return
        w = max(len(old), len(bits))
        oa = old.rjust(w, old[0] if old and old[0] in "xz" else "0")
        na = bits.rjust(w, "0")
        diff = 0
        for a, b in zip(oa, na):
            if a != b and a in "01" and b in "01":
                diff += 1
        if diff:
            transitions[vid] = transitions.get(vid, 0) + diff

    # ---- value-change section ----
    while i < n:
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        if line.startswith("$"):
            continue
        c = line[0]
        if c in "01xz":
            apply_scalar(line[1:], c)
        elif c in "bB":
            parts = line.split()
            if len(parts) >= 2:
                apply_vector(parts[1], parts[0][1:])
        # ignore real ('r') and unknown records

    # ---- pick clock by activity if not found by name ----
    if clock_id is None:
        scalar_rising = {v: r for v, r in rising.items() if id_width.get(v, 1) == 1}
        if scalar_rising:
            clock_id = max(scalar_rising, key=scalar_rising.get)

    cycles = rising.get(clock_id, 0) if clock_id else 0
    if cycles == 0 and rising:
        cycles = max(rising.values())

    report = ActivityReport(clock=id_name.get(clock_id) if clock_id else None)
    report.cycles = cycles
    if cycles <= 0:
        return report

    total_bits = 0
    total_tr = 0
    rates: list[tuple[str, float]] = []
    for vid, width in id_width.items():
        if vid == clock_id:
            continue
        total_bits += width
        tr = transitions.get(vid, 0)
        total_tr += tr
        rate = tr / (width * cycles) if width and cycles else 0.0
        rates.append((id_name.get(vid, vid), rate))

    report.n_signals = sum(1 for vid in id_width if vid != clock_id)
    report.n_bits = total_bits
    report.total_transitions = total_tr
    report.activity = (total_tr / (total_bits * cycles)) if total_bits and cycles else 0.0
    report.busiest = sorted(rates, key=lambda kv: -kv[1])[:8]
    return report


def measure_activity(vcd_path: str | Path, clock_hint: str | None = None) -> ActivityReport:
    """Measure switching activity from a VCD file on disk."""
    return parse_vcd_activity(Path(vcd_path).read_text(), clock_hint)
