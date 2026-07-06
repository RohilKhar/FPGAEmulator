"""Timing-constraint (SDC-style) ingestion.

Real designs carry timing *intent* that a single implicit target clock cannot
express: multiple clocks, asynchronous relationships, false paths, and
multicycle paths. Without it, a readiness verdict can be wrong in both
directions -- falsely flagging a legitimate false path, or (worse) silently
trusting an unhandled clock-domain crossing.

This parses the common subset of SDC used in practice:

* ``create_clock -name clk -period 10.0 [get_ports clk]``
* ``set_false_path -from [get_clocks a] -to [get_clocks b]``
* ``set_multicycle_path N -from ... -to ...``
* ``set_input_delay`` / ``set_output_delay``

It is deliberately tolerant: unknown commands are ignored, so a real vendor SDC
can be fed in and we extract what we understand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClockConstraint:
    name: str
    period_ns: float
    port: str = ""

    @property
    def freq_mhz(self) -> float:
        return 1000.0 / self.period_ns if self.period_ns > 0 else 0.0


@dataclass
class PathException:
    kind: str            # "false" | "multicycle"
    from_clock: str = ""
    to_clock: str = ""
    from_port: str = ""
    to_port: str = ""
    cycles: int = 1      # for multicycle


@dataclass
class Constraints:
    clocks: dict[str, ClockConstraint] = field(default_factory=dict)
    exceptions: list[PathException] = field(default_factory=list)
    input_delays: dict[str, float] = field(default_factory=dict)
    output_delays: dict[str, float] = field(default_factory=dict)

    @property
    def is_multiclock(self) -> bool:
        return len(self.clocks) > 1

    def fastest_clock(self) -> ClockConstraint | None:
        return max(self.clocks.values(), default=None,
                   key=lambda c: c.freq_mhz) if self.clocks else None

    def async_pair(self, a: str, b: str) -> bool:
        """True if a false-path exception marks these two clocks as async."""
        for e in self.exceptions:
            if e.kind != "false":
                continue
            pair = {e.from_clock, e.to_clock}
            if a in pair and b in pair:
                return True
        return False

    def summary(self) -> str:
        lines = ["timing constraints:"]
        for c in self.clocks.values():
            lines.append(f"  clock {c.name}: {c.period_ns:g} ns ({c.freq_mhz:.1f} MHz)"
                         + (f" on {c.port}" if c.port else ""))
        for e in self.exceptions:
            if e.kind == "false":
                lines.append(f"  false_path: {e.from_clock or e.from_port} -> "
                             f"{e.to_clock or e.to_port}")
            else:
                lines.append(f"  multicycle x{e.cycles}: {e.from_clock or e.from_port} -> "
                             f"{e.to_clock or e.to_port}")
        return "\n".join(lines)


_NUM = r"[-+]?\d*\.?\d+"


def _extract(pattern: str, text: str, group: int = 1) -> str:
    m = re.search(pattern, text)
    return m.group(group) if m else ""


def _get_tokens(text: str, keyword: str) -> str:
    """Return the argument following ``-keyword`` (handles ``[get_clocks x]``)."""
    m = re.search(rf"-{keyword}\s+(?:\[\s*get_(?:clocks|ports|pins)\s+([^\]]+)\]|(\S+))", text)
    if not m:
        return ""
    return (m.group(1) or m.group(2) or "").strip().strip("{}").strip()


def parse_sdc(text: str) -> Constraints:
    """Parse the understood subset of SDC from ``text``."""
    con = Constraints()
    # Join line continuations and drop comments.
    text = re.sub(r"#.*", "", text)
    text = text.replace("\\\n", " ")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cmd = line.split()[0]

        if cmd == "create_clock":
            period = _extract(rf"-period\s+({_NUM})", line)
            name = _get_tokens(line, "name")
            port = ""
            m = re.search(r"\[\s*get_ports\s+([^\]]+)\]", line)
            if m:
                port = m.group(1).strip().strip("{}").strip()
            if not name:
                name = port or f"clk{len(con.clocks)}"
            if period:
                con.clocks[name] = ClockConstraint(name=name, period_ns=float(period),
                                                   port=port)

        elif cmd == "set_false_path":
            con.exceptions.append(PathException(
                kind="false",
                from_clock=_get_tokens(line, "from"),
                to_clock=_get_tokens(line, "to"),
            ))

        elif cmd == "set_multicycle_path":
            m = re.search(rf"set_multicycle_path\s+(\d+)", line)
            cycles = int(m.group(1)) if m else 1
            con.exceptions.append(PathException(
                kind="multicycle", cycles=cycles,
                from_clock=_get_tokens(line, "from"),
                to_clock=_get_tokens(line, "to"),
            ))

        elif cmd == "set_input_delay":
            val = _extract(rf"set_input_delay\s+({_NUM})", line)
            port = _get_tokens(line, "port") or _extract(r"\[\s*get_ports\s+([^\]]+)\]", line)
            if val and port:
                con.input_delays[port.strip().strip("{}")] = float(val)

        elif cmd == "set_output_delay":
            val = _extract(rf"set_output_delay\s+({_NUM})", line)
            port = _get_tokens(line, "port") or _extract(r"\[\s*get_ports\s+([^\]]+)\]", line)
            if val and port:
                con.output_delays[port.strip().strip("{}")] = float(val)

    return con


def load_sdc(path: str | Path) -> Constraints:
    return parse_sdc(Path(path).read_text())
