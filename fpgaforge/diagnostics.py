"""Extract human-readable errors and warnings from tool logs.

The backends capture raw stdout/stderr from yosys, nextpnr, iverilog, and vvp.
This module distills that into structured `Diagnostic`s (severity + message +
tool + optional file:line) so failures surface the *actual* tool message
instead of a generic "synthesis failed".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# iverilog / gcc-style: "file.v:12: error: message"
_FILELINE_RE = re.compile(
    r"^\s*(?P<file>[^\s:][^:]*):(?P<line>\d+):\s*(?P<sev>error|warning|syntax error)\b[: ]*(?P<msg>.*)$",
    re.IGNORECASE,
)
# yosys / nextpnr-style: "ERROR: message" / "Warning: message"
_PREFIX_RE = re.compile(
    r"^\s*(?P<sev>ERROR|Warning|WARNING)\s*:\s*(?P<msg>.*)$"
)
# Command echo lines the backends emit: "$ yosys -q -s ..."
_CMD_RE = re.compile(r"^\$\s+(?P<cmd>\S+)")

# Timing "ERROR" from nextpnr is really an informational timing miss, not a
# tool failure; classify it as a warning so it does not read as a crash.
_TIMING_HINT = "max frequency for clock"


@dataclass(frozen=True)
class Diagnostic:
    severity: str          # "error" | "warning"
    message: str
    tool: str | None = None
    location: str | None = None  # "file:line" when available

    def format(self) -> str:
        parts = [f"[{self.severity}]"]
        if self.tool:
            parts.append(f"{self.tool}:")
        if self.location:
            parts.append(f"{self.location}:")
        parts.append(self.message)
        return " ".join(parts)


def _tool_from_cmd(cmd: str) -> str:
    base = cmd.rsplit("/", 1)[-1]
    return base


def extract(log: str, limit: int = 25) -> list[Diagnostic]:
    """Parse a captured tool log into de-duplicated diagnostics (in order)."""
    diags: list[Diagnostic] = []
    seen: set[tuple[str, str, str | None]] = set()
    current_tool: str | None = None

    for raw in log.splitlines():
        cmd_match = _CMD_RE.match(raw)
        if cmd_match:
            current_tool = _tool_from_cmd(cmd_match.group("cmd"))
            continue

        sev: str | None = None
        msg = ""
        location: str | None = None

        m = _FILELINE_RE.match(raw)
        if m:
            raw_sev = m.group("sev").lower()
            sev = "warning" if raw_sev == "warning" else "error"
            msg = (m.group("msg") or raw_sev).strip()
            location = f"{m.group('file')}:{m.group('line')}"
        else:
            m = _PREFIX_RE.match(raw)
            if m:
                msg = m.group("msg").strip()
                raw_sev = m.group("sev").lower()
                sev = "warning" if raw_sev.startswith("warn") else "error"
                # Reclassify nextpnr timing-miss "ERROR" as a warning.
                if _TIMING_HINT in msg.lower():
                    sev = "warning"

        if sev is None or not msg:
            continue

        key = (sev, msg, location)
        if key in seen:
            continue
        seen.add(key)
        diags.append(Diagnostic(severity=sev, message=msg, tool=current_tool, location=location))
        if len(diags) >= limit:
            break

    return diags


def errors(log: str, limit: int = 25) -> list[Diagnostic]:
    return [d for d in extract(log, limit) if d.severity == "error"]


def warnings(log: str, limit: int = 25) -> list[Diagnostic]:
    return [d for d in extract(log, limit) if d.severity == "warning"]
