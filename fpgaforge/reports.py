"""Parse tool logs/reports into a normalized `RunMetrics`.

Kept pure and text-based so it can be unit-tested against captured sample
output without any tools installed.
"""

from __future__ import annotations

import re
from typing import Any

from .backends.base import RunMetrics

_FMAX_RE = re.compile(
    r"Max frequency for clock\s+'(?P<clk>[^']+)':\s+"
    r"(?P<fmax>[\d.]+)\s+MHz\s+"
    r"\((?P<status>PASS|FAIL)\s+at\s+(?P<target>[\d.]+)\s+MHz\)"
)

# nextpnr device utilisation lines, e.g. "ICESTORM_LC:   123/ 5280    2%"
_UTIL_RE = re.compile(
    r"(?P<res>[A-Z0-9_]+):\s+(?P<used>\d+)\s*/\s*(?P<avail>\d+)"
)

# yosys text-mode "stat" lines, e.g. "     SB_LUT4                        123"
_STAT_CELL_RE = re.compile(r"^\s+(?P<cell>[\\$A-Za-z][\w$]*)\s+(?P<count>\d+)\s*$")


def parse_nextpnr_log(text: str) -> dict[str, Any]:
    """Extract timing, routing status, and utilisation from a nextpnr log."""
    out: dict[str, Any] = {
        "fmax_mhz": 0.0,
        "target_freq_mhz": 0.0,
        "routed_ok": False,
        "util": {},
    }

    fmaxes: list[float] = []
    targets: list[float] = []
    for m in _FMAX_RE.finditer(text):
        fmaxes.append(float(m.group("fmax")))
        targets.append(float(m.group("target")))
    if fmaxes:
        # Worst-case clock governs the design.
        out["fmax_mhz"] = min(fmaxes)
        out["target_freq_mhz"] = max(targets)

    for m in _UTIL_RE.finditer(text):
        out["util"][m.group("res")] = int(m.group("used"))

    routing_failed = (
        "Routing failed" in text
        or "Failed to route" in text
        or "Failed to place" in text
        or "Unable to find a placement location" in text
        or "Unable to place" in text
    )
    routing_done = (
        "Program finished normally" in text
        or "Routing complete" in text
        or "Checksum" in text
        or bool(fmaxes)
    )
    out["routed_ok"] = routing_done and not routing_failed
    return out


def parse_yosys_stat_json(stat: dict[str, Any]) -> dict[str, int]:
    """Extract cell counts from `yosys stat -json` output.

    The JSON has the shape {"design": {"num_cells_by_type": {...}, ...}} or,
    for multi-module designs, per-module blocks. We aggregate defensively.
    """
    counts: dict[str, int] = {}

    def absorb(block: dict[str, Any]) -> None:
        by_type = block.get("num_cells_by_type") or {}
        for cell, n in by_type.items():
            counts[cell] = counts.get(cell, 0) + int(n)

    if "design" in stat and isinstance(stat["design"], dict):
        absorb(stat["design"])
    if "modules" in stat and isinstance(stat["modules"], dict):
        for block in stat["modules"].values():
            if isinstance(block, dict):
                absorb(block)
    if not counts:
        absorb(stat)
    return counts


def parse_yosys_stat_text(text: str) -> dict[str, int]:
    """Extract cell counts from human-readable `yosys stat` output."""
    counts: dict[str, int] = {}
    in_cells = False
    for line in text.splitlines():
        if "Number of cells:" in line:
            in_cells = True
            continue
        if in_cells:
            m = _STAT_CELL_RE.match(line)
            if m:
                counts[m.group("cell")] = int(m.group("count"))
            elif line.strip() == "":
                continue
            elif not line.startswith(" "):
                in_cells = False
    return counts


def _sum_matching(counts: dict[str, int], predicate) -> int:
    return sum(n for cell, n in counts.items() if predicate(cell.upper()))


def build_metrics(
    nextpnr_log: str = "",
    yosys_stat: dict[str, int] | None = None,
    target_freq_mhz: float = 0.0,
) -> RunMetrics:
    """Combine parsed nextpnr + yosys data into a RunMetrics."""
    npr = parse_nextpnr_log(nextpnr_log) if nextpnr_log else {
        "fmax_mhz": 0.0,
        "target_freq_mhz": 0.0,
        "routed_ok": False,
        "util": {},
    }
    counts = yosys_stat or {}
    util = npr["util"]

    metrics = RunMetrics()
    metrics.fmax_mhz = float(npr["fmax_mhz"])
    metrics.target_freq_mhz = float(npr["target_freq_mhz"] or target_freq_mhz)
    metrics.routed_ok = bool(npr["routed_ok"])
    if metrics.fmax_mhz > 0:
        metrics.crit_path_ns = 1000.0 / metrics.fmax_mhz

    # Utilisation: prefer nextpnr device numbers (iCE40 ICESTORM_* or ECP5
    # TRELLIS_*/DP16KD/MULT names), fall back to yosys cell counts.
    def _util_any(*keys: str) -> int:
        return sum(int(util.get(k, 0)) for k in keys)

    metrics.luts = _util_any("ICESTORM_LC", "TRELLIS_COMB", "TRELLIS_SLICE") \
        or _sum_matching(counts, lambda c: "LUT" in c)
    metrics.bram = _util_any("ICESTORM_RAM", "DP16KD", "PDPW16KD") \
        or _sum_matching(counts, lambda c: "RAM" in c or "BRAM" in c or "DP16KD" in c)
    metrics.dsp = _util_any("ICESTORM_DSP", "MULT18X18D", "MULT9X9D") \
        or _sum_matching(counts, lambda c: "MAC16" in c or "DSP" in c or "MULT" in c)
    metrics.ffs = _util_any("TRELLIS_FF") \
        or _sum_matching(counts, lambda c: "DFF" in c or c == "TRELLIS_FF")
    metrics.carries = _util_any("CCU2C", "CCU2") \
        or _sum_matching(counts, lambda c: "CARRY" in c or "CCU2" in c)
    return metrics
