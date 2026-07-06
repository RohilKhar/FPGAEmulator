"""Feature extraction for the predictive model.

Two sources are supported so the same feature schema works everywhere:

* `from_yosys_json` — precise features from a synthesized netlist (used by the
  real iCE40 backend).
* `from_rtl_text` — cheap heuristic features straight from Verilog source (used
  by the offline MockBackend and as a fallback before synthesis).

The ordered `FEATURE_NAMES` guarantees consistent model vectors across runs.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

FEATURE_NAMES: tuple[str, ...] = (
    "num_cells",
    "num_luts",
    "num_ffs",
    "num_carries",
    "num_dsp",
    "num_mem_bits",
    "num_nets",
    "max_fanout",
    "avg_fanout",
    "num_inputs",
    "num_outputs",
)


def empty_features() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES}


def to_vector(features: dict[str, float]) -> list[float]:
    """Project a feature dict onto the fixed ordered schema."""
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def _is_ff(cell_type: str) -> bool:
    t = cell_type.upper()
    return "DFF" in t or t.startswith("$DFF") or t.startswith("$_DFF")


def _is_lut(cell_type: str) -> bool:
    t = cell_type.upper()
    return "LUT" in t


def _is_carry(cell_type: str) -> bool:
    return "CARRY" in cell_type.upper()


def _is_dsp(cell_type: str) -> bool:
    t = cell_type.upper()
    return "MAC16" in t or "DSP" in t or "MUL" in t


def from_yosys_json(netlist: dict[str, Any], top: str | None = None) -> dict[str, float]:
    """Extract features from a Yosys `write_json` netlist dict."""
    feats = empty_features()
    modules = netlist.get("modules", {})
    if not modules:
        return feats

    if top and top in modules:
        module = modules[top]
    else:
        # Pick the module with the most cells as the likely top.
        module = max(
            modules.values(),
            key=lambda m: len(m.get("cells", {})),
            default={},
        )

    cells = module.get("cells", {})
    ports = module.get("ports", {})
    memories = module.get("memories", {})

    type_counts: Counter[str] = Counter()
    bit_sink_counts: Counter[Any] = Counter()

    for cell in cells.values():
        ctype = cell.get("type", "")
        type_counts[ctype] += 1
        for bits in cell.get("connections", {}).values():
            for bit in bits:
                if isinstance(bit, int):  # ignore constant strings "0"/"1"/"x"
                    bit_sink_counts[bit] += 1

    feats["num_cells"] = float(len(cells))
    feats["num_luts"] = float(sum(c for t, c in type_counts.items() if _is_lut(t)))
    feats["num_ffs"] = float(sum(c for t, c in type_counts.items() if _is_ff(t)))
    feats["num_carries"] = float(
        sum(c for t, c in type_counts.items() if _is_carry(t))
    )
    feats["num_dsp"] = float(sum(c for t, c in type_counts.items() if _is_dsp(t)))

    mem_bits = 0
    for mem in memories.values():
        width = int(mem.get("width", 0))
        size = int(mem.get("size", 0))
        mem_bits += width * size
    feats["num_mem_bits"] = float(mem_bits)

    feats["num_nets"] = float(len(bit_sink_counts))
    if bit_sink_counts:
        feats["max_fanout"] = float(max(bit_sink_counts.values()))
        feats["avg_fanout"] = float(
            sum(bit_sink_counts.values()) / len(bit_sink_counts)
        )

    n_in = sum(1 for p in ports.values() if p.get("direction") == "input")
    n_out = sum(1 for p in ports.values() if p.get("direction") == "output")
    feats["num_inputs"] = float(n_in)
    feats["num_outputs"] = float(n_out)
    return feats


def from_rtl_text(rtl: str) -> dict[str, float]:
    """Cheap heuristic features derived from Verilog source text.

    This is intentionally approximate: it exists so the platform can produce a
    feature vector without invoking synthesis (offline mock flow, cold start).
    """
    feats = empty_features()
    # Strip comments to reduce noise.
    text = re.sub(r"//.*", "", rtl)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)

    reg_bits = _count_declared_bits(text, r"\breg\b")
    wire_bits = _count_declared_bits(text, r"\bwire\b")
    input_bits = _count_declared_bits(text, r"\binput\b")
    output_bits = _count_declared_bits(text, r"\boutput\b")

    adders = len(re.findall(r"[^=!<>+]\+[^=+]", text))
    mults = len(re.findall(r"\*", text))
    always_blocks = len(re.findall(r"\balways\b", text))
    mem_arrays = re.findall(r"\breg\s*\[(\d+)\s*:\s*(\d+)\]\s*\w+\s*\[(\d+)\s*:\s*(\d+)\]", text)

    mem_bits = 0
    for msb, lsb, hi, lo in mem_arrays:
        width = abs(int(msb) - int(lsb)) + 1
        depth = abs(int(hi) - int(lo)) + 1
        mem_bits += width * depth

    # Rough cell/LUT/FF proxies.
    feats["num_ffs"] = float(reg_bits)
    feats["num_cells"] = float(reg_bits + wire_bits + adders * 4 + mults * 16)
    feats["num_luts"] = float(wire_bits + adders * 2 + mults * 8)
    feats["num_carries"] = float(adders * 2 + mults * 4)
    feats["num_dsp"] = float(mults)
    feats["num_mem_bits"] = float(mem_bits)
    feats["num_nets"] = float(reg_bits + wire_bits + input_bits + output_bits)
    # Combinational fan-in per always block is a crude fanout proxy.
    feats["max_fanout"] = float(max(4, adders + mults + always_blocks))
    feats["avg_fanout"] = float(2.0 + 0.5 * always_blocks)
    feats["num_inputs"] = float(input_bits)
    feats["num_outputs"] = float(output_bits)
    return feats


def _count_declared_bits(text: str, keyword_pattern: str) -> int:
    """Sum bit-widths of declarations matching `keyword_pattern`.

    Handles both ANSI port headers (``input [15:0] a,``) and body declarations
    (``reg [31:0] prod;``), including optional ``reg``/``wire``/``logic``/
    ``signed`` qualifiers. Counts one identifier per keyword occurrence, which
    slightly undercounts comma-lists but is fine for heuristic features.
    """
    total = 0
    pattern = re.compile(
        keyword_pattern
        + r"\s*(?:(?:reg|wire|logic|signed)\s+)*"
        + r"(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*"
        + r"([A-Za-z_]\w*)"
    )
    for match in pattern.finditer(text):
        msb, lsb = match.group(1), match.group(2)
        width = (abs(int(msb) - int(lsb)) + 1) if msb is not None else 1
        total += width
    return total


def from_rtl_files(rtl_files: Iterable[str]) -> dict[str, float]:
    """Aggregate heuristic features across multiple RTL files."""
    combined = ""
    for p in rtl_files:
        path = Path(p)
        if path.exists():
            combined += "\n" + path.read_text()
        else:
            combined += "\n" + str(p)
    return from_rtl_text(combined)
