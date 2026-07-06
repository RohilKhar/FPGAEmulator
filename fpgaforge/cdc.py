"""Structural clock-domain-crossing (CDC) analysis on the synthesized netlist.

CDC bugs are the canonical class that passes functional simulation *and* static
timing analysis, yet fails intermittently on real hardware (metastability). A
name-count heuristic ("there are 2 clocks, beware") is not enough. This walks
the actual Yosys netlist graph:

1. group flip-flops by their clock net (the clock domains);
2. for every FF, trace its data input backward through combinational logic to
   find which domains drive it;
3. flag a **crossing** when a signal launched in domain A is captured in domain
   B, and classify it:
   - *synchronized*: captured directly by a 2+ FF chain in the destination
     domain (the standard two-flop synchronizer);
   - *single-flop*: only one capture FF -> residual metastability risk;
   - *unsynchronized*: combinational logic sits on the crossing before capture,
     or the raw signal fans out to multiple loads -> dangerous.

This is a structural heuristic, not a full metastability proof, but it catches
the mistakes that most often burn a first spin. Pure function over the Yosys
JSON, so it is fully unit-testable without tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Yosys clock port names across generic and iCE40/ECP5 FF primitives.
_CLK_PORTS = ("CLK", "C", "clk", "clock")
_D_PORTS = ("D", "d")
_Q_PORTS = ("Q", "q")


def _is_ff(cell_type: str) -> bool:
    t = cell_type.upper()
    return "DFF" in t or "SDFF" in t or t.startswith("$_DFF") or t.startswith("$_SDFF")


@dataclass
class Crossing:
    from_domain: str
    to_domain: str
    classification: str          # "synchronized" | "single_flop" | "unsynchronized"
    signal: str = ""

    @property
    def is_safe(self) -> bool:
        return self.classification == "synchronized"


@dataclass
class CDCReport:
    domains: list[str] = field(default_factory=list)
    crossings: list[Crossing] = field(default_factory=list)

    @property
    def n_domains(self) -> int:
        return len(self.domains)

    @property
    def unsynchronized(self) -> list[Crossing]:
        return [c for c in self.crossings if c.classification == "unsynchronized"]

    @property
    def single_flop(self) -> list[Crossing]:
        return [c for c in self.crossings if c.classification == "single_flop"]

    @property
    def worst(self) -> str:
        if self.unsynchronized:
            return "unsynchronized"
        if self.single_flop:
            return "single_flop"
        if self.crossings:
            return "synchronized"
        return "none"

    def summary(self) -> str:
        lines = [f"CDC: {self.n_domains} clock domain(s), {len(self.crossings)} crossing(s)"]
        for c in self.crossings:
            lines.append(f"  {c.from_domain} -> {c.to_domain}: {c.classification}"
                         + (f" ({c.signal})" if c.signal else ""))
        return "\n".join(lines)


def _pick_module(netlist: dict, top: str | None):
    modules = netlist.get("modules", {})
    if not modules:
        return {}
    if top and top in modules:
        return modules[top]
    return max(modules.values(), key=lambda m: len(m.get("cells", {})), default={})


def _first_bit(conn: list) -> Any:
    for b in conn:
        if isinstance(b, int):
            return b
    return None


def analyze_cdc(netlist: dict, top: str | None = None, constraints=None) -> CDCReport:
    """Analyze clock-domain crossings in a Yosys ``write_json`` netlist."""
    module = _pick_module(netlist, top)
    cells = module.get("cells", {})
    report = CDCReport()
    if not cells:
        return report

    # Map net bit -> a readable name (for reporting).
    bit_name: dict[int, str] = {}
    for name, net in module.get("netnames", {}).items():
        for b in net.get("bits", []):
            if isinstance(b, int) and b not in bit_name:
                bit_name[b] = name

    # Collect FFs: clock net, D bits, Q bits.
    ffs = []                       # (clk_bit, set(d_bits), set(q_bits), name)
    q_to_domain: dict[int, int] = {}
    comb_out_to_cell: dict[int, dict] = {}   # comb cell output bit -> cell
    for cname, cell in cells.items():
        ctype = cell.get("type", "")
        conns = cell.get("connections", {})
        if _is_ff(ctype):
            clk = None
            for p in _CLK_PORTS:
                if p in conns:
                    clk = _first_bit(conns[p])
                    break
            d_bits = set()
            q_bits = set()
            for p in _D_PORTS:
                if p in conns:
                    d_bits.update(b for b in conns[p] if isinstance(b, int))
            for p in _Q_PORTS:
                if p in conns:
                    q_bits.update(b for b in conns[p] if isinstance(b, int))
            if clk is not None:
                ffs.append((clk, d_bits, q_bits, cname))
                for qb in q_bits:
                    q_to_domain[qb] = clk
        else:
            # Combinational cell: map each output-ish bit to the cell.
            for pname, bits in conns.items():
                for b in bits:
                    if isinstance(b, int):
                        comb_out_to_cell.setdefault(b, cell)

    domains = sorted({clk for clk, _, _, _ in ffs})
    report.domains = [bit_name.get(d, f"net{d}") for d in domains]
    if len(domains) < 2:
        return report        # single clock -> no crossings possible

    # Fanout count of each FF Q bit (to detect raw async fanning to many loads).
    fanout: dict[int, int] = {}
    for _cname, cell in cells.items():
        for bits in cell.get("connections", {}).values():
            for b in bits:
                if isinstance(b, int):
                    fanout[b] = fanout.get(b, 0) + 1

    def sources_in_cone(start_bits: set[int], max_depth: int = 40):
        """Backward-reachable FF-Q source bits, and whether comb logic was traversed."""
        seen: set[int] = set()
        stack = [(b, 0) for b in start_bits]
        src_domains: dict[int, int] = {}   # source q bit -> its domain
        through_comb = False
        while stack:
            bit, depth = stack.pop()
            if bit in seen or depth > max_depth:
                continue
            seen.add(bit)
            if bit in q_to_domain:
                src_domains[bit] = q_to_domain[bit]
                continue                    # stop at a register boundary
            cell = comb_out_to_cell.get(bit)
            if cell is None:
                continue                    # primary input / constant
            if depth > 0:
                through_comb = True
            if _is_ff(cell.get("type", "")):
                continue
            for bits in cell.get("connections", {}).values():
                for b in bits:
                    if isinstance(b, int) and b not in seen:
                        stack.append((b, depth + 1))
        return src_domains, through_comb

    def second_stage_exists(q_bits: set[int], domain: int) -> bool:
        """True if this FF's Q feeds directly into another FF in the same domain."""
        for clk, d_bits, _q, _n in ffs:
            if clk == domain and d_bits & q_bits:
                return True
        return False

    rank = {"synchronized": 0, "single_flop": 1, "unsynchronized": 2}
    for clk, d_bits, q_bits, name in ffs:
        src_domains, through_comb = sources_in_cone(d_bits)
        for src_bit, src_dom in src_domains.items():
            if src_dom == clk:
                continue                    # same domain, fine
            # Classify the crossing.
            if not through_comb and second_stage_exists(q_bits, clk):
                cls = "synchronized"        # captured by a 2+ FF chain
            elif not through_comb:
                cls = "single_flop"         # captured, but only one stage
            else:
                cls = "unsynchronized"      # comb logic on the crossing path
            from_name = bit_name.get(src_dom, f"net{src_dom}")
            to_name = bit_name.get(clk, f"net{clk}")
            existing = next((c for c in report.crossings
                             if (c.from_domain, c.to_domain) == (from_name, to_name)), None)
            if existing is None:
                report.crossings.append(Crossing(
                    from_domain=from_name, to_domain=to_name,
                    classification=cls, signal=bit_name.get(src_bit, ""),
                ))
            elif rank[cls] > rank[existing.classification]:
                existing.classification = cls
                existing.signal = bit_name.get(src_bit, existing.signal)

    return report
