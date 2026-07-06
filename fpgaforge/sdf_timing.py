"""A static-timing engine that reads the routed design's SDF directly.

nextpnr writes a full SDF for the placed & routed design: ``INTERCONNECT``
(routing) delays, ``IOPATH`` (cell arc) delays including the flop clk-to-Q, and
``SETUPHOLD`` (the setup requirement at every capture pin). That is a complete
timing graph, so we can rebuild the register-to-register longest path *from the
delays that will actually exist on the die* -- an independent cross-check of
nextpnr's own Fmax, and, crucially, a per-endpoint arrival model.

This powers delay-aware ("timing-accurate") emulation: at any chosen clock we
can say whether every flop's data arrives before its setup deadline, and if not,
exactly which endpoint fails and by how much -- the failure a purely functional,
zero-delay emulation cannot see.

The parser and longest-path solver are pure and unit-tested against SDF text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def _triple(tok: str) -> float:
    """Parse an SDF ``(min:typ:max)`` triple, returning the max in ps."""
    nums = re.findall(r"[-+]?\d*\.?\d+", tok)
    if not nums:
        return 0.0
    return float(nums[-1])


def _norm(pin: str) -> str:
    """Normalize an SDF pin path: drop escaping backslashes."""
    return pin.replace("\\", "")


@dataclass
class SDFCell:
    celltype: str
    instance: str
    iopaths: list[tuple[str, str, float]] = field(default_factory=list)   # (from, to, ps)
    setup: dict[str, float] = field(default_factory=dict)                 # pin -> setup ps
    setup_clk: dict[str, str] = field(default_factory=dict)               # data pin -> clk pin


@dataclass
class SDFTiming:
    timescale_ps: float = 1.0
    interconnects: list[tuple[str, str, float]] = field(default_factory=list)
    cells: list[SDFCell] = field(default_factory=list)


_IOPATH_RE = re.compile(r"\(IOPATH\s+(\S+)\s+(\S+)\s+(\([^)]*\))")
_INTER_RE = re.compile(r"\(INTERCONNECT\s+(\S+)\s+(\S+)\s+(\([^)]*\))")
# Capture both the data pin and the reference clock pin so multiple clock
# domains (each flop may be clocked by a different CLK-like pin) are handled.
_SETUP_RE = re.compile(
    r"\(SETUPHOLD\s+\((?:pos|neg)edge\s+(\S+)\)\s+\((?:pos|neg)edge\s+(\S+)\)\s+(\([^)]*\))"
)
_CELLTYPE_RE = re.compile(r'\(CELLTYPE\s+"([^"]*)"\)')
_INSTANCE_RE = re.compile(r"\(INSTANCE\s*([^)]*)\)")

DEFAULT_DOMAIN = "clk"


def parse_sdf(text: str) -> SDFTiming:
    """Parse a nextpnr SDF into interconnect + per-cell IOPATH/SETUPHOLD delays."""
    sdf = SDFTiming()
    m = re.search(r"\(TIMESCALE\s+([\d.]+)\s*ps\)", text)
    if m:
        sdf.timescale_ps = float(m.group(1))

    # Split into CELL blocks (naive but robust for nextpnr output).
    for block in re.split(r"\(CELL\b", text)[1:]:
        ct = _CELLTYPE_RE.search(block)
        inst = _INSTANCE_RE.search(block)
        celltype = ct.group(1) if ct else ""
        instance = _norm(inst.group(1).strip()) if inst else ""
        cell = SDFCell(celltype=celltype, instance=instance)
        for f, t, d in _INTER_RE.findall(block):
            sdf.interconnects.append((_norm(f), _norm(t), _triple(d)))
        for f, t, d in _IOPATH_RE.findall(block):
            cell.iopaths.append((_norm(f), _norm(t), _triple(d)))
        for pin, clkpin, d in _SETUP_RE.findall(block):
            pin = _norm(pin)
            cell.setup[pin] = max(cell.setup.get(pin, 0.0), _triple(d))
            cell.setup_clk[pin] = _norm(clkpin)
        if cell.iopaths or cell.setup:
            sdf.cells.append(cell)
    return sdf


@dataclass
class TimingArc:
    endpoint: str
    data_arrival_ps: float
    setup_ps: float
    path: list[str] = field(default_factory=list)
    domain: str = ""

    @property
    def required_period_ps(self) -> float:
        return self.data_arrival_ps + self.setup_ps


@dataclass
class DomainTiming:
    """Per-clock-domain register-to-register timing."""

    domain: str
    min_period_ps: float = 0.0
    fmax_mhz: float = 0.0
    worst: TimingArc | None = None
    n_launch: int = 0
    n_endpoints: int = 0


@dataclass
class CrossArc:
    """A register-to-register path that launches and captures in different
    clock domains -- a clock-domain crossing, whose setup is not a meaningful
    single-clock constraint (the structural CDC analysis judges its safety)."""

    endpoint: str
    launch_domain: str
    capture_domain: str
    data_arrival_ps: float
    setup_ps: float


@dataclass
class SDFTimingResult:
    min_period_ps: float = 0.0
    fmax_mhz: float = 0.0
    worst: TimingArc | None = None
    n_endpoints: int = 0
    n_launch: int = 0
    domains: dict[str, DomainTiming] = field(default_factory=dict)
    cross_domain: list[CrossArc] = field(default_factory=list)

    @property
    def multi_clock(self) -> bool:
        return len(self.domains) > 1

    def slack_ns_at(self, clock_mhz: float) -> float:
        """Setup slack (ns) if the design is clocked at ``clock_mhz``."""
        if clock_mhz <= 0:
            return 0.0
        period_ps = 1e6 / clock_mhz
        return (period_ps - self.min_period_ps) / 1000.0

    def settles_at(self, clock_mhz: float) -> bool:
        return self.slack_ns_at(clock_mhz) >= 0.0

    def summary(self) -> str:
        lines = [
            "SDF static timing (from routed delays)",
            f"  launch flops : {self.n_launch}, capture endpoints : {self.n_endpoints}",
            f"  min period   : {self.min_period_ps / 1000.0:.3f} ns "
            f"-> Fmax {self.fmax_mhz:.1f} MHz"
            + (f" (binding domain: {self.worst.domain})" if self.worst and self.worst.domain else ""),
        ]
        if self.worst:
            w = self.worst
            lines.append(
                f"  worst path   : {w.data_arrival_ps / 1000.0:.3f} ns data + "
                f"{w.setup_ps / 1000.0:.3f} ns setup -> {w.endpoint}"
            )
        if self.multi_clock:
            lines.append(f"  clock domains: {len(self.domains)}")
            for dom in sorted(self.domains.values(), key=lambda d: d.fmax_mhz):
                lines.append(
                    f"    {dom.domain:<16} Fmax {dom.fmax_mhz:7.1f} MHz "
                    f"({dom.n_launch} launch, {dom.n_endpoints} capture)"
                )
            if self.cross_domain:
                lines.append(
                    f"  cross-domain : {len(self.cross_domain)} reg-to-reg path(s) "
                    "cross clock domains (verify CDC synchronizers)"
                )
                worst_x = max(self.cross_domain, key=lambda x: x.data_arrival_ps)
                lines.append(
                    f"    e.g. {worst_x.launch_domain} -> {worst_x.capture_domain} "
                    f"@ {worst_x.endpoint}"
                )
        return "\n".join(lines)


def longest_paths(sdf: SDFTiming) -> SDFTimingResult:
    """Compute register-to-register longest paths from SDF delays, per clock domain.

    Flop Q outputs are launch sources (arrival = clk-to-Q). Flop data pins with
    a SETUPHOLD are capture endpoints. The combinational logic + routing between
    them forms a DAG (every loop passes through a flop). Each flop is assigned a
    clock *domain* by tracing the net that drives its clock pin back to its root,
    so a path only sets a domain's Fmax when it launches and captures in the same
    domain. Paths that cross domains are reported separately (they are clock-
    domain crossings, not single-clock setup constraints).
    """
    incoming: dict[str, list[tuple[str, float]]] = {}
    launch_base: dict[str, float] = {}      # FF Q pin -> clk-to-Q ps
    launch_clk: dict[str, str] = {}         # FF Q pin -> its clk pin (e.g. inst/CLK)
    # (data pin, setup_ps, clk pin) per capture endpoint.
    captures: list[tuple[str, float, str]] = []

    def add_edge(src: str, dst: str, d: float):
        incoming.setdefault(dst, []).append((src, d))

    for f, t, d in sdf.interconnects:
        add_edge(f, t, d)

    for cell in sdf.cells:
        inst = cell.instance
        clk_pins = {f for f, _t, _d in cell.iopaths if _is_clock_pin(f)}
        for f, t, d in cell.iopaths:
            if _is_clock_pin(f):
                # clk-to-Q: the driven pin is a launch source clocked by f.
                q = f"{inst}/{t}"
                if d >= launch_base.get(q, 0.0):
                    launch_base[q] = d
                    launch_clk[q] = f"{inst}/{f}"
            else:
                add_edge(f"{inst}/{f}", f"{inst}/{t}", d)
        for pin, s in cell.setup.items():
            clkpin = cell.setup_clk.get(pin) or (next(iter(clk_pins)) if clk_pins else "CLK")
            captures.append((f"{inst}/{pin}", s, f"{inst}/{clkpin}"))

    # ---- clock-domain roots: trace each clock pin back to its source net ----
    root_memo: dict[str, str] = {}

    def _trace_root(pin: str, stack: set[str]) -> str:
        inc = incoming.get(pin)
        if not inc or pin in stack:
            return pin           # terminal net = the clock's source (domain id)
        stack.add(pin)
        # Clock nets are single-driver; follow the (max-delay) source upstream.
        src = max(inc, key=lambda sd: sd[1])[0]
        return _trace_root(src, stack)

    def clock_root(clkpin: str) -> str:
        if clkpin in root_memo:
            return root_memo[clkpin]
        # A flop clock pin with no routing in the SDF -> a single default domain
        # (we cannot distinguish domains without clock-net routing information).
        root = DEFAULT_DOMAIN if not incoming.get(clkpin) else _trace_root(clkpin, set())
        root_memo[clkpin] = root
        return root

    launch_domain = {q: clock_root(clk) for q, clk in launch_clk.items()}

    # ---- domain-aware longest path: arrival per originating domain ----
    memo: dict[str, dict[str, tuple[float, str | None]]] = {}

    def amap(pin: str, stack: set[str]) -> dict[str, tuple[float, str | None]]:
        if pin in launch_base:
            return {launch_domain.get(pin, DEFAULT_DOMAIN): (launch_base[pin], None)}
        if pin in memo:
            return memo[pin]
        inc = incoming.get(pin)
        if not inc or pin in stack:
            memo[pin] = {}
            return memo[pin]
        stack.add(pin)
        res: dict[str, tuple[float, str | None]] = {}
        for src, d in inc:
            for dom, (a, _s) in amap(src, stack).items():
                cand = a + d
                if dom not in res or cand > res[dom][0]:
                    res[dom] = (cand, src)
        stack.discard(pin)
        memo[pin] = res
        return res

    def reconstruct(endpoint: str, dom: str) -> list[str]:
        path = [endpoint]
        cur = memo.get(endpoint, {}).get(dom, (0.0, None))[1]
        guard = 0
        while cur is not None and guard < 100000:
            path.append(cur)
            if cur in launch_base:
                break
            cur = memo.get(cur, {}).get(dom, (0.0, None))[1]
            guard += 1
        return list(reversed(path))

    result = SDFTimingResult(n_launch=len(launch_base), n_endpoints=len(captures))
    domains: dict[str, DomainTiming] = {}
    dom_launch: dict[str, set[str]] = {}
    for q, dom in launch_domain.items():
        dom_launch.setdefault(dom, set()).add(q)

    for pin, setup, clkpin in captures:
        cap_dom = clock_root(clkpin)
        arrivals = amap(pin, set())
        dom = domains.get(cap_dom) or DomainTiming(domain=cap_dom)
        dom.n_endpoints += 1
        domains[cap_dom] = dom
        for src_dom, (a, _src) in arrivals.items():
            if src_dom == cap_dom:
                req = a + setup
                if dom.worst is None or req > dom.worst.required_period_ps:
                    dom.worst = TimingArc(
                        endpoint=pin, data_arrival_ps=a, setup_ps=setup,
                        path=reconstruct(pin, cap_dom), domain=cap_dom,
                    )
            else:
                result.cross_domain.append(CrossArc(
                    endpoint=pin, launch_domain=src_dom, capture_domain=cap_dom,
                    data_arrival_ps=a, setup_ps=setup,
                ))

    for dom_name, dom in domains.items():
        dom.n_launch = len(dom_launch.get(dom_name, ()))
        if dom.worst:
            dom.min_period_ps = dom.worst.required_period_ps
            if dom.min_period_ps > 0:
                dom.fmax_mhz = 1e6 / dom.min_period_ps
    result.domains = domains

    # Overall binding constraint = the domain with the largest required period
    # (lowest Fmax). Preserves single-clock behavior exactly.
    binding = [d for d in domains.values() if d.worst is not None]
    if binding:
        worst_dom = max(binding, key=lambda d: d.min_period_ps)
        result.worst = worst_dom.worst
        result.min_period_ps = worst_dom.min_period_ps
        result.fmax_mhz = worst_dom.fmax_mhz
    return result


def _is_clock_pin(pin: str) -> bool:
    """Whether an IOPATH source pin is a clock (launch) pin.

    Pin names are local to the cell ("CLK" for iCE40/ECP5 flops); the *domain*
    is distinguished later by tracing the net that drives the pin, not the name.
    """
    p = pin.upper()
    return p == "CLK" or p.endswith("CLK")


def analyze_sdf(sdf_path: str | Path) -> SDFTimingResult:
    return longest_paths(parse_sdf(Path(sdf_path).read_text()))
