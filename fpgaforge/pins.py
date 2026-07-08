"""Pin-constraint parsing and validation: the board-level first-shot killer.

A wrong or missing pin assignment is the classic way a design that simulates
perfectly fails on the bench: the tools happily auto-place I/O on whatever pins
route best, which are not the pins your PCB is wired to. This module makes pin
constraints a first-class, *checked* input to the readiness gate:

* :func:`load_pins` parses the constraint dialects of every supported flow --
  ``.pcf`` (iCE40/nextpnr), ``.lpf`` (ECP5), ``.xdc`` (AMD/Vivado),
  ``.qsf`` (Intel/Quartus) -- into one normalized :class:`PinConstraints`.
* :func:`check_pins` validates them against the design's real top-level ports
  (every port bit pinned, no double-booked pins, no constraints for ports that
  don't exist), the device package (pin actually exists, fits the package I/O
  budget), and optionally a board spec (clock port wired to a board clock
  source of a sufficient frequency; I/O-standard voltage backed by a rail).

The same PCF is then fed to the real place-and-route run, so the bitstream that
gets proved/verified is constrained to the *board's* pins -- not auto-placed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PinAssignment:
    port: str                    # port name, possibly bit-indexed: "count[3]"
    pin: str                     # package pin/site, e.g. "35", "E3", "PIN_R8"
    io_standard: str | None = None
    line: int = 0                # 1-based line in the constraints file


@dataclass
class PinConstraints:
    path: str
    fmt: str                                  # "pcf" | "lpf" | "xdc" | "qsf"
    assignments: list[PinAssignment] = field(default_factory=list)

    def by_port(self) -> dict[str, PinAssignment]:
        return {a.port: a for a in self.assignments}


@dataclass
class PinReport:
    """Outcome of validating pin constraints against design/package/board."""

    constrained_ports: int = 0
    total_port_bits: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        head = (f"pin constraints: {self.constrained_ports}/{self.total_port_bits} "
                f"port bits pinned")
        lines = [head]
        lines += [f"  [error] {e}" for e in self.errors]
        lines += [f"  [warn]  {w}" for w in self.warnings]
        return "\n".join(lines)


# ------------------------------- parsers --------------------------------- #
_PCF_RE = re.compile(r"^\s*set_io\s+(?:--?\S+\s+)*(\S+)\s+(\S+)")
_LPF_LOCATE_RE = re.compile(
    r'LOCATE\s+COMP\s+"([^"]+)"\s+SITE\s+"([^"]+)"', re.IGNORECASE)
_LPF_IOBUF_RE = re.compile(
    r'IOBUF\s+PORT\s+"([^"]+)"\s+(.*?);', re.IGNORECASE)
_XDC_PKG_RE = re.compile(
    r"set_property\s+(?:-quiet\s+)?PACKAGE_PIN\s+(\S+)\s+\[\s*get_ports\s+"
    r"(?:\{([^}]+)\}|(\S+?))\s*\]")
_XDC_IOSTD_RE = re.compile(
    r"set_property\s+(?:-quiet\s+)?IOSTANDARD\s+(\S+)\s+\[\s*get_ports\s+"
    r"(?:\{([^}]+)\}|(\S+?))\s*\]")
_XDC_DICT_RE = re.compile(
    r"set_property\s+-dict\s+\{([^}]*)\}\s+\[\s*get_ports\s+"
    r"(?:\{([^}]+)\}|(\S+?))\s*\]")
_QSF_LOC_RE = re.compile(r"set_location_assignment\s+(\S+)\s+-to\s+(\S+)")
_QSF_IOSTD_RE = re.compile(
    r'set_instance_assignment\s+-name\s+IO_STANDARD\s+"([^"]+)"\s+-to\s+(\S+)')


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def _parse_pcf(text: str) -> list[PinAssignment]:
    out = []
    for i, raw in enumerate(text.splitlines(), 1):
        m = _PCF_RE.match(_strip_comment(raw))
        if m:
            out.append(PinAssignment(port=m.group(1), pin=m.group(2), line=i))
    return out


def _parse_lpf(text: str) -> list[PinAssignment]:
    out, iostd = [], {}
    for m in _LPF_IOBUF_RE.finditer(text):
        km = re.search(r"IO_TYPE\s*=\s*(\S+)", m.group(2), re.IGNORECASE)
        if km:
            iostd[m.group(1)] = km.group(1)
    for i, raw in enumerate(text.splitlines(), 1):
        m = _LPF_LOCATE_RE.search(raw)
        if m:
            out.append(PinAssignment(port=m.group(1), pin=m.group(2),
                                     io_standard=iostd.get(m.group(1)), line=i))
    return out


def _parse_xdc(text: str) -> list[PinAssignment]:
    pins: dict[str, PinAssignment] = {}
    order: list[str] = []

    def _record(port: str, pin: str | None, iostd: str | None, line: int):
        a = pins.get(port)
        if a is None:
            a = PinAssignment(port=port, pin="", line=line)
            pins[port] = a
            order.append(port)
        if pin:
            a.pin = pin
        if iostd:
            a.io_standard = iostd

    for i, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw)
        m = _XDC_DICT_RE.search(line)
        if m:
            props = dict(zip(m.group(1).split()[0::2], m.group(1).split()[1::2]))
            port = (m.group(2) or m.group(3)).strip()
            _record(port, props.get("PACKAGE_PIN"), props.get("IOSTANDARD"), i)
            continue
        m = _XDC_PKG_RE.search(line)
        if m:
            _record((m.group(2) or m.group(3)).strip(), m.group(1), None, i)
            continue
        m = _XDC_IOSTD_RE.search(line)
        if m:
            _record((m.group(2) or m.group(3)).strip(), None, m.group(1), i)
    return [pins[p] for p in order if pins[p].pin]


def _parse_qsf(text: str) -> list[PinAssignment]:
    out, iostd = [], {}
    for m in _QSF_IOSTD_RE.finditer(text):
        iostd[m.group(2)] = m.group(1)
    for i, raw in enumerate(text.splitlines(), 1):
        m = _QSF_LOC_RE.search(_strip_comment(raw))
        if m:
            pin = m.group(1)
            pin = pin[4:] if pin.upper().startswith("PIN_") else pin
            out.append(PinAssignment(port=m.group(2), pin=pin,
                                     io_standard=iostd.get(m.group(2)), line=i))
    return out


_PARSERS = {"pcf": _parse_pcf, "lpf": _parse_lpf, "xdc": _parse_xdc,
            "qsf": _parse_qsf}


def load_pins(path: str | Path) -> PinConstraints:
    """Parse a pin-constraints file; format inferred from the extension."""
    p = Path(path)
    fmt = p.suffix.lstrip(".").lower()
    if fmt not in _PARSERS:
        raise ValueError(
            f"unsupported pin-constraint format {p.suffix!r} "
            f"(expected one of: .pcf .lpf .xdc .qsf)")
    return PinConstraints(path=str(p), fmt=fmt,
                          assignments=_PARSERS[fmt](p.read_text()))


# ------------------------------ validation ------------------------------- #
_BIT_RE = re.compile(r"^(.*)\[(\d+)\]$")

# I/O standard name -> required rail voltage. Covers the common single-ended
# standards across all four vendors' naming schemes.
_IOSTD_VOLTS = {
    "LVCMOS33": 3.3, "LVCMOS25": 2.5, "LVCMOS18": 1.8, "LVCMOS15": 1.5,
    "LVCMOS12": 1.2, "LVTTL": 3.3,
    "3.3-V LVTTL": 3.3, "3.3-V LVCMOS": 3.3, "2.5 V": 2.5, "1.8 V": 1.8,
    "1.5 V": 1.5, "1.2 V": 1.2,
}


def _split_bit(name: str) -> tuple[str, int | None]:
    m = _BIT_RE.match(name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


def check_pins(
    pc: PinConstraints,
    ports,                              # iterable with .name/.width (or (name, width))
    *,
    valid_pins: set[str] | None = None,     # package pins, when a pin DB exists
    io_capacity: int | None = None,         # package I/O budget
    board: dict | None = None,              # board spec (rails / clock sources)
    clock_port: str | None = None,          # detected clock port name
    clock_ns: float | None = None,          # design clock-period constraint
) -> PinReport:
    """Validate pin constraints against the design, package, and board."""
    rep = PinReport()
    norm = [(p.name, p.width) if hasattr(p, "name") else (p[0], p[1])
            for p in ports]
    port_widths = dict(norm)
    rep.total_port_bits = sum(w for _, w in norm)

    # -- constraints referencing ports that don't exist ------------------- #
    covered: dict[str, set[int]] = {}
    for a in pc.assignments:
        base, bit = _split_bit(a.port)
        if base not in port_widths:
            rep.errors.append(
                f"{Path(pc.path).name}:{a.line}: constraint for unknown port "
                f"{a.port!r} (not a top-level port)")
            continue
        w = port_widths[base]
        if bit is None:
            if w > 1:
                rep.errors.append(
                    f"{Path(pc.path).name}:{a.line}: port {base!r} is {w} bits "
                    f"wide but constrained without an index")
                continue
            bit = 0
        elif bit >= w:
            rep.errors.append(
                f"{Path(pc.path).name}:{a.line}: bit {a.port} out of range "
                f"(port is {w} bits)")
            continue
        covered.setdefault(base, set()).add(bit)

    # -- every port bit pinned (the first-shot killer) -------------------- #
    for name, w in norm:
        missing = sorted(set(range(w)) - covered.get(name, set()))
        if len(missing) == w:
            rep.errors.append(
                f"port {name!r} has no pin assignment -- it will be "
                f"auto-placed on a pin your board is not wired to")
        elif missing:
            rep.errors.append(
                f"port {name!r} bits {missing} have no pin assignment")
    rep.constrained_ports = sum(len(v) for v in covered.values())

    # -- double-booked pins ------------------------------------------------ #
    seen: dict[str, str] = {}
    for a in pc.assignments:
        key = a.pin.upper()
        if key in seen and seen[key] != a.port:
            rep.errors.append(
                f"pin {a.pin!r} assigned to both {seen[key]!r} and {a.port!r}")
        seen.setdefault(key, a.port)

    # -- pins exist on the package / fit the budget ------------------------ #
    if valid_pins is not None:
        vp = {str(v).upper() for v in valid_pins}
        for a in pc.assignments:
            if a.pin.upper() not in vp:
                rep.errors.append(
                    f"pin {a.pin!r} (port {a.port!r}) does not exist on this "
                    f"package")
    if io_capacity is not None and len(seen) > io_capacity:
        rep.errors.append(
            f"{len(seen)} pins constrained but the package exposes only "
            f"{io_capacity} user I/O")

    # -- board spec: clock source + voltage rails -------------------------- #
    if board:
        _check_board(pc, rep, board, clock_port, clock_ns)
    return rep


def _check_board(pc: PinConstraints, rep: PinReport, board: dict,
                 clock_port: str | None, clock_ns: float | None) -> None:
    by_port = pc.by_port()

    sources = {str(s.get("pin", "")).upper(): float(s.get("mhz", 0.0))
               for s in board.get("clock_sources", [])}
    if sources and clock_port:
        a = by_port.get(clock_port)
        if a is not None:
            if a.pin.upper() not in sources:
                rep.errors.append(
                    f"clock port {clock_port!r} is pinned to {a.pin!r}, but the "
                    f"board's clock sources are on pins "
                    f"{sorted(sources)} -- the design will have no clock")
            elif clock_ns and clock_ns > 0:
                need = 1000.0 / clock_ns
                have = sources[a.pin.upper()]
                if have + 1e-9 < need:
                    rep.warnings.append(
                        f"board clock on pin {a.pin!r} is {have:g} MHz but the "
                        f"design targets {need:g} MHz -- a PLL must multiply it")

    rails = {round(float(r.get("volts", 0.0)), 2)
             for r in board.get("rails", [])}
    if rails:
        for a in pc.assignments:
            volts = _IOSTD_VOLTS.get((a.io_standard or "").upper()) or \
                _IOSTD_VOLTS.get(a.io_standard or "")
            if volts is not None and round(volts, 2) not in rails:
                rep.errors.append(
                    f"port {a.port!r} uses {a.io_standard} ({volts} V) but the "
                    f"board provides no {volts} V rail")
