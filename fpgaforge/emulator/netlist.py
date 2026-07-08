"""Reconstruct a simulatable netlist from a bitstream and make it runnable.

``icebox_vlog`` turns an ``.asc`` back into a Verilog netlist of the *actually
configured* fabric (resolved LUTs + routing + flops). That is the heart of a
bitstream-level emulator: whatever we simulate here is what the silicon would
do, independent of the original RTL. This module:

  * locates the Project IceStorm chip database and parses its package pin list,
  * generates a ``.pcf`` so the flow assigns real pins and ``icebox_vlog`` can
    restore the original top-level port names,
  * runs ``icebox_vlog`` and normalizes its output so Icarus Verilog accepts it
    (icebox emits ANSI ports *and* separate ``reg`` decls, which iverilog
    rejects), and
  * builds a thin wrapper that re-buses the per-bit scalar ports back into the
    original vector ports so the standard virtual-board harness can drive it.
"""

from __future__ import annotations

import glob
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..devices import by_backend
from ..virtual.board import Port

# target -> (icestorm device tag, nextpnr flag, default package).
# Only IceStorm-reconstructable devices belong here; derived from the registry.
DEVICE_INFO: dict[str, tuple[str, str, str]] = {
    d.target: (d.chipdb_tag, d.pnr_flag, d.package)
    for d in by_backend("ice40")
    if d.reconstructor == "icestorm"
}


def find_chipdb(device_tag: str) -> Path | None:
    """Locate ``chipdb-<device_tag>.txt`` from a Project IceStorm install."""
    name = f"chipdb-{device_tag}.txt"
    roots = [
        Path(sys.prefix) / "share" / "icestorm" / "chipdb",
        Path("/opt/homebrew/share/icestorm/chipdb"),
        Path("/usr/local/share/icestorm/chipdb"),
        Path("/usr/share/icestorm/chipdb"),
    ]
    for r in roots:
        cand = r / name
        if cand.exists():
            return cand
    # Fall back to a broad glob (covers Homebrew Cellar installs).
    for base in ("/opt/homebrew", "/usr/local", sys.prefix):
        hits = glob.glob(f"{base}/**/icestorm/chipdb/{name}", recursive=True)
        if hits:
            return Path(hits[0])
    return None


def parse_package_pins(chipdb_path: Path, package: str) -> list[str]:
    """Return the list of usable pin identifiers for ``package``."""
    pins: list[str] = []
    in_section = False
    for line in chipdb_path.read_text().splitlines():
        if line.startswith(".pins "):
            in_section = line.split()[1] == package
            continue
        if in_section:
            if line.startswith("."):
                break
            parts = line.split()
            if parts:
                pins.append(parts[0])
    return pins


def expand_bits(ports: list[Port]) -> list[str]:
    """Flatten ports into per-bit signal names (``clk``, ``count[0]`` ...)."""
    names: list[str] = []
    for p in ports:
        if p.width <= 1:
            names.append(p.name)
        else:
            names.extend(f"{p.name}[{i}]" for i in range(p.width))
    return names


def generate_pcf(ports: list[Port], pins: list[str]) -> str:
    """Assign every IO bit to a distinct package pin -> a ``.pcf`` string."""
    bits = expand_bits(ports)
    if len(bits) > len(pins):
        raise RuntimeError(
            f"design needs {len(bits)} IO pins but package only exposes {len(pins)}"
        )
    return "".join(f"set_io {b} {pins[i]}\n" for i, b in enumerate(bits))


def reconstruct(
    asc_path: str | Path,
    out_v: str | Path,
    pcf_path: str | Path | None = None,
    module: str = "chip",
    icebox_vlog: str = "icebox_vlog",
) -> str:
    """Run ``icebox_vlog`` to rebuild a netlist; return the (normalized) text.

    The normalized text is written to ``out_v`` and returned.
    """
    if shutil.which(icebox_vlog) is None:
        raise RuntimeError(f"{icebox_vlog} not found on PATH (Project IceStorm)")
    cmd = [icebox_vlog, "-n", module]
    if pcf_path is not None:
        cmd += ["-p", str(pcf_path)]
    cmd.append(str(asc_path))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"icebox_vlog failed: {proc.stderr.strip()}")
    text = normalize_netlist(proc.stdout)
    Path(out_v).write_text(text)
    return text


_PORT_DECL_RE = re.compile(r"(input|output)\s+(\\\S+\s|\w+)")


def normalize_netlist(text: str) -> str:
    """Make ``icebox_vlog`` output compile under Icarus Verilog.

    icebox emits an ANSI header like ``module m(output \\count[0] , ...)`` and
    then, separately, ``reg \\count[0] = 0 ;`` -- which iverilog rejects as a
    redeclaration. We turn the output ports into ``output reg`` and convert the
    standalone reg decls for those ports into ``initial`` assignments.
    """
    m = re.search(r"module\s+\w+\s*\((.*?)\);", text, re.S)
    if not m:
        return text
    header = m.group(1)

    outputs: list[str] = []
    all_ports: list[str] = []
    for kind, name in _PORT_DECL_RE.findall(header):
        all_ports.append(name.strip())
        if kind == "output":
            outputs.append(name.strip())

    def _promote(match: re.Match) -> str:
        kind, name = match.group(1), match.group(2)
        if kind == "output":
            return f"output reg {name}"
        return match.group(0)

    new_header = _PORT_DECL_RE.sub(_promote, header)
    text = text[: m.start(1)] + new_header + text[m.end(1) :]

    out_set = {o.strip() for o in outputs}
    port_set = {p.strip() for p in all_ports}

    # icebox re-declares input ports as bare `wire NAME ;`; iverilog rejects the
    # redeclaration, so drop those lines (the ANSI port already provides the net).
    def _drop_port_wire(match: re.Match) -> str:
        return "" if match.group(1).strip() in port_set else match.group(0)

    text = re.sub(r"^wire\s+(\\\S+\s|\w+\s*);\s*$", _drop_port_wire, text, flags=re.M)

    def _reg_to_initial(match: re.Match) -> str:
        name = match.group(1).strip()
        val = match.group(2)
        if name.strip() in out_set:
            return f"initial {match.group(1)}= {val};"
        return match.group(0)

    text = re.sub(r"reg\s+(\\\S+\s|\w+\s*)=\s*(\S+)\s*;", _reg_to_initial, text)
    return text


def _port_ref(name: str, bit: int | None) -> str:
    """Escaped identifier for a (possibly bit-selected) reconstructed port."""
    if bit is None:
        return name
    return f"\\{name}[{bit}] "


def make_rebus_wrapper(
    inner_module: str, wrapper_name: str, ports: list[Port]
) -> str:
    """Wrap the scalar-port reconstructed module in one with the original buses.

    Lets the standard virtual-board harness instantiate ``wrapper_name`` with
    the design's original vector ports.
    """
    port_names = ", ".join(p.name for p in ports)
    lines = [f"module {wrapper_name} ({port_names});"]
    for p in ports:
        rng = f"[{p.width - 1}:0] " if p.width > 1 else ""
        lines.append(f"  {p.direction} {rng}{p.name};")
    conns: list[str] = []
    for p in ports:
        if p.width <= 1:
            conns.append(f".{_port_ref(p.name, None)}({p.name})")
        else:
            for i in range(p.width):
                conns.append(f".{_port_ref(p.name, i)}({p.name}[{i}])")
    lines.append(f"  {inner_module} u ({', '.join(conns)});")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"
