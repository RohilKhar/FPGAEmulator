"""Parse a real iCE40 bitstream into an in-memory fabric image.

An iCE40 bitstream (`.bin`) is the exact byte stream that gets flashed to the
device. Project IceStorm's ``iceunpack`` losslessly converts it to an ASCII
form (`.asc`) that lists every configuration tile as a bit matrix. This module
parses that ASCII form into structured tiles so the rest of the emulator can
decode what the fabric is actually configured to do -- LUT truth tables, flop
enables, routing, IO -- straight from the bits that would drive the silicon.

Nothing here needs the vendor tools *except* the one-shot ``.bin -> .asc``
unpack; the parser itself is pure Python and unit-testable on a text fixture.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Tile:
    """One configuration tile: a rectangular matrix of config bits.

    ``rows`` is a list of strings of '0'/'1'; ``rows[r][c]`` is bit (row r,
    col c), matching Project IceStorm's ``B<r>[<c>]`` addressing.
    """

    kind: str  # "logic" | "io" | "ramb" | "ramt" | "dsp" | "ipcon"
    x: int
    y: int
    rows: list[str] = field(default_factory=list)

    def bit(self, row: int, col: int) -> int:
        return 1 if self.rows[row][col] == "1" else 0

    @property
    def n_set(self) -> int:
        return sum(line.count("1") for line in self.rows)


@dataclass
class Bitstream:
    """A parsed bitstream image."""

    device: str = ""  # e.g. "5k", "8k", "1k"
    comment: str = ""
    tiles: dict[tuple[str, int, int], Tile] = field(default_factory=dict)
    ram_data: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    symbols: dict[int, str] = field(default_factory=dict)
    warmboot: str | None = None
    source_path: str | None = None

    def tiles_of(self, kind: str) -> list[Tile]:
        return [t for k, t in self.tiles.items() if k[0] == kind]

    def logic_tiles(self) -> list[Tile]:
        return self.tiles_of("logic")


# Directives that introduce a tile whose following lines are a bit matrix.
_TILE_KINDS = {
    ".io_tile": "io",
    ".logic_tile": "logic",
    ".ramb_tile": "ramb",
    ".ramt_tile": "ramt",
    ".dsp_tile": "dsp",
    ".ipcon_tile": "ipcon",
}


def parse_asc(text: str) -> Bitstream:
    """Parse the text of an iCE40 ``.asc`` file into a :class:`Bitstream`."""
    bs = Bitstream()
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if not line.startswith("."):
            # Stray data outside a directive; ignore.
            continue

        parts = line.split()
        directive = parts[0]
        args = parts[1:]

        if directive == ".device":
            bs.device = args[0] if args else ""
        elif directive == ".comment":
            comment_lines: list[str] = []
            while i < n and not lines[i].startswith("."):
                comment_lines.append(lines[i])
                i += 1
            bs.comment = "\n".join(comment_lines)
        elif directive in _TILE_KINDS:
            kind = _TILE_KINDS[directive]
            x, y = int(args[0]), int(args[1])
            rows: list[str] = []
            while i < n and lines[i] and not lines[i].startswith("."):
                rows.append(lines[i].strip())
                i += 1
            bs.tiles[(kind, x, y)] = Tile(kind=kind, x=x, y=y, rows=rows)
        elif directive == ".ram_data":
            x, y = int(args[0]), int(args[1])
            data: list[str] = []
            while i < n and lines[i] and not lines[i].startswith("."):
                data.append(lines[i].strip())
                i += 1
            bs.ram_data[(x, y)] = data
        elif directive == ".sym":
            if len(args) >= 2:
                try:
                    bs.symbols[int(args[0])] = " ".join(args[1:])
                except ValueError:
                    pass
        elif directive == ".warmboot":
            bs.warmboot = " ".join(args)
        else:
            # Unknown directive (e.g. .extra_bit); skip any matrix lines.
            while i < n and lines[i] and not lines[i].startswith("."):
                i += 1
    return bs


def unpack_bin(bin_path: str | Path, iceunpack: str = "iceunpack",
               out_path: str | Path | None = None) -> Path:
    """Convert a binary ``.bin`` bitstream to ``.asc`` using ``iceunpack``.

    Returns the path to the written ``.asc`` file. Raises if the tool is
    missing or fails.
    """
    bin_path = Path(bin_path)
    if shutil.which(iceunpack) is None:
        raise RuntimeError(f"{iceunpack} not found on PATH (Project IceStorm)")
    out = Path(out_path) if out_path else bin_path.with_suffix(".asc")
    proc = subprocess.run(
        [iceunpack, str(bin_path), str(out)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"iceunpack failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return out


def load(path: str | Path, iceunpack: str = "iceunpack") -> Bitstream:
    """Load a bitstream from a ``.asc`` or ``.bin`` file into a Bitstream.

    ``.bin`` inputs are unpacked with ``iceunpack`` first (proving we read the
    exact bytes that would be flashed to the chip).
    """
    path = Path(path)
    if path.suffix == ".bin":
        asc = unpack_bin(path, iceunpack=iceunpack)
    else:
        asc = path
    bs = parse_asc(asc.read_text())
    bs.source_path = str(path)
    return bs
