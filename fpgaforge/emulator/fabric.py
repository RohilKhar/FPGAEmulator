"""Decode a parsed bitstream into the logic the fabric is configured to run.

This reads the configuration bits straight out of the tiles and reconstructs,
per logic cell, the 16-entry LUT truth table plus the flip-flop / carry
control bits -- i.e. exactly what each silicon cell will compute once the
bitstream is loaded. The bit layout follows Project IceStorm (see icebox.py
``get_lutff_*``): each 8-cell logic tile stores 20 bits per cell in columns
36..45 of the two rows for that cell, and the 16 LUT bits are a fixed
permutation of those.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bitstream import Bitstream, Tile

# Fixed bit permutation mapping the 20 raw cell bits -> 16 LUT truth-table bits
# (Project IceStorm icebox.py: get_lutff_lut_bits).
_LUT_PERM = [4, 14, 15, 5, 6, 16, 17, 7, 3, 13, 12, 2, 1, 11, 10, 0]
# Sequential/control bits within the 20: CarryEnable, DffEnable, Set_NoReset,
# AsyncSetReset.
_SEQ_PERM = [8, 9, 18, 19]


def _cell_raw_bits(tile: Tile, index: int) -> list[int]:
    """The 20 raw config bits for logic cell ``index`` (0..7) in ``tile``."""
    bits = [0] * 20
    for k, line in enumerate(tile.rows):
        if k // 2 != index:
            continue
        for col in range(36, 46):
            bitnum = (col - 36) + 10 * (k % 2)
            bits[bitnum] = 1 if line[col] == "1" else 0
    return bits


@dataclass
class LogicCell:
    """One configured iCE40 logic cell (LUT4 + optional DFF + carry)."""

    x: int
    y: int
    index: int  # 0..7 within the tile
    lut_init: int  # 16-bit truth table, bit i = output for input combo i
    carry_enable: bool
    dff_enable: bool
    set_noreset: bool
    async_setreset: bool

    @property
    def lut_used(self) -> bool:
        # A LUT is "in use" if it computes something (non-constant) or feeds a
        # flop. A carry-only cell with a constant LUT is not counted as a LUT.
        return self.lut_init not in (0x0000, 0xFFFF) or self.dff_enable

    @property
    def used(self) -> bool:
        return self.lut_used or self.carry_enable

    def lut_equation(self, names: tuple[str, str, str, str] = ("i0", "i1", "i2", "i3")) -> str:
        """Render the LUT truth table as a sum-of-products Boolean equation."""
        minterms = [i for i in range(16) if (self.lut_init >> i) & 1]
        if not minterms:
            return "0"
        if len(minterms) == 16:
            return "1"
        terms = []
        for m in minterms:
            lits = []
            for b in range(4):
                lits.append(names[b] if (m >> b) & 1 else f"~{names[b]}")
            terms.append(" & ".join(lits))
        return " | ".join(f"({t})" for t in terms)

    def truth_table(self) -> list[int]:
        return [(self.lut_init >> i) & 1 for i in range(16)]


def decode_logic_tile(tile: Tile) -> list[LogicCell]:
    """Decode all 8 logic cells of one logic tile."""
    cells: list[LogicCell] = []
    for index in range(8):
        raw = _cell_raw_bits(tile, index)
        lut_bits = [raw[i] for i in _LUT_PERM]
        lut_init = 0
        for i, b in enumerate(lut_bits):
            lut_init |= b << i
        seq = [raw[i] for i in _SEQ_PERM]
        cells.append(
            LogicCell(
                x=tile.x, y=tile.y, index=index,
                lut_init=lut_init,
                carry_enable=bool(seq[0]),
                dff_enable=bool(seq[1]),
                set_noreset=bool(seq[2]),
                async_setreset=bool(seq[3]),
            )
        )
    return cells


@dataclass
class FabricConfig:
    """The decoded contents of a whole bitstream's programmable fabric."""

    device: str = ""
    cells: list[LogicCell] = field(default_factory=list)
    io_tiles: int = 0
    ramb_tiles: int = 0
    dsp_tiles: int = 0
    logic_tiles: int = 0

    @property
    def used_cells(self) -> list[LogicCell]:
        return [c for c in self.cells if c.used]

    @property
    def luts_used(self) -> int:
        return sum(1 for c in self.cells if c.lut_used)

    @property
    def dffs_used(self) -> int:
        return sum(1 for c in self.cells if c.dff_enable)

    @property
    def carries_used(self) -> int:
        return sum(1 for c in self.cells if c.carry_enable)

    def summary(self) -> str:
        lines = [
            "decoded fabric configuration",
            f"device      : iCE40-{self.device}",
            f"logic tiles : {self.logic_tiles} ({len(self.cells)} cells)",
            f"LUTs used   : {self.luts_used}",
            f"DFFs used   : {self.dffs_used}",
            f"carry cells : {self.carries_used}",
            f"IO tiles    : {self.io_tiles}",
            f"BRAM tiles  : {self.ramb_tiles}",
        ]
        if self.dsp_tiles:
            lines.append(f"DSP tiles   : {self.dsp_tiles}")
        return "\n".join(lines)


def decode_fabric(bs: Bitstream) -> FabricConfig:
    """Decode every logic tile of a bitstream into a :class:`FabricConfig`."""
    cfg = FabricConfig(device=bs.device)
    logic = bs.logic_tiles()
    cfg.logic_tiles = len(logic)
    for tile in logic:
        cfg.cells.extend(decode_logic_tile(tile))
    cfg.io_tiles = len(bs.tiles_of("io"))
    cfg.ramb_tiles = len(bs.tiles_of("ramb"))
    cfg.dsp_tiles = len(bs.tiles_of("dsp"))
    return cfg
