"""Bitstream-level iCE40 fabric emulator.

Load a real bitstream, decode what the fabric is configured to do, and run it.
"""

from .bitstream import Bitstream, Tile, load, parse_asc, unpack_bin
from .fabric import FabricConfig, LogicCell, decode_fabric, decode_logic_tile
from .peripherals import BoardConfig, BoardResult, UartCapture, classify_pins
from .emulator import (
    Emulator,
    EmulationResult,
    ProofResult,
    VerificationResult,
    emulate,
    emulate_board,
    prove,
    verify_bitstream,
)
from .mutation import MutationResult, flip_logic_bit, mutation_test
from .timing_emu import TimingEmulationResult, timing_emulate

__all__ = [
    "Bitstream",
    "Tile",
    "load",
    "parse_asc",
    "unpack_bin",
    "FabricConfig",
    "LogicCell",
    "decode_fabric",
    "decode_logic_tile",
    "BoardConfig",
    "BoardResult",
    "UartCapture",
    "classify_pins",
    "Emulator",
    "EmulationResult",
    "VerificationResult",
    "ProofResult",
    "emulate",
    "emulate_board",
    "verify_bitstream",
    "prove",
    "MutationResult",
    "flip_logic_bit",
    "mutation_test",
    "TimingEmulationResult",
    "timing_emulate",
]
