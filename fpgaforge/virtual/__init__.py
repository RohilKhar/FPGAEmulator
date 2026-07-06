"""Virtual FPGA bring-up.

Take RTL, synthesize it down to real FPGA primitives, and run that *mapped*
netlist in a cycle-accurate virtual fabric (Icarus + Yosys sim models) with a
generated virtual-board harness. This validates that the implemented design
comes up and behaves before any physical deployment.
"""

from .board import BringUpConfig, detect_clock, detect_reset, render_testbench
from .vfpga import BringUpResult, VirtualFPGA, bringup

__all__ = [
    "BringUpConfig",
    "BringUpResult",
    "VirtualFPGA",
    "bringup",
    "render_testbench",
    "detect_clock",
    "detect_reset",
]
