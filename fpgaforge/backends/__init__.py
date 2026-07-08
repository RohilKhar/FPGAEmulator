"""Flow backends: adapters that turn a Design + FlowOptions into a RunResult.

`Ice40Backend`/`Ecp5Backend` drive the open-source Lattice toolchain;
`VivadoBackend` (AMD) and `QuartusBackend` (Intel) drive the vendor CLIs.
`MockBackend` produces deterministic pseudo-metrics so the platform runs fully
offline. All implement the tool-agnostic `Backend` interface, so the corpus,
model, optimizer, and readiness stack work across every vendor unchanged.
"""

from .base import Backend, Design, FlowOptions, RunMetrics, RunResult
from .ecp5 import Ecp5Backend
from .ice40 import Ice40Backend
from .mock import MockBackend
from .quartus import QuartusBackend
from .vivado import VivadoBackend
from .gowin import GowinBackend

__all__ = [
    "Backend",
    "Design",
    "FlowOptions",
    "RunMetrics",
    "RunResult",
    "Ice40Backend",
    "Ecp5Backend",
    "VivadoBackend",
    "QuartusBackend",
    "GowinBackend",
    "MockBackend",
]
