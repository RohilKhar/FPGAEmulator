"""Flow backends: adapters that turn a Design + FlowOptions into a RunResult.

`Ice40Backend` drives the real open-source iCE40 toolchain. `MockBackend`
produces deterministic pseudo-metrics so the platform runs fully offline.
Both implement the tool-agnostic `Backend` interface, leaving room for future
vendor adapters (Vivado / Quartus).
"""

from .base import Backend, Design, FlowOptions, RunMetrics, RunResult
from .ice40 import Ice40Backend
from .mock import MockBackend

__all__ = [
    "Backend",
    "Design",
    "FlowOptions",
    "RunMetrics",
    "RunResult",
    "Ice40Backend",
    "MockBackend",
]
