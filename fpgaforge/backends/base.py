"""Core data model and the tool-agnostic Backend interface.

Everything downstream (corpus, model, optimizer) speaks in terms of these
dataclasses, which is what lets us swap the iCE40 flow for a vendor flow later
without touching the rest of the platform.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Design:
    """A design to implement: one or more RTL files plus a top module.

    `target` selects the device (e.g. "ice40_up5k"). `clock_ns` is the target
    clock period constraint used to derive the requested Fmax.
    """

    rtl_files: tuple[str, ...]
    top: str
    target: str = "ice40_up5k"
    clock_ns: float = 10.0  # 100 MHz target by default
    name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rtl_files, (str, Path)):
            object.__setattr__(self, "rtl_files", (str(self.rtl_files),))
        else:
            object.__setattr__(
                self, "rtl_files", tuple(str(p) for p in self.rtl_files)
            )
        if self.name is None:
            first = Path(self.rtl_files[0]).stem if self.rtl_files else self.top
            object.__setattr__(self, "name", f"{first}.{self.top}")

    @property
    def target_freq_mhz(self) -> float:
        return 1000.0 / self.clock_ns if self.clock_ns > 0 else 0.0

    def rtl_hash(self) -> str:
        """Content hash of all RTL files, used to group runs in the corpus."""
        h = hashlib.sha256()
        for p in self.rtl_files:
            path = Path(p)
            if path.exists():
                h.update(path.read_bytes())
            else:
                h.update(p.encode())
        h.update(self.top.encode())
        return h.hexdigest()[:16]

    def design_id(self) -> str:
        return f"{self.name}:{self.rtl_hash()}"


@dataclass(frozen=True)
class FlowOptions:
    """Implementation knobs the optimizer searches over.

    Kept small and explicit for the first milestone. Each field maps to a real
    lever in the open-source flow (or, for `pipeline_output`, an RTL transform
    stub that later becomes automatic pipelining).
    """

    abc9: bool = False          # yosys: use the abc9 timing-driven mapper
    retime: bool = False        # yosys synth_ice40 -retime
    dsp: bool = True            # yosys synth_ice40 -dsp (map multipliers to DSP)
    seed: int = 1               # nextpnr placement seed
    placer: str = "heap"        # nextpnr placer: "heap" or "sa"
    pipeline_output: bool = False  # RTL transform stub (see rtl_transform.py)

    def key(self) -> str:
        """Stable identifier for de-duplicating candidate runs."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowOptions":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class RunMetrics:
    """Parsed outcome of a single implementation run."""

    fmax_mhz: float = 0.0
    target_freq_mhz: float = 0.0
    crit_path_ns: float = 0.0
    luts: int = 0
    ffs: int = 0
    bram: int = 0
    dsp: int = 0
    carries: int = 0
    routed_ok: bool = False

    @property
    def meets_timing(self) -> bool:
        return self.routed_ok and self.fmax_mhz >= self.target_freq_mhz

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["meets_timing"] = self.meets_timing
        return d


@dataclass
class RunResult:
    """Full record of one backend run: inputs, features, metrics, artifacts."""

    design_id: str
    options: FlowOptions
    metrics: RunMetrics
    features: dict[str, float] = field(default_factory=dict)
    success: bool = False
    backend: str = ""
    workdir: str | None = None
    bitstream_path: str | None = None
    sdf_path: str | None = None            # nextpnr SDF (delay back-annotation)
    routed_netlist_path: str | None = None  # nextpnr post-route JSON netlist
    log: str = ""
    error: str | None = None

    def diagnostics(self, limit: int = 25):
        """Structured errors/warnings parsed from the captured tool log."""
        from ..diagnostics import extract

        return extract(self.log, limit)

    def errors(self, limit: int = 25):
        return [d for d in self.diagnostics(limit) if d.severity == "error"]

    def warnings(self, limit: int = 25):
        return [d for d in self.diagnostics(limit) if d.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "options": self.options.to_dict(),
            "metrics": self.metrics.to_dict(),
            "features": self.features,
            "success": self.success,
            "backend": self.backend,
            "bitstream_path": self.bitstream_path,
            "error": self.error,
            "errors": [d.format() for d in self.errors()],
        }


class Backend(ABC):
    """A flow backend turns (Design, FlowOptions) into a RunResult.

    Implementations must be tolerant: on tool failure they should return a
    RunResult with success=False and a populated `error`, never raise, so the
    optimizer can keep exploring other candidates.
    """

    name: str = "backend"

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the backend's required tools are installed and runnable."""

    @abstractmethod
    def run(self, design: Design, options: FlowOptions, workdir: Path) -> RunResult:
        """Implement `design` with `options`, writing artifacts under `workdir`."""
