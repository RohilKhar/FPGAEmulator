"""RTL transform seam.

This is where source-level optimizations (automatic pipelining, high-fanout
replication, arithmetic rebalancing, memory rewriting for BRAM inference) will
live. For the first milestone it is an explicit, safe pass-through: it copies
the RTL into the run workdir and records which transforms were requested, so the
optimizer's search space already includes RTL-level knobs without risking
correctness. Real transforms replace `_apply` incrementally.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .backends.base import Design, FlowOptions


def prepare_rtl(design: Design, options: FlowOptions, workdir: Path) -> list[str]:
    """Materialize (possibly transformed) RTL into `workdir`; return file paths.

    Currently a structural pass-through. `options.pipeline_output` is carried
    through to the flow (nextpnr/yosys can still act on retiming), and this is
    the hook point for genuine source rewrites later.
    """
    src_dir = workdir / "rtl"
    src_dir.mkdir(parents=True, exist_ok=True)
    out_files: list[str] = []
    for f in design.rtl_files:
        src = Path(f)
        dst = src_dir / src.name
        if src.exists():
            shutil.copyfile(src, dst)
        else:
            # Treat the string itself as inline RTL.
            dst.write_text(str(f))
        _apply(dst, options)
        out_files.append(str(dst))
    return out_files


def _apply(rtl_path: Path, options: FlowOptions) -> None:
    """Placeholder for real source-to-source transforms (no-op for now)."""
    # Intentionally left as a no-op. Future: parse, pipeline critical paths,
    # replicate high-fanout nets, rebalance arithmetic, rewrite memories.
    return None
