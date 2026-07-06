"""Standardized implementation corpus.

Every flow run is appended as one JSON line: the raw material for predictive
models and, eventually, for identifying stages where our methods beat the
vendor flow. This is the "enormous corpus" the platform is built to accumulate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from .backends.base import RunResult

DEFAULT_CORPUS = Path("data/corpus.jsonl")


class Corpus:
    def __init__(self, path: str | Path = DEFAULT_CORPUS) -> None:
        self.path = Path(path)

    def append(self, result: RunResult, extra: dict[str, Any] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = result.to_dict()
        record["timestamp"] = time.time()
        if extra:
            record.update(extra)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def load(self) -> list[dict[str, Any]]:
        return list(self.records())

    def __len__(self) -> int:
        return sum(1 for _ in self.records())

    def training_rows(self) -> list[dict[str, Any]]:
        """Successful, routed runs usable as (features -> fmax) samples."""
        rows: list[dict[str, Any]] = []
        for rec in self.records():
            metrics = rec.get("metrics", {})
            if rec.get("success") and metrics.get("routed_ok") and metrics.get("fmax_mhz"):
                rows.append(rec)
        return rows
