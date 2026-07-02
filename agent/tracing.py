"""Structured, replayable run traces.

Every step of every run is appended as one JSON line: what the model thought,
what it did, what came back, how long it took. A run can be audited or
replayed after the fact — the non-negotiable for agents in real workflows.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Tracer:
    trace_dir: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[dict] = field(default_factory=list)

    def log(self, kind: str, **data) -> None:
        self.events.append({"run_id": self.run_id, "ts": round(time.time(), 3),
                            "step": len(self.events), "kind": kind, **data})

    def flush(self) -> Path:
        path = Path(self.trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        f = path / f"{self.run_id}.jsonl"
        with f.open("w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
        return f
