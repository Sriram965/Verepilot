"""
Append-only JSONL execution trace.

Each line is one event:

    {"seq": 7, "t": 1.532, "ts": "2026-10-09T10:15:02.114Z",
     "event": "tool_result", "run_id": "...", "payload": {...}}

seq   1, 2, 3... in the order events were written
t     seconds since the trace was opened (monotonic clock, so it is safe for
      measuring durations)
ts    wall-clock time in UTC

Payloads may be Pydantic models, LangChain messages, dicts, lists or plain
values; they are converted to JSON. Writes are thread-safe and each one
reaches the file immediately, so a crashed run still leaves a usable trace.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlTraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._seq = 0
        self._started = time.monotonic()
        self._lock = threading.Lock()

    def write(
        self,
        *,
        event: str,
        run_id: str,
        payload: Any | None = None,
    ) -> None:
        with self._lock:
            self._seq += 1

            record = {
                "seq": self._seq,
                "t": round(time.monotonic() - self._started, 3),
                "ts": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                )[:-3]
                + "Z",
                "event": event,
                "run_id": run_id,
                "payload": self._serialize(payload),
            }

            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    @classmethod
    def _serialize(cls, value: Any) -> Any:
        if value is None:
            return None

        if hasattr(value, "model_dump"):
            return cls._serialize(value.model_dump(mode="json"))

        if isinstance(value, dict):
            return {str(key): cls._serialize(item) for key, item in value.items()}

        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls._serialize(item) for item in value]

        if isinstance(value, (str, int, float, bool)):
            return value

        return str(value)


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load every event of a trace file, in order."""

    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]