from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlTraceWriter:
    """
    Append-only JSONL execution trace.

    Each line represents one runtime event.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write(
        self,
        *,
        event: str,
        run_id: str,
        payload: Any | None = None,
    ) -> None:

        record = {
            "event": event,
            "run_id": run_id,
            "payload": self._serialize(
                payload
            ),
        }

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

            handle.write("\n")

    @classmethod
    def _serialize(
        cls,
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if hasattr(
            value,
            "model_dump",
        ):
            return cls._serialize(
                value.model_dump(
                    mode="json"
                )
            )

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): cls._serialize(
                    item
                )
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            (list, tuple),
        ):
            return [
                cls._serialize(item)
                for item in value
            ]

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        return str(value)
