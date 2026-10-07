"""
Environment contract for VeriPilot benchmark sites.

Every benchmark site (shop, forms, booking portal, ...) implements the
`Environment` protocol below. The protocol is deliberately small, and its
most important member is `snapshot()`.

Authoritative state
-------------------
`snapshot(session_id)` returns the environment's TRUE internal state as plain
JSON-compatible data. It is the single source of truth used by the evaluation
oracle. It reads server-side state directly; it never goes through HTTP, the
browser, or anything the agent could observe or influence.

A snapshot has exactly two top-level sections:

    {
        "session": {...},   # state owned by the given session
                            # (its cart, its form submissions, its bookings)
        "shared":  {...},   # state not owned by any one session
                            # (inventory, global order log, ...)
    }

Splitting the two lets a task author write expectations about "my session"
without knowing the random session id, while still letting the oracle catch
side effects that leak into shared state.

Boundary rules
--------------
- The agent and the in-loop verifier receive only `base_url`. They must never
  be handed the Environment object or its snapshot.
- Only the evaluation oracle calls `snapshot()`.
- A snapshot is read-only evidence. Callers get a normalized deep copy, so
  they can never mutate live environment state through it.
- Snapshots must be deterministic: the same underlying state always produces
  the same snapshot (and therefore the same hash). Do not include timestamps,
  random ids, or insertion-order-dependent data.

This module contains no site-specific logic.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Protocol, TypedDict, runtime_checkable


class Snapshot(TypedDict):
    session: dict[str, Any]
    shared: dict[str, Any]


class SnapshotError(ValueError):
    """Raised when an environment returns a malformed snapshot."""


@runtime_checkable
class Environment(Protocol):
    @property
    def base_url(self) -> str:
        """Loopback origin the agent is allowed to talk to."""
        ...

    def start(self) -> "Environment":
        ...

    def stop(self) -> None:
        ...

    def __enter__(self) -> "Environment":
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        ...

    def snapshot(self, session_id: str) -> Snapshot:
        """Authoritative state for `session_id`. Oracle use only."""   
        ...


def _check_json(value: Any, path: str) -> None:
    """Reject anything that is not strictly JSON-representable."""

    if value is None or isinstance(value, (bool, str)):
        return

    if isinstance(value, int):
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise SnapshotError(
                f"non-finite float at {path or '<root>'}"
            )
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_json(item, f"{path}[{index}]")
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SnapshotError(
                    f"non-string key {key!r} at {path or '<root>'}"
                )
            _check_json(item, f"{path}.{key}" if path else key)
        return

    raise SnapshotError(
        f"unsupported type {type(value).__name__} "
        f"at {path or '<root>'}"
    )


def validate_snapshot(snapshot: Any) -> Snapshot:
    """
    Validate an environment snapshot and return a normalized deep copy.

    Raises SnapshotError if the snapshot does not have exactly the
    `session` and `shared` sections, or contains data that is not strictly
    JSON-representable (tuples, sets, non-string keys, NaN, objects).
    """

    if not isinstance(snapshot, dict):
        raise SnapshotError("snapshot must be a dict")

    if set(snapshot) != {"session", "shared"}:
        raise SnapshotError(
            "snapshot must have exactly the keys "
            "'session' and 'shared', got "
            f"{sorted(snapshot)!r}"
        )

    for section in ("session", "shared"):
        if not isinstance(snapshot[section], dict):
            raise SnapshotError(
                f"snapshot['{section}'] must be a dict"
            )

    _check_json(snapshot, "")

    return json.loads(json.dumps(snapshot, allow_nan=False))


def take_snapshot(
    environment: Environment,
    session_id: str,
) -> Snapshot:
    """Oracle entry point: snapshot an environment, validated and copied."""

    return validate_snapshot(environment.snapshot(session_id))


def snapshot_hash(snapshot: Snapshot) -> str:
    """Stable content hash, for traces and replay checks."""

    canonical = json.dumps(
        validate_snapshot(snapshot),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()