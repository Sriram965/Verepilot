"""
Generic structural comparison of two JSON-like states.

Used by the evaluation oracle to compare the environment's authoritative
snapshot against the state a task expects. It knows nothing about carts,
forms, or any particular site.

Paths are RFC 6901 JSON Pointers over the snapshot, for example
"/session/cart/Laptop_A". Keys containing "~" or "/" are escaped as "~0" and
"~1". The empty path "" is the root and is not a valid ignore path.

Each difference is one of:

    missing     expected has it, observed does not
    unexpected  observed has it, expected does not   (collateral damage)
    changed     both have it, values or types differ

Comparison rules:
- dicts are compared key by key, lists index by index;
- bool is never equal to an int (True != 1);
- int and float compare numerically (1 == 1.0);
- everything else must match in type and value.

`ignore_paths` removes a path and everything beneath it from the comparison,
for fields that are legitimately volatile.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel


class StateDiff(BaseModel):
    path: str
    kind: Literal["missing", "unexpected", "changed"]
    expected: Any = None
    observed: Any = None


def escape_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def is_valid_pointer(path: str) -> bool:
    return path.startswith("/") and len(path) > 1


def _ignored(path: str, ignore: tuple[str, ...]) -> bool:
    return any(path == p or path.startswith(p + "/") for p in ignore)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalars_equal(a: Any, b: Any) -> bool:
    if _is_number(a) and _is_number(b):
        return a == b

    return type(a) is type(b) and a == b


def _walk(
    expected: Any,
    observed: Any,
    path: str,
    ignore: tuple[str, ...],
    out: list[StateDiff],
) -> None:
    if _ignored(path, ignore):
        return

    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(set(expected) | set(observed)):
            child = f"{path}/{escape_token(key)}"

            if _ignored(child, ignore):
                continue

            if key not in observed:
                out.append(
                    StateDiff(path=child, kind="missing", expected=expected[key])
                )
            elif key not in expected:
                out.append(
                    StateDiff(path=child, kind="unexpected", observed=observed[key])
                )
            else:
                _walk(expected[key], observed[key], child, ignore, out)
        return

    if isinstance(expected, list) and isinstance(observed, list):
        for index in range(max(len(expected), len(observed))):
            child = f"{path}/{index}"

            if _ignored(child, ignore):
                continue

            if index >= len(observed):
                out.append(
                    StateDiff(path=child, kind="missing", expected=expected[index])
                )
            elif index >= len(expected):
                out.append(
                    StateDiff(path=child, kind="unexpected", observed=observed[index])
                )
            else:
                _walk(expected[index], observed[index], child, ignore, out)
        return

    if isinstance(expected, (dict, list)) or isinstance(observed, (dict, list)):
        out.append(
            StateDiff(
                path=path or "/",
                kind="changed",
                expected=expected,
                observed=observed,
            )
        )
        return

    if not _scalars_equal(expected, observed):
        out.append(
            StateDiff(
                path=path or "/",
                kind="changed",
                expected=expected,
                observed=observed,
            )
        )


def diff_state(
    expected: Any,
    observed: Any,
    ignore_paths: Iterable[str] = (),
) -> list[StateDiff]:
    """All differences between `expected` and `observed`, sorted by path."""

    ignore = tuple(ignore_paths)

    for path in ignore:
        if not is_valid_pointer(path):
            raise ValueError(f"invalid ignore path {path!r}")

    out: list[StateDiff] = []
    _walk(expected, observed, "", ignore, out)
    out.sort(key=lambda d: (d.path, d.kind))

    return out