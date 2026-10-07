"""
Deterministic, event-triggered fault injection for VeriPilot environments.

Faults are keyed to environment EVENTS, never wall-clock time, so the same
experiment reproduces exactly.

An event is the string "<METHOD> <path>", for example:

    "POST /cart/add"
    "GET /products"

A FaultRule fires on the Nth occurrence of its event (default: the first).
The environment calls `FaultInjector.check(event)` once per request.

Fault modes
-----------
fail_before_commit
    The request is rejected before the server changes any state.
    Retrying is safe. This is the "ordinary" failure.

fail_after_commit
    The server APPLIES the change, then returns an error (or responds only
    after `delay_seconds`, long enough for the client to time out). The
    caller cannot tell from the response that the mutation happened.
    This is the uncertain-outcome case behind the verify-before-retry rule:
    a blind retry here creates a duplicate. It is what makes that rule
    testable at all.

Probe bypass
------------
Verifier probes read the environment through the same HTTP surface as the
agent. They must not consume fault occurrences or trigger faults, or
measuring the environment would change it. `check(..., bypass=True)` neither
counts the event nor returns a rule.

Scope
-----
This module covers HTTP-level faults only. DOM-level faults (overlays,
delayed elements, renamed attributes, no-op clicks) will be added later as
additional rule types next to `HttpFault`, selected by the `type` field.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FaultMode = Literal["fail_before_commit", "fail_after_commit"]

_EVENT_PATTERN = re.compile(
    r"^(GET|POST|PUT|PATCH|DELETE) /\S*$"
)


class HttpFault(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["http_error"] = "http_error"

    event: str
    occurrence: int = Field(default=1, ge=1)
    mode: FaultMode = "fail_before_commit"

    response_status: int = Field(default=500, ge=400, le=599)
    response_body: str = "Injected deterministic fault"

    # Wait this long before responding. Use with fail_after_commit and a
    # value above the client timeout to simulate a timeout whose outcome is
    # unknown to the caller.
    delay_seconds: float = Field(default=0.0, ge=0.0, le=30.0)

    @field_validator("event")
    @classmethod
    def _event_format(cls, value: str) -> str:
        if not _EVENT_PATTERN.match(value):
            raise ValueError(
                "event must look like 'POST /cart/add' "
                "(METHOD, space, absolute path, no query string)"
            )
        return value


# Alias kept so existing imports (benchmark/schema.py) keep working.
# When DOM-level faults are added this becomes a discriminated union.
FaultRule = HttpFault


class FiredFault(BaseModel):
    model_config = ConfigDict(frozen=True)

    event: str
    occurrence: int
    mode: FaultMode


class FaultInjector:
    """Thread-safe, per-environment fault state."""

    def __init__(self, rules: Sequence[FaultRule] = ()) -> None:
        seen: set[tuple[str, int]] = set()

        for rule in rules:
            key = (rule.event, rule.occurrence)
            if key in seen:
                raise ValueError(
                    f"duplicate fault rule for {rule.event!r} "
                    f"occurrence {rule.occurrence}"
                )
            seen.add(key)

        self.rules: list[FaultRule] = list(rules)
        self._counts: dict[str, int] = {}
        self._fired: list[FiredFault] = []
        self._lock = threading.Lock()

    def check(
        self,
        event: str,
        *,
        bypass: bool = False,
    ) -> FaultRule | None:
        """
        Register one occurrence of `event`; return the rule that fires
        for it, if any. Probe traffic passes bypass=True and is ignored.  
        """

        if bypass:
            return None

        with self._lock:
            occurrence = self._counts.get(event, 0) + 1
            self._counts[event] = occurrence

            for rule in self.rules:
                if (
                    rule.event == event
                    and rule.occurrence == occurrence
                ):
                    self._fired.append(
                        FiredFault(
                            event=event,
                            occurrence=occurrence,
                            mode=rule.mode,
                        )
                    )
                    return rule

        return None

    def count(self, event: str) -> int:
        with self._lock:
            return self._counts.get(event, 0)

    @property
    def fired(self) -> list[FiredFault]:
        """Faults that actually triggered (needed to score recovery)."""
        with self._lock:
            return list(self._fired)

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._fired.clear()

    # Deprecated alias; removed when shop/server.py is rewritten (step 3).
    maybe_inject = check