"""
State-access interface for VeriPilot verification.

A probe provides read-only access to the state a verifier is allowed to
inspect. It answers "what is the relevant state?" and nothing else:

    probe:            "What is the state?"
    check:            "Does that state satisfy this condition?"   (checks.py)
    verifier/oracle:  "Does the required set of conditions pass?"

Two kinds of probe exist, and the difference between them is the core of the
experimental design:

- SESSION probes (in-loop verifier) see only what the agent/user could
  legitimately observe: pages they could open, the URL they are on. They
  never touch authoritative server state.
- ORACLE probes (evaluation only) read an authoritative snapshot taken
  through `environments.base.take_snapshot`.

Probes are read-only. They must not change the environment, must not consume
fault-injection occurrences, and must not alter the agent's own client or
browser state.

Capabilities
------------
Not every probe can answer every question. Each probe declares the leaf check
types it supports in `supported_checks`. Use `assert_supports` BEFORE a run so
an unsupported check fails at setup, not halfway through an experiment.

Site-specific probes live next to their site
(for example environments/sites/shop/probes.py). This module stays generic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verifier.spec import CheckSpec, iter_leaf_checks


class ProbeError(RuntimeError):
    """The probe could not obtain evidence (page missing, HTTP error...)."""


class ProbeCapabilityError(ValueError):
    """A check needs something this probe cannot observe."""


@runtime_checkable
class Probe(Protocol):
    supported_checks: frozenset[str]

    def current_url(self) -> str:
        ...

    def cart(self) -> dict[str, int]:
        ...

    def element_text(self, selector: str) -> str | None:
        ...

    def submitted_form(self) -> dict[str, str] | None:
        ...


def unsupported_checks(probe: Probe, spec: CheckSpec) -> list[str]:
    """Leaf check types in `spec` that `probe` cannot evaluate."""

    missing: list[str] = []

    for leaf in iter_leaf_checks(spec):
        if (
            leaf.type not in probe.supported_checks
            and leaf.type not in missing
        ):
            missing.append(leaf.type)

    return missing


def assert_supports(probe: Probe, spec: CheckSpec) -> None:
    """Raise ProbeCapabilityError if `probe` cannot evaluate all of `spec`."""

    missing = unsupported_checks(probe, spec)

    if missing:
        raise ProbeCapabilityError(
            f"{type(probe).__name__} cannot evaluate check type(s): "
            f"{', '.join(missing)}"
        )