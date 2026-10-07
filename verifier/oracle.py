"""
Evaluation oracle for VeriPilot.

The oracle decides the authoritative benchmark outcome of a run. It is used
ONLY by the evaluation harness and must never be reachable from the agent or
the in-loop verifier.

It answers two independent questions, both against the environment's
AUTHORITATIVE state (an `environments.base` snapshot), never against what the
agent claimed or what a page displayed:

1. Goal      - is the task's success spec satisfied?
2. State     - does the authoritative state equal the state the task
               expects, exactly? Anything extra, missing, or different is
               reported path by path.

Outcomes
--------
success            goal achieved and state matches
collateral_damage  goal achieved, but state differs (extra cart items,
                   duplicate submissions, unrelated changes...)
failed             goal not achieved

"collateral_damage" is what lets a run look successful to the in-loop
verifier while being a failure here, which is the false-completion signal.

Expected state
--------------
`OracleConfig.expected_state` is the complete final state a correct run must
leave behind, in the same {session, shared} shape as a snapshot. Fields that
are legitimately volatile can be excluded with `ignore_paths` (JSON
Pointers, e.g. "/shared/inventory"). Set `no_collateral_changes: false` to
score the goal only.

This module is environment-agnostic: it works on any probe that exposes the
snapshot it was built from.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from environments.base import Snapshot, snapshot_hash
from verifier.checks import (
    CheckContext,
    CheckResult,
    evaluate_check,
    failure_evidence,
)
from verifier.probe import Probe, assert_supports
from verifier.spec import CheckSpec
from verifier.statediff import StateDiff, diff_state, is_valid_pointer


class OracleError(RuntimeError):
    """The oracle could not produce an authoritative verdict."""


class ExpectedState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session: dict[str, Any]
    shared: dict[str, Any] = Field(default_factory=dict)


class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    no_collateral_changes: bool = True
    expected_state: ExpectedState | None = None
    ignore_paths: list[str] = Field(default_factory=list)

    @field_validator("ignore_paths")
    @classmethod
    def _pointers_are_valid(cls, value: list[str]) -> list[str]:
        for path in value:
            if not is_valid_pointer(path):
                raise ValueError(
                    f"ignore path {path!r} must be a JSON Pointer "
                    "starting with '/'"
                )
        return value

    @model_validator(mode="after")
    def _expected_state_required(self) -> "OracleConfig":
        if self.no_collateral_changes and self.expected_state is None:
            raise ValueError(
                "expected_state is required when "
                "no_collateral_changes is true"
            )
        return self


@runtime_checkable
class OracleProbe(Probe, Protocol):
    """A probe backed by an authoritative snapshot."""

    snapshot: Snapshot


class OracleResult(BaseModel):
    passed: bool
    outcome: Literal["success", "collateral_damage", "failed"]

    goal_achieved: bool
    success_check: CheckResult
    failed_checks: list[CheckResult]

    state_checked: bool
    state_matches: bool
    state_diff: list[StateDiff]

    observed_state: dict[str, Any]
    state_hash: str


def _contains_error(result: CheckResult) -> bool:
    return result.error is not None or any(
        _contains_error(child) for child in result.children
    )


def evaluate_oracle(
    success: CheckSpec,
    config: OracleConfig,
    probe: OracleProbe,
) -> OracleResult:
    assert_supports(probe, success)

    success_check = evaluate_check(success, CheckContext(probe=probe))

    if _contains_error(success_check):
        raise OracleError(
            "authoritative evaluation hit an error: "
            f"{failure_evidence(success_check)}"
        )

    goal_achieved = success_check.passed
    observed = probe.snapshot

    if config.no_collateral_changes:
        assert config.expected_state is not None  # enforced by the config

        diff = diff_state(
            config.expected_state.model_dump(mode="json"),
            observed,
            config.ignore_paths,
        )
        state_matches = not diff
        state_checked = True
    else:
        diff = []
        state_matches = True
        state_checked = False

    if goal_achieved and state_matches:
        outcome = "success"
    elif goal_achieved:
        outcome = "collateral_damage"
    else:
        outcome = "failed"

    return OracleResult(
        passed=outcome == "success",
        outcome=outcome,
        goal_achieved=goal_achieved,
        success_check=success_check,
        failed_checks=failure_evidence(success_check),
        state_checked=state_checked,
        state_matches=state_matches,
        state_diff=diff,
        observed_state=dict(observed),
        state_hash=snapshot_hash(observed),
    )