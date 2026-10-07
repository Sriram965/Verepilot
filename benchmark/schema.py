"""
Benchmark task contract for VeriPilot.

A TaskSpec is the benchmark-level description of one task: what the agent is
told, which environment to run, the step budget, the declarative success
condition, optional fault injection, the oracle's expectations, and metadata.

Boundaries
----------
- A TaskSpec is an EVALUATION object. The agent must only ever receive
  `task.agent_view()`, which carries the instruction and constraints and
  nothing else. `success`, `oracle`, and `fault_config` are hidden from the
  agent: the success spec is the verifier's and the oracle's business, and
  the oracle's expected state and the injected faults are ground truth the
  agent must not see.
- This schema is environment-agnostic. Site-specific facts live in the
  environment, probes, and task data, never in this module.
- Behavior (running tasks, injecting faults, evaluating) belongs to other
  modules. This module only defines and validates the contract.

Metadata
--------
tags      free-form labels, e.g. "shopping", "requires_vision".
split     "dev" or "test" (prompts are tuned on dev only).
pair_id   links a clean task to its faulted twin. See `validate_suite`.

Suite rules (`validate_suite`)
------------------------------
- task ids are unique;
- every task with faults belongs to a pair;
- each pair is exactly one clean task and one faulted task that are
  identical in everything except `task_id`, `fault_config` and `tags`
  (same instruction, start page, budget, environment, success spec, oracle,
  and split), so recovery is measured against a true counterfactual.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from environments.faults import FaultRule
from verifier.oracle import ExpectedState, OracleConfig
from verifier.spec import CheckSpec, iter_leaf_checks

__all__ = [
    "AgentTaskView",
    "ExpectedState",
    "OracleConfig",
    "SuiteError",
    "TaskSpec",
    "required_checks",
    "validate_suite",
]

_SLUG = r"^[a-z0-9][a-z0-9_-]*$"


class AgentTaskView(BaseModel):
    """The only part of a task the agent is allowed to see."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    instruction: str
    start_url: str
    max_steps: int


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=_SLUG)
    instruction: str = Field(min_length=1)
    start_url: str
    max_steps: int = Field(gt=0)

    success: CheckSpec

    fault_config: list[FaultRule] = Field(default_factory=list)
    oracle: OracleConfig

    environment: str = Field(default="shop", pattern=_SLUG)
    tags: list[str] = Field(default_factory=list)
    split: Literal["dev", "test"] | None = None
    pair_id: str | None = Field(default=None, pattern=_SLUG)

    @field_validator("start_url")
    @classmethod
    def _start_url_is_site_relative(cls, value: str) -> str:
        if not value.startswith("/") or value.startswith("//"):
            raise ValueError(
                "start_url must be a site-relative path such as '/products'"
            )
        return value

    @field_validator("tags")
    @classmethod
    def _tags_are_clean(cls, value: list[str]) -> list[str]:
        import re

        for tag in value:
            if not re.match(_SLUG, tag):
                raise ValueError(f"invalid tag {tag!r}")

        if len(set(value)) != len(value):
            raise ValueError("tags must be unique")

        return value

    @property
    def is_faulted(self) -> bool:
        return bool(self.fault_config)

    @property
    def requires_vision(self) -> bool:
        return "requires_vision" in self.tags

    def agent_view(self) -> AgentTaskView:
        return AgentTaskView(
            task_id=self.task_id,
            instruction=self.instruction,
            start_url=self.start_url,
            max_steps=self.max_steps,
        )


def required_checks(task: TaskSpec) -> set[str]:
    """Leaf check types the task's success spec needs from a probe."""

    return {leaf.type for leaf in iter_leaf_checks(task.success)}


class SuiteError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def _pair_signature(task: TaskSpec) -> dict:
    return {
        "instruction": task.instruction,
        "start_url": task.start_url,
        "max_steps": task.max_steps,
        "environment": task.environment,
        "split": task.split,
        "success": task.success.model_dump(mode="json"),
        "oracle": task.oracle.model_dump(mode="json"),
    }


def validate_suite(tasks: Sequence[TaskSpec]) -> None:
    """Raise SuiteError listing every problem found across `tasks`."""

    problems: list[str] = []

    for task_id, count in Counter(t.task_id for t in tasks).items():
        if count > 1:
            problems.append(f"duplicate task_id {task_id!r}")

    pairs: dict[str, list[TaskSpec]] = {}

    for task in tasks:
        if task.is_faulted and task.pair_id is None:
            problems.append(
                f"{task.task_id}: faulted task must have a pair_id"
            )

        if task.pair_id is not None:
            pairs.setdefault(task.pair_id, []).append(task)

    for pair_id, members in sorted(pairs.items()):
        clean = [t for t in members if not t.is_faulted]
        faulted = [t for t in members if t.is_faulted]

        if len(clean) != 1 or len(faulted) != 1 or len(members) != 2:
            problems.append(
                f"pair {pair_id!r} must be exactly one clean and one "
                f"faulted task (found {len(clean)} clean, "
                f"{len(faulted)} faulted)"
            )
            continue

        a, b = _pair_signature(clean[0]), _pair_signature(faulted[0])

        for field_name in a:
            if a[field_name] != b[field_name]:
                problems.append(
                    f"pair {pair_id!r}: {field_name!r} differs between "
                    f"{clean[0].task_id} and {faulted[0].task_id}"
                )

    if problems:
        raise SuiteError(problems)