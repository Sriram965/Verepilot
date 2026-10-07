"""
Hermetic benchmark runner for VeriPilot (scripted solutions, no LLM).

`run_scripted_task` executes one task end to end:

    fresh environment -> open start page -> scripted solution
        -> in-loop verifier (observable state)
        -> evaluation oracle (authoritative state)
        -> one RunRecord

Each run gets its own environment on its own loopback port, so runs share
nothing. Probes and snapshots are built by the task's SiteAdapter
(benchmark/sites.py); this module has no site-specific code.

Verifier vs oracle
------------------
The record states how the two verdicts relate:

    agree                  both pass or both fail
    verifier_false_pass    verifier passed, oracle failed
                           (this is a false completion)
    verifier_false_fail    verifier failed, oracle passed

Harness errors
--------------
If the environment recorded an unexpected internal error, the run is not a
valid measurement and `HarnessError` is raised instead of returning a
record. Injected faults are not harness errors; they are recorded in
`faults_fired`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from benchmark.schema import TaskSpec, required_checks, validate_suite
from benchmark.sites import get_site
from environments.client import ShopClient
from environments.faults import FaultInjector, FiredFault
from verifier.checks import CheckContext, CheckResult, evaluate_check
from verifier.oracle import OracleResult, evaluate_oracle
from verifier.probe import ProbeCapabilityError

ScriptedSolution = Callable[[ShopClient, str], None]


class HarnessError(RuntimeError):
    """The environment hit an unexpected internal error during a run."""


class RunRecord(BaseModel):
    task_id: str
    environment: str
    split: str | None
    pair_id: str | None
    faulted: bool

    verifier: CheckResult
    oracle: OracleResult
    agreement: Literal[
        "agree", "verifier_false_pass", "verifier_false_fail"
    ]

    faults_fired: list[FiredFault]
    final_url: str
    wall_seconds: float

    @property
    def fault_triggered(self) -> bool:
        return bool(self.faults_fired)


def load_task(path: str | Path) -> TaskSpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        return TaskSpec.model_validate(yaml.safe_load(handle))


def load_suite(directory: str | Path) -> list[TaskSpec]:
    """Load every *.yaml task in `directory` and validate them as a suite."""

    tasks = [
        load_task(path) for path in sorted(Path(directory).glob("*.yaml"))
    ]
    validate_suite(tasks)

    return tasks


def preflight(task: TaskSpec) -> None:
    """
    Fail BEFORE any run if the task cannot be evaluated: unknown
    environment, or a success check that the in-loop probe or the oracle
    probe cannot observe.
    """

    site = get_site(task.environment)
    needed = required_checks(task)

    for label, supported in (
        ("session", site.session_supported),
        ("oracle", site.oracle_supported),
    ):
        missing = sorted(needed - supported)

        if missing:
            raise ProbeCapabilityError(
                f"{task.task_id}: the {label} probe for "
                f"{task.environment!r} cannot evaluate "
                f"{', '.join(missing)}"
            )


def preflight_suite(tasks: list[TaskSpec]) -> None:
    validate_suite(tasks)

    for task in tasks:
        preflight(task)


def run_scripted_task(
    task: TaskSpec,
    solution: ScriptedSolution,
) -> RunRecord:
    preflight(task)

    site = get_site(task.environment)
    injector = FaultInjector(rules=task.fault_config)
    started = time.perf_counter()

    with site.make_environment(injector) as environment:
        client = site.make_client(environment)

        start_url = (
            task.start_url
            if task.start_url.startswith("http")
            else environment.base_url + task.start_url
        )

        status, _ = client.get(task.start_url)

        if status != 200:
            raise RuntimeError(
                f"failed to open task start URL: HTTP {status}"
            )

        solution(client, start_url)

        final_url = site.current_url(client)

        verifier = evaluate_check(
            task.success,
            CheckContext(probe=site.session_probe(environment, client)),
        )

        oracle = evaluate_oracle(
            task.success,
            task.oracle,
            site.oracle_probe(environment, client),
        )

        internal_errors = list(getattr(environment, "internal_errors", []))

        if internal_errors:
            raise HarnessError(
                f"{task.task_id}: {'; '.join(internal_errors)}"
            )

    if verifier.passed == oracle.passed:
        agreement = "agree"
    elif verifier.passed:
        agreement = "verifier_false_pass"
    else:
        agreement = "verifier_false_fail"

    return RunRecord(
        task_id=task.task_id,
        environment=task.environment,
        split=task.split,
        pair_id=task.pair_id,
        faulted=task.is_faulted,
        verifier=verifier,
        oracle=oracle,
        agreement=agreement,
        faults_fired=injector.fired,
        final_url=final_url,
        wall_seconds=time.perf_counter() - started,
    )