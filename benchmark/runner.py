from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml

from benchmark.schema import TaskSpec
from environments.client import ShopClient
from environments.faults import FaultInjector
from environments.sites.shop import ShopEnvironment
from verifier.checks import (
    CheckContext,
    CheckResult,
    evaluate_check,
)
from verifier.oracle import (
    OracleResult,
    evaluate_oracle,
)
from verifier.probe import (
    HiddenShopProbe,
    ShopSessionProbe,
)


ScriptedSolution = Callable[
    [ShopClient, str],
    None,
]


def load_task(
    path: str | Path,
) -> TaskSpec:

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:

        data = yaml.safe_load(handle)

    return TaskSpec.model_validate(
        data
    )


def run_scripted_task(
    task: TaskSpec,
    solution: ScriptedSolution,
) -> tuple[
    CheckResult,
    OracleResult,
]:

    with ShopEnvironment(
        fault_injector=FaultInjector(
            rules=task.fault_config
        )
    ) as environment:

        client = ShopClient(
            environment.base_url
        )

        start_url = (
            task.start_url
            if task.start_url.startswith(
                "http"
            )
            else (
                environment.base_url
                + task.start_url
            )
        )

        status, _ = client.get(
            task.start_url
        )

        if status != 200:
            raise RuntimeError(
                "failed to open task "
                f"start URL: HTTP {status}"
            )

        solution(
            client,
            start_url,
        )

        session_result = evaluate_check(
            task.success,
            CheckContext(
                probe=ShopSessionProbe(
                    client
                )
            ),
        )

        oracle_probe = HiddenShopProbe(
            environment.state,
            client.session_id(),
            client.last_url,
        )

        oracle_result = evaluate_oracle(
            task,
            task.oracle,
            oracle_probe,
        )

        return (
            session_result,
            oracle_result,
        )
