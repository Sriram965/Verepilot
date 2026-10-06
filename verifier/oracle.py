from __future__ import annotations

from pydantic import BaseModel

from benchmark.schema import (
    OracleConfig,
    TaskSpec,
)
from verifier.checks import (
    CheckContext,
    CheckResult,
    evaluate_check,
)
from verifier.probe import HiddenShopProbe


class OracleResult(BaseModel):
    passed: bool

    success_check: CheckResult

    collateral_passed: bool

    observed_cart: dict[str, int]

    expected_cart: dict[str, int]


def evaluate_oracle(
    task: TaskSpec,
    config: OracleConfig,
    probe: HiddenShopProbe,
) -> OracleResult:

    success_result = evaluate_check(
        task.success,
        CheckContext(probe=probe),
    )

    observed_cart = probe.cart()

    collateral_passed = True

    if config.no_collateral_changes:
        collateral_passed = (
            observed_cart
            == config.expected_cart
        )

    return OracleResult(
        passed=(
            success_result.passed
            and collateral_passed
        ),
        success_check=success_result,
        collateral_passed=(
            collateral_passed
        ),
        observed_cart=observed_cart,
        expected_cart=(
            config.expected_cart
        ),
    )
