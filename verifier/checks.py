from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from verifier.spec import (
    AllCheck,
    AnyCheck,
    CartContains,
    CheckSpec,
    ElementText,
    FormSubmitted,
    NotCheck,
    UrlMatches,
)


class CheckResult(BaseModel):
    passed: bool
    check: str
    expected: object
    observed: object


class Probe(Protocol):
    def current_url(self) -> str:
        ...

    def cart(self) -> dict[str, int]:
        ...

    def element_text(self, selector: str) -> str | None:
        ...

    def submitted_form(self) -> dict[str, str] | None:
        ...


@dataclass(frozen=True)
class CheckContext:
    probe: Probe


def evaluate_check(
    spec: CheckSpec,
    context: CheckContext,
) -> CheckResult:

    if isinstance(spec, CartContains):
        observed_cart = context.probe.cart()
        observed_quantity = observed_cart.get(
            spec.product_id,
            0,
        )

        return CheckResult(
            passed=(
                observed_quantity
                == spec.quantity
            ),
            check="CartContains",
            expected={
                "product_id": spec.product_id,
                "quantity": spec.quantity,
            },
            observed={
                "product_id": spec.product_id,
                "quantity": observed_quantity,
            },
        )

    if isinstance(spec, UrlMatches):
        observed_url = (
            context.probe.current_url()
        )

        return CheckResult(
            passed=(
                re.search(
                    spec.pattern,
                    observed_url,
                )
                is not None
            ),
            check="UrlMatches",
            expected=spec.pattern,
            observed=observed_url,
        )

    if isinstance(spec, ElementText):
        observed_text = (
            context.probe.element_text(
                spec.selector
            )
        )

        return CheckResult(
            passed=(
                observed_text
                == spec.expected_text
            ),
            check="ElementText",
            expected={
                "selector": spec.selector,
                "text": spec.expected_text,
            },
            observed={
                "selector": spec.selector,
                "text": observed_text,
            },
        )

    if isinstance(spec, FormSubmitted):
        observed_fields = (
            context.probe.submitted_form()
        )

        passed = (
            observed_fields is not None
            and all(
                observed_fields.get(
                    field
                )
                == expected_value
                for field, expected_value
                in spec.fields.items()
            )
        )

        return CheckResult(
            passed=passed,
            check="FormSubmitted",
            expected=dict(spec.fields),
            observed=observed_fields,
        )

    if isinstance(spec, AllCheck):
        results = [
            evaluate_check(
                child,
                context,
            )
            for child in spec.checks
        ]

        return CheckResult(
            passed=all(
                result.passed
                for result in results
            ),
            check="All",
            expected="all checks pass",
            observed=[
                result.model_dump(
                    mode="json"
                )
                for result in results
            ],
        )

    if isinstance(spec, AnyCheck):
        results = [
            evaluate_check(
                child,
                context,
            )
            for child in spec.checks
        ]

        return CheckResult(
            passed=any(
                result.passed
                for result in results
            ),
            check="Any",
            expected="at least one check passes",
            observed=[
                result.model_dump(
                    mode="json"
                )
                for result in results
            ],
        )

    if isinstance(spec, NotCheck):
        child_result = evaluate_check(
            spec.check,
            context,
        )

        return CheckResult(
            passed=not child_result.passed,
            check="Not",
            expected="child check fails",
            observed=child_result.model_dump(
                mode="json"
            ),
        )

    raise TypeError(
        f"Unsupported check type: {type(spec)!r}"
    )
