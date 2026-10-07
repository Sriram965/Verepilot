"""
Deterministic check evaluation for VeriPilot.

`evaluate_check(spec, context)` turns a declarative spec (verifier/spec.py)
plus a probe (verifier/probe.py) into a structured, inspectable CheckResult.
No LLM is involved anywhere in this module.

Result semantics
----------------
passed    True only if the condition was observed to hold.
error     Set when the evidence could not be obtained (page missing, HTTP
          error). The check is then NOT passed, and `error` tells a metrics
          or failure-analysis pass that this was "could not observe", not
          "observed and wrong".
children  Results of sub-checks for all / any / not, so the full evaluation
          tree is preserved as evidence.

Rules worth knowing
-------------------
- `all` and `any` evaluate EVERY child (no short-circuit), so the evidence
  is complete and the number of probe calls does not depend on results.
- `not` over a child that could not be evaluated does NOT pass. Otherwise an
  unreadable page would make "the cart must not contain X" pass by accident.
- A probe that lacks the capability a check needs raises
  ProbeCapabilityError. That is a task/setup mistake, not a task failure, so
  it is never converted into a failed check. Use `assert_supports` before a
  run to catch it early.
- Element text is compared after trimming and collapsing whitespace, since
  rendered HTML does not preserve source whitespace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from verifier.probe import Probe, ProbeCapabilityError, ProbeError
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
    expected: object = None
    observed: object = None
    error: str | None = None
    children: list["CheckResult"] = Field(default_factory=list)


CheckResult.model_rebuild()


@dataclass(frozen=True)
class CheckContext:
    probe: Probe


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _require_capability(probe: Probe, spec: CheckSpec) -> None:
    supported = getattr(probe, "supported_checks", None)

    if supported is not None and spec.type not in supported:
        raise ProbeCapabilityError(
            f"{type(probe).__name__} cannot evaluate '{spec.type}'"
        )


def _unobservable(check: str, expected: object, error: str) -> CheckResult:
    return CheckResult(
        passed=False,
        check=check,
        expected=expected,
        observed=None,
        error=error,
    )


def _evaluate_leaf(spec: CheckSpec, probe: Probe) -> CheckResult:
    if isinstance(spec, CartContains):
        expected = {
            "product_id": spec.product_id,
            "quantity": spec.quantity,
            "comparison": spec.comparison,
        }

        try:
            quantity = probe.cart().get(spec.product_id, 0)
        except ProbeError as exc:
            return _unobservable("CartContains", expected, str(exc))

        passed = (
            quantity == spec.quantity
            if spec.comparison == "exactly"
            else quantity >= spec.quantity
        )

        return CheckResult(
            passed=passed,
            check="CartContains",
            expected=expected,
            observed={"product_id": spec.product_id, "quantity": quantity},
        )

    if isinstance(spec, UrlMatches):
        try:
            url = probe.current_url()
        except ProbeError as exc:
            return _unobservable("UrlMatches", spec.pattern, str(exc))

        return CheckResult(
            passed=re.search(spec.pattern, url) is not None,
            check="UrlMatches",
            expected=spec.pattern,
            observed=url,
        )

    if isinstance(spec, ElementText):
        expected = {"selector": spec.selector, "text": spec.expected_text}

        try:
            text = probe.element_text(spec.selector)
        except ProbeError as exc:
            return _unobservable("ElementText", expected, str(exc))

        return CheckResult(
            passed=(
                text is not None
                and _normalize(text) == _normalize(spec.expected_text)
            ),
            check="ElementText",
            expected=expected,
            observed={"selector": spec.selector, "text": text},
        )

    if isinstance(spec, FormSubmitted):
        expected = dict(spec.fields)

        try:
            submitted = probe.submitted_form()
        except ProbeError as exc:
            return _unobservable("FormSubmitted", expected, str(exc))

        return CheckResult(
            passed=(
                submitted is not None
                and all(
                    submitted.get(name) == value
                    for name, value in spec.fields.items()
                )
            ),
            check="FormSubmitted",
            expected=expected,
            observed=submitted,
        )

    raise TypeError(f"Unsupported check type: {type(spec)!r}")


def evaluate_check(spec: CheckSpec, context: CheckContext) -> CheckResult:
    if isinstance(spec, (AllCheck, AnyCheck)):
        children = [evaluate_check(child, context) for child in spec.checks]
        passes = sum(child.passed for child in children)

        if isinstance(spec, AllCheck):
            name, expected = "All", "all checks pass"
            passed = passes == len(children)
        else:
            name, expected = "Any", "at least one check passes"
            passed = passes >= 1

        return CheckResult(
            passed=passed,
            check=name,
            expected=expected,
            observed={"passed": passes, "total": len(children)},
            children=children,
        )

    if isinstance(spec, NotCheck):
        child = evaluate_check(spec.check, context)

        if _has_error(child):
            return CheckResult(
                passed=False,
                check="Not",
                expected="child check fails",
                observed=None,
                error="child check could not be evaluated",
                children=[child],
            )

        return CheckResult(
            passed=not child.passed,
            check="Not",
            expected="child check fails",
            observed={"child_passed": child.passed},
            children=[child],
        )

    _require_capability(context.probe, spec)

    return _evaluate_leaf(spec, context.probe)


def _has_error(result: CheckResult) -> bool:
    return result.error is not None or any(
        _has_error(child) for child in result.children
    )


def failure_evidence(result: CheckResult) -> list[CheckResult]:
    """
    The minimal set of results that explain why `result` did not pass.

    - a failed leaf explains itself;
    - a failed `all` / `any` is explained by its failed children;
    - a failed `not` is explained by itself (its child passed).

    This is the "structured failed-check evidence" the agent can be shown
    instead of a bare boolean.
    """

    if result.passed:
        return []

    if result.check in {"All", "Any"}:
        evidence: list[CheckResult] = []

        for child in result.children:
            evidence.extend(failure_evidence(child))

        return evidence or [result]

    return [result]