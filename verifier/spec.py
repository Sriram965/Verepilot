"""
Declarative success-condition schema for VeriPilot tasks.

This module defines the vocabulary used to say what it means for a task to
succeed. A task selects and composes these checks into its success spec; the
verifier does not run every check type automatically.

The SAME spec is evaluated against two different state sources:

- the in-loop verifier evaluates it against state the agent/user could
  legitimately observe (parsed pages, session-visible data);
- the evaluation oracle evaluates it against authoritative hidden state.

Which source answers is decided by the *probe* (verifier/probe.py), never by
the spec. Specs describe the requested goal only. They do not execute browser
actions, enforce permissions, classify failures, or perform recovery.

Leaf checks
-----------
cart_contains   product_id, quantity, comparison ("exactly" | "at_least")
url_matches     pattern (regular expression, searched in the current URL)
element_text    selector, expected_text
form_submitted  fields (every listed field must have the given value)

Composition
-----------
all / any / not

Design notes
------------
- Models are immutable and reject unknown fields, so a typo in a task file
  fails when the task is loaded, not mid-experiment.
- Regular expressions are compiled at load time for the same reason.
- `cart_contains` defaults to "exactly": a duplicated mutation (quantity 2
  when 1 was requested) FAILS the check. Use "at_least" when a task should
  tolerate extra quantity; the oracle's expected-state comparison still
  catches the surplus as collateral damage.
- Use `iter_leaf_checks` to find which leaf types a spec needs, so a runner 
  can verify that a probe supports all of them before the run starts.

Keep this module generic and independent of any particular environment.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
)

_STRICT = ConfigDict(extra="forbid", frozen=True)


class CartContains(BaseModel):
    model_config = _STRICT

    type: Literal["cart_contains"]
    product_id: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    comparison: Literal["exactly", "at_least"] = "exactly"


class UrlMatches(BaseModel):
    model_config = _STRICT

    type: Literal["url_matches"]
    pattern: str = Field(min_length=1)

    @field_validator("pattern")
    @classmethod
    def _pattern_compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}")
        return value


class ElementText(BaseModel):
    model_config = _STRICT

    type: Literal["element_text"]
    selector: str = Field(min_length=1)
    expected_text: str


class FormSubmitted(BaseModel):
    model_config = _STRICT

    type: Literal["form_submitted"]
    fields: dict[str, str] = Field(min_length=1)

    @field_validator("fields")
    @classmethod
    def _field_names_non_empty(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        if any(not name for name in value):
            raise ValueError("field names must be non-empty")
        return value


class AllCheck(BaseModel):
    model_config = _STRICT

    type: Literal["all"]
    checks: list["CheckSpec"] = Field(min_length=1)


class AnyCheck(BaseModel):
    model_config = _STRICT

    type: Literal["any"]
    checks: list["CheckSpec"] = Field(min_length=1)


class NotCheck(BaseModel):
    model_config = _STRICT

    type: Literal["not"]
    check: "CheckSpec"


CheckSpec = Annotated[
    Union[
        CartContains,
        UrlMatches,
        ElementText,
        FormSubmitted,
        AllCheck,
        AnyCheck,
        NotCheck,
    ],
    Field(discriminator="type"),
]

for _model in (AllCheck, AnyCheck, NotCheck):
    _model.model_rebuild()

_ADAPTER: TypeAdapter = TypeAdapter(CheckSpec)

LeafCheck = Union[CartContains, UrlMatches, ElementText, FormSubmitted]


def parse_check(data: object) -> CheckSpec:
    """Validate raw data (e.g. from YAML) into a CheckSpec."""
    return _ADAPTER.validate_python(data)


def iter_leaf_checks(spec: CheckSpec) -> Iterator[LeafCheck]:
    """Yield every leaf check in `spec`, depth-first, in order."""

    if isinstance(spec, (AllCheck, AnyCheck)):
        for child in spec.checks:
            yield from iter_leaf_checks(child)
    elif isinstance(spec, NotCheck):
        yield from iter_leaf_checks(spec.check)
    else:
        yield spec