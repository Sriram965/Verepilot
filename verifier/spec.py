"""
Declarative success-condition schema for VeriPilot tasks.

This module defines the vocabulary used to express what it means for a
particular task to succeed. A task selects and composes these checks into its
SuccessSpec; the verifier does not automatically run every check type.

The same SuccessSpec can be evaluated against different state sources:
- the in-loop verifier evaluates it against state legitimately observable by
  the agent/user during execution;
- the evaluation oracle evaluates the same specification against authoritative
  real environment state.

The checks describe the task's requested goal only. They do not execute browser
actions, enforce tool permissions, classify failures, or perform recovery.

Supported checks may include URL conditions, element text, cart contents,
form submission, and logical composition such as All, Any, and Not.

Keep this module generic and independent of any particular environment.
Environment-specific state access belongs in probes, while actual check
evaluation belongs in verifier/checks.py.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class CartContains(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cart_contains"]
    product_id: str
    quantity: int = Field(ge=1)


class UrlMatches(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["url_matches"]
    pattern: str


class ElementText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["element_text"]
    selector: str
    expected_text: str


class FormSubmitted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["form_submitted"]
    fields: dict[str, str] = Field(min_length=1)


class AllCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["all"]
    checks: list["CheckSpec"] = Field(min_length=1)


class AnyCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["any"]
    checks: list["CheckSpec"] = Field(min_length=1)


class NotCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


for model in (AllCheck, AnyCheck, NotCheck):
    model.model_rebuild()
