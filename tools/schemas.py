"""
Typed requests and results for the browser tools.

Requests are what the model asks for. A ToolResult is what the runtime
reports back: whether it worked, a typed error if not, and a ToolEffect that
says what actually happened in the browser.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class SideEffectClass(str, Enum):
    NONE = "none"
    NAVIGATION = "navigation"
    MUTATING = "mutating"


class ToolErrorType(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    STALE_OBSERVATION = "STALE_OBSERVATION"
    ELEMENT_NOT_ACTIONABLE = "ELEMENT_NOT_ACTIONABLE"
    NAVIGATION_ERROR = "NAVIGATION_ERROR"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"


class NavigateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["navigate"]
    url: str


class GetPageStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["get_page_state"]


class ClickRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["click"]
    element_id: str
    observation_id: str


class TypeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["type"]
    element_id: str
    text: str
    observation_id: str


class PressRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["press"]
    key: str
    observation_id: str


class GoBackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["go_back"]
    observation_id: str


ToolRequest = Annotated[
    Union[
        NavigateRequest,
        GetPageStateRequest,
        ClickRequest,
        TypeRequest,
        PressRequest,
        GoBackRequest,
    ],
    Field(discriminator="tool"),
]


class ToolEffect(BaseModel):
    """What the browser actually did, as observed after the action."""

    model_config = ConfigDict(extra="forbid")

    before_url: str
    after_url: str

    url_changed: bool
    dom_changed: bool

    dialog_appeared: bool
    dialog_messages: list[str] = Field(default_factory=list)

    # HTTP status of the page the action navigated to (None if it did not
    # navigate). A 4xx/5xx here is how an injected server fault shows up.
    http_status: int | None = None

    # Requests the browser refused because the origin is not allowed.
    blocked_requests: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool

    error_type: ToolErrorType | None = None
    message: str | None = None

    effect: ToolEffect

    data: Any | None = None


TOOL_SIDE_EFFECTS: dict[str, SideEffectClass] = {
    "navigate": SideEffectClass.NAVIGATION,
    "get_page_state": SideEffectClass.NONE,
    "click": SideEffectClass.MUTATING,
    "type": SideEffectClass.MUTATING,
    "press": SideEffectClass.MUTATING,
    "go_back": SideEffectClass.NAVIGATION,
}