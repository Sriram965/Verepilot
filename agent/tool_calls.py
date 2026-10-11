"""
Parsing the model's tool calls into typed requests.

The model calls tools by name with a dict of arguments. This module turns
that into one of the typed request models: the six browser tools from
tools/schemas.py, plus the two agent-control tools (`finish`, `give_up`).

A bad call (unknown tool, missing or extra argument) raises `ToolCallError`
with a short message written for the model to read, so the agent loop can
hand it back as the tool result instead of ending the run.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tools.schemas import (
    ClickRequest,
    GetPageStateRequest,
    GoBackRequest,
    NavigateRequest,
    PressRequest,
    ToolRequest,
    TypeRequest,
)


class ToolCallError(ValueError):
    """The model made a tool call that cannot be turned into a request."""


class FinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["finish"] = "finish"
    claim: str = Field(min_length=1)


class GiveUpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["give_up"] = "give_up"
    reason: str = Field(min_length=1)


AgentToolRequest = (
    ToolRequest
    | FinishRequest
    | GiveUpRequest
)

BROWSER_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "navigate": NavigateRequest,
    "get_page_state": GetPageStateRequest,
    "click": ClickRequest,
    "type": TypeRequest,
    "press": PressRequest,
    "go_back": GoBackRequest,
}

TERMINAL_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "finish": FinishRequest,
    "give_up": GiveUpRequest,
}

# Every tool the model may call, in the order it is shown to the model.
AGENT_REQUEST_MODELS: dict[str, type[BaseModel]] = {
    **BROWSER_REQUEST_MODELS,
    **TERMINAL_REQUEST_MODELS,
}


def _validate(
    name: str,
    model: type[BaseModel],
    args: dict[str, Any],
) -> Any:
    try:
        return model.model_validate({**args, "tool": name})
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: "
            f"{error['msg']}"
            for error in exc.errors()
            if error["loc"] != ("tool",)
        )

        raise ToolCallError(
            f"Invalid arguments for '{name}': {problems}."
        ) from exc


def parse_browser_tool_call(
    name: str,
    args: dict[str, Any],
) -> ToolRequest:
    model = BROWSER_REQUEST_MODELS.get(name)

    if model is None:
        raise ToolCallError(f"'{name}' is not a browser tool.")

    return _validate(name, model, args)


def parse_agent_tool_call(
    name: str,
    args: dict[str, Any],
) -> AgentToolRequest:
    model = TERMINAL_REQUEST_MODELS.get(name)

    if model is None:
        return parse_browser_tool_call(name, args)

    return _validate(name, model, args)