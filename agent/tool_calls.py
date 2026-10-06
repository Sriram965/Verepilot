from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from tools.schemas import (
    ClickRequest,
    GetPageStateRequest,
    GoBackRequest,
    NavigateRequest,
    PressRequest,
    ToolRequest,
    TypeRequest,
)


class FinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = "finish"
    claim: str


class GiveUpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = "give_up"
    reason: str


AgentToolRequest = (
    ToolRequest
    | FinishRequest
    | GiveUpRequest
)


BROWSER_REQUEST_MODELS: dict[
    str,
    type[BaseModel],
] = {
    "navigate": NavigateRequest,
    "get_page_state": GetPageStateRequest,
    "click": ClickRequest,
    "type": TypeRequest,
    "press": PressRequest,
    "go_back": GoBackRequest,
}


def parse_browser_tool_call(
    name: str,
    args: dict[str, Any],
) -> ToolRequest:

    model = BROWSER_REQUEST_MODELS.get(
        name
    )

    if model is None:
        raise ValueError(
            f"'{name}' is not a browser tool."
        )

    payload = {
        "tool": name,
        **args,
    }

    return model.model_validate(
        payload
    )


def parse_agent_tool_call(
    name: str,
    args: dict[str, Any],
) -> AgentToolRequest:

    if name == "finish":
        return FinishRequest.model_validate(
            {
                "tool": name,
                **args,
            }
        )

    if name == "give_up":
        return GiveUpRequest.model_validate(
            {
                "tool": name,
                **args,
            }
        )

    return parse_browser_tool_call(
        name,
        args,
    )
