"""
The tool schemas shown to the model.

These are generated from the same request models the runtime uses
(tools/schemas.py and agent/tool_calls.py), so what the model is told and
what the code accepts cannot drift apart. Only the descriptions are written
here.

The descriptions state the tool contract: every observation-based action
must use ids from the LATEST observation, and older ids are rejected as
stale. This is a fact about the tools, not a hint about the task, so every
system under comparison sees the same text.
"""

from __future__ import annotations

from typing import Any

from agent.tool_calls import AGENT_REQUEST_MODELS

LATEST = "from the LATEST observation (older ones are rejected as stale)"

TOOL_DESCRIPTIONS: dict[str, str] = {
    "navigate": (
        "Navigate the browser to a URL. Only pages of the benchmark site "
        "are allowed; anything else is blocked."
    ),
    "get_page_state": (
        "Capture the current browser state again. You are given a fresh "
        "observation after every action, so you rarely need this."
    ),
    "click": f"Click an interactive element. Use ids {LATEST}.",
    "type": (
        "Replace the contents of a text field with the given text. It only "
        "types: to submit a form, click its button or press Enter. "
        f"Use ids {LATEST}."
    ),
    "press": (
        "Press an allowed keyboard key or key combination, for example "
        f"Enter or Control+KeyA. Use the observation_id {LATEST}."
    ),
    "go_back": (
        f"Go back to the previous page. Use the observation_id {LATEST}."
    ),
    "finish": "Declare that the task has been completed.",
    "give_up": "Declare that the task cannot be completed.",
}

FIELD_DESCRIPTIONS: dict[str, str] = {
    "url": "Destination URL.",
    "element_id": "ID of an interactive element, such as el_003.",
    "observation_id": "observation_id of the latest observation.",
    "text": "The text to enter.",
    "key": "Key or combination, such as Enter, Tab or Control+KeyA.",
    "claim": "Concise statement of why you believe the task is complete.",
    "reason": "Why the task cannot be completed.",
}


def _function_definition(name: str, model: Any) -> dict[str, Any]:
    schema = model.model_json_schema()

    properties = {}

    for field, spec in schema.get("properties", {}).items():
        if field == "tool":  # the tool name is the function name
            continue

        spec = {key: value for key, value in spec.items() if key != "title"}
        spec["description"] = FIELD_DESCRIPTIONS[field]
        properties[field] = spec

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [
                    field
                    for field in schema.get("required", [])
                    if field != "tool"
                ],
                "additionalProperties": False,
            },
        },
    }


LLM_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _function_definition(name, model)
    for name, model in AGENT_REQUEST_MODELS.items()
]