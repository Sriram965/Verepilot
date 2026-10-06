from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InteractiveElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_id: str

    # Runtime grounding information.
    #
    # This identifies the exact DOM element represented
    # by element_id for this observation.
    selector: str

    tag: str
    role: str | None

    name: str | None

    field_name: str | None
    input_type: str | None

    value: str | None
    placeholder: str | None
    href: str | None

    checked: bool | None
    selected: bool | None

    visible: bool
    enabled: bool


class TruncationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visible_text_truncated: bool
    accessibility_tree_truncated: bool
    interactive_elements_truncated: bool

    original_interactive_element_count: int


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    page_version: str

    url: str
    title: str

    visible_text: str
    accessibility_tree: str

    interactive_elements: list[
        InteractiveElement
    ] = Field(default_factory=list)

    previous_action_effect: dict[str, Any] | None = None

    truncation: TruncationInfo
