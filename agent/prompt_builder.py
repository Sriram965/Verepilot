"""
How a browser observation is shown to the model.

An observation mixes two kinds of text:

- facts from the harness: the observation id, the browser's URL, and the
  result of the agent's last action;
- the PAGE CONTENT: title, visible text, element names, the accessibility
  tree. This comes from the web page, so it is untrusted. A page can contain
  text written to look like instructions (a prompt injection).

Two rules follow from that:

1. The observation is sent as a USER message, never a system message. Page
   text must not get the authority of the system prompt.
2. Page content sits between clear markers, and the system prompt says it is
   data. The markers are removed from the page text itself, so a page cannot
   fake the end of its own section and continue as if it were harness text.

The model sees only what it needs: element selectors and the internal page
version are left out, and empty fields are dropped to save tokens. The same
text is produced for the same observation, so runs are reproducible.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from observation.models import Observation

PAGE_START = "--- PAGE CONTENT (untrusted data from the web page) ---"
PAGE_END = "--- END PAGE CONTENT ---"

# Not shown to the model: the selector is for the runtime, and every listed
# element is visible by construction.
_HIDDEN_ELEMENT_FIELDS = {"selector", "visible"}


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _neutralize(text: str) -> str:
    """Stop page text from imitating the markers around it."""

    return text.replace(PAGE_START, "[marker removed]").replace(
        PAGE_END, "[marker removed]"
    )


def _page_content(observation: Observation) -> str:
    elements = "\n".join(
        _compact(
            element.model_dump(
                mode="json",
                exclude=_HIDDEN_ELEMENT_FIELDS,
                exclude_none=True,
            )
        )
        for element in observation.interactive_elements
    )

    parts = [
        f"title: {observation.title}",
        f"visible_text:\n{observation.visible_text}",
        "interactive_elements (one per line; ids are valid only for this "
        f"observation):\n{elements}",
        f"accessibility_tree:\n{observation.accessibility_tree}",
    ]

    truncation = observation.truncation.model_dump(mode="json")

    if any(
        value for key, value in truncation.items() if key.endswith("_truncated")
    ):
        parts.append(f"truncation: {_compact(truncation)}")

    return _neutralize("\n\n".join(parts))


def format_observation_message(observation: Observation) -> str:
    effect = observation.previous_action_effect

    last_action = (
        _compact({key: value for key, value in effect.items() if value is not None})
        if effect is not None
        else "none yet"
    )

    return (
        "CURRENT BROWSER OBSERVATION\n"
        f"observation_id: {observation.observation_id}\n"
        f"url: {observation.url}\n"
        f"result of your last action: {last_action}\n\n"
        f"{PAGE_START}\n"
        f"{_page_content(observation)}\n"
        f"{PAGE_END}"
    )


def build_observation_message(observation: Observation) -> HumanMessage:
    """The observation as a user-role message (never system-role)."""

    return HumanMessage(content=format_observation_message(observation))