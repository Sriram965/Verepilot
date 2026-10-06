from __future__ import annotations

import json

from observation.models import Observation


def format_observation_message(
    observation: Observation,
) -> str:
    """
    Convert the structured browser observation into the
    raw message representation used by Baseline A.

    Baseline A does not store this separately as structured
    state. The observation is inserted into message history.
    """

    payload = {
        "observation_id": (
            observation.observation_id
        ),
        "page_version": (
            observation.page_version
        ),
        "url": observation.url,
        "title": observation.title,
        "visible_text": (
            observation.visible_text
        ),
        "accessibility_tree": (
            observation.accessibility_tree
        ),
        "interactive_elements": [
            element.model_dump(
                mode="json"
            )
            for element
            in observation.interactive_elements
        ],
        "previous_action_effect": (
            observation.previous_action_effect
        ),
        "truncation": (
            observation.truncation.model_dump(
                mode="json"
            )
        ),
    }

    return (
        "CURRENT BROWSER OBSERVATION\n"
        "===========================\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
