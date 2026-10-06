from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class BaselineAState(TypedDict):
    """
    Baseline A deliberately keeps only raw message history.

    No structured task state, observation state, failure
    state, evidence, or recovery state belongs here.

    Those are introduced in later experimental systems.
    """

    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]
