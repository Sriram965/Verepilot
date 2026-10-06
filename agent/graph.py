from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.prompt_builder import (
    format_observation_message,
)
from agent.prompts.baseline_a import (
    BASELINE_A_SYSTEM_PROMPT,
)
from agent.state import BaselineAState
from agent.tool_calls import (
    FinishRequest,
    GiveUpRequest,
    parse_agent_tool_call,
)
from agent.trace import JsonlTraceWriter
from benchmark.schema import TaskSpec
from observation.models import Observation
from tools.runtime import BrowserToolRuntime
from tools.schemas import (
    GetPageStateRequest,
)


@dataclass
class BaselineAContext:
    """
    Run-scoped dependencies.

    These are deliberately outside BaselineAState.

    They are runtime context, not model-facing structured state.
    """

    task: TaskSpec
    model: BaseChatModel
    browser_runtime: BrowserToolRuntime
    tracer: JsonlTraceWriter

    run_id: str

    steps_used: int = 0
    llm_calls: int = 0
    status: Literal[
        "running",
        "success_claimed",
        "gave_up",
        "budget_exhausted",
        "failed",
    ] = "running"


def observe_node(
    state: BaselineAState,
    runtime: Runtime[BaselineAContext],
) -> dict:

    ctx = runtime.context

    result = ctx.browser_runtime.execute(
        GetPageStateRequest(
            tool="get_page_state"
        )
    )

    ctx.tracer.write(
        event="observe",
        run_id=ctx.run_id,
        payload=result,
    )

    if not result.ok:
        message = SystemMessage(
            content=(
                "BROWSER OBSERVATION FAILED\n"
                f"error_type={result.error_type}\n"
                f"message={result.message}"
            )
        )

        ctx.status = "failed"

        return {
            "messages": [message]
        }

    observation = result.data

    if not isinstance(
        observation,
        Observation,
    ):
        ctx.status = "failed"

        return {
            "messages": [
                SystemMessage(
                    content=(
                        "BROWSER OBSERVATION FAILED\n"
                        "runtime returned unexpected data."
                    )
                )
            ]
        }

    observation_message = (
        format_observation_message(
            observation
        )
    )

    ctx.tracer.write(
        event="observation_captured",
        run_id=ctx.run_id,
        payload=observation,
    )

    return {
        "messages": [
            SystemMessage(
                content=observation_message
            )
        ]
    }


def decide_node(
    state: BaselineAState,
    runtime: Runtime[BaselineAContext],
) -> Command:

    ctx = runtime.context

    ctx.llm_calls += 1

    response = ctx.model.invoke(
        state["messages"]
    )

    ctx.tracer.write(
        event="llm_decision",
        run_id=ctx.run_id,
        payload={
            "llm_call": ctx.llm_calls,
            "response": response,
        },
    )

    if not isinstance(
        response,
        AIMessage,
    ):
        ctx.status = "failed"

        return Command(
            update={
                "messages": [
                    SystemMessage(
                        content=(
                            "LLM returned an "
                            "unexpected message type."
                        )
                    )
                ]
            },
            goto=END,
        )

    if len(response.tool_calls) != 1:
        ctx.status = "failed"

        return Command(
            update={
                "messages": [
                    response,
                    SystemMessage(
                        content=(
                            "Baseline A requires "
                            "exactly one tool call "
                            "per decision."
                        )
                    ),
                ]
            },
            goto=END,
        )

    return Command(
        update={
            "messages": [response]
        },
        goto="act",
    )


def act_node(
    state: BaselineAState,
    runtime: Runtime[BaselineAContext],
) -> Command:

    ctx = runtime.context

    last_message = state["messages"][-1]

    if not isinstance(
        last_message,
        AIMessage,
    ):
        ctx.status = "failed"

        return Command(
            update={
                "messages": [
                    SystemMessage(
                        content=(
                            "ACT expected an "
                            "AIMessage."
                        )
                    )
                ]
            },
            goto=END,
        )

    tool_call = last_message.tool_calls[0]

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_call_id = tool_call["id"]

    # Every model-selected action consumes one step.
    if (
        ctx.steps_used
        >= ctx.task.max_steps
    ):
        ctx.status = "budget_exhausted"

        tool_message = ToolMessage(
            content=(
                "Execution budget exhausted "
                "before this action could execute."
            ),
            tool_call_id=tool_call_id,
        )

        return Command(
            update={
                "messages": [tool_message]
            },
            goto=END,
        )

    ctx.steps_used += 1

    ctx.tracer.write(
        event="action_selected",
        run_id=ctx.run_id,
        payload={
            "step": ctx.steps_used,
            "tool": tool_name,
            "args": tool_args,
        },
    )

    try:
        request = parse_agent_tool_call(
            tool_name,
            tool_args,
        )

    except Exception as exc:
        ctx.status = "failed"

        tool_message = ToolMessage(
            content=(
                "Invalid tool call: "
                f"{exc}"
            ),
            tool_call_id=tool_call_id,
        )

        ctx.tracer.write(
            event="tool_call_validation_failed",
            run_id=ctx.run_id,
            payload={
                "step": ctx.steps_used,
                "tool": tool_name,
                "args": tool_args,
                "error": str(exc),
            },
        )

        return Command(
            update={
                "messages": [tool_message]
            },
            goto=END,
        )

    if isinstance(
        request,
        FinishRequest,
    ):
        ctx.status = "success_claimed"

        tool_message = ToolMessage(
            content=(
                "finish accepted as the "
                "agent's completion claim.\n"
                f"claim: {request.claim}"
            ),
            tool_call_id=tool_call_id,
        )

        ctx.tracer.write(
            event="finish_claimed",
            run_id=ctx.run_id,
            payload={
                "step": ctx.steps_used,
                "claim": request.claim,
            },
        )

        return Command(
            update={
                "messages": [tool_message]
            },
            goto=END,
        )

    if isinstance(
        request,
        GiveUpRequest,
    ):
        ctx.status = "gave_up"

        tool_message = ToolMessage(
            content=(
                "give_up accepted.\n"
                f"reason: {request.reason}"
            ),
            tool_call_id=tool_call_id,
        )

        ctx.tracer.write(
            event="give_up",
            run_id=ctx.run_id,
            payload={
                "step": ctx.steps_used,
                "reason": request.reason,
            },
        )

        return Command(
            update={
                "messages": [tool_message]
            },
            goto=END,
        )

    result = ctx.browser_runtime.execute(
        request
    )

    ctx.tracer.write(
        event="tool_result",
        run_id=ctx.run_id,
        payload={
            "step": ctx.steps_used,
            "tool": tool_name,
            "result": result,
        },
    )

    tool_message = ToolMessage(
        content=result.model_dump_json(),
        tool_call_id=tool_call_id,
    )

    if (
        ctx.steps_used
        >= ctx.task.max_steps
    ):
        ctx.status = "budget_exhausted"

        return Command(
            update={
                "messages": [tool_message]
            },
            goto=END,
        )

    return Command(
        update={
            "messages": [tool_message]
        },
        goto="observe",
    )


def build_baseline_a_graph():
    builder = StateGraph(
        BaselineAState,
        context_schema=BaselineAContext,
    )

    builder.add_node(
        "observe",
        observe_node,
    )

    builder.add_node(
        "decide",
        decide_node,
    )

    builder.add_node(
        "act",
        act_node,
    )

    builder.add_edge(
        START,
        "observe",
    )

    builder.add_edge(
        "observe",
        "decide",
    )

    return builder.compile()
