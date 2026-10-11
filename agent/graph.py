"""
The Baseline A agent loop (LangGraph).

    observe  ->  decide  ->  act  ->  observe  ->  ...

observe  captures the browser state and shows it to the model.
decide   asks the model for its next action (exactly one tool call).
act      runs it: a browser tool through the runtime, or `finish` /
         `give_up`, which end the run.

Baseline A keeps only the raw message history. The model alone decides when
the task is done: `finish` is a CLAIM, and nothing here checks it.

Budget
------
`max_steps` is the number of BROWSER ACTIONS the agent may run. `finish` and
`give_up` are free, so an agent that spends its whole budget can still say it
is done. After the last browser action the loop still observes and decides;
the run ends as `budget_exhausted` only if the model asks for ANOTHER browser
action.

Bad model output does not end the run
-------------------------------------
A malformed tool call, a reply with no tool call, or several tool calls at
once is a protocol error: the model is told what was wrong and asked again.
Protocol errors do not use browser steps, but `max_protocol_errors`
consecutive ones end the run as `failed`, so a broken model cannot loop
forever.

Errors are results, not crashes
-------------------------------
An exception from the model API or the browser ends the run with status
`error` and the exception recorded, so one bad run never takes down an
experiment.

Statuses
--------
success_claimed   the model called `finish`
gave_up           the model called `give_up`
budget_exhausted  the model asked for a browser action beyond `max_steps`
failed            the run could not continue (browser observation failed,
                  too many protocol errors, unexpected model reply type)
error             an exception was raised (model API, browser)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.prompt_builder import build_observation_message
from agent.state import BaselineAState
from agent.tool_calls import (
    FinishRequest,
    GiveUpRequest,
    parse_agent_tool_call,
)
from agent.trace import JsonlTraceWriter
from benchmark.schema import AgentTaskView
from observation.models import Observation
from tools.runtime import BrowserToolRuntime
from tools.schemas import GetPageStateRequest, ToolResult

RunStatus = Literal[
    "running",
    "success_claimed",
    "gave_up",
    "budget_exhausted",
    "failed",
    "error",
]


@dataclass
class BaselineAContext:
    """
    Run-scoped dependencies and counters.

    Deliberately outside BaselineAState: they are harness bookkeeping, not
    state shown to the model. `task` is the agent's view of the task only;
    the success spec, the oracle's expectations and the injected faults never
    reach the agent.
    """

    task: AgentTaskView
    model: BaseChatModel
    browser_runtime: BrowserToolRuntime
    tracer: JsonlTraceWriter

    run_id: str

    max_protocol_errors: int = 3

    steps_used: int = 0
    llm_calls: int = 0
    status: RunStatus = "running"

    claim: str | None = None
    give_up_reason: str | None = None
    error: str | None = None

    protocol_errors: int = 0
    consecutive_protocol_errors: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls_without_usage: int = 0

    llm_seconds: float = 0.0
    browser_seconds: float = 0.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _browser_call(ctx: BaselineAContext, request: Any) -> ToolResult:
    started = time.perf_counter()

    try:
        return ctx.browser_runtime.execute(request)
    finally:
        ctx.browser_seconds += time.perf_counter() - started


def _crash(ctx: BaselineAContext, stage: str, exc: Exception) -> Command:
    """An exception ends the run as a recorded result, not a crash."""

    ctx.status = "error"
    ctx.error = f"{stage}: {type(exc).__name__}: {exc}"

    ctx.tracer.write(
        event="run_error",
        run_id=ctx.run_id,
        payload={"stage": stage, "error": ctx.error},
    )

    return Command(goto=END)


def _protocol_error(
    ctx: BaselineAContext,
    kind: str,
    detail: str,
    messages: list[Any],
) -> Command:
    """Tell the model what was wrong and ask again (up to a limit)."""

    ctx.protocol_errors += 1
    ctx.consecutive_protocol_errors += 1

    ctx.tracer.write(
        event="protocol_error",
        run_id=ctx.run_id,
        payload={
            "kind": kind,
            "detail": detail,
            "consecutive": ctx.consecutive_protocol_errors,
        },
    )

    if ctx.consecutive_protocol_errors >= ctx.max_protocol_errors:
        ctx.status = "failed"
        ctx.error = (
            f"{ctx.consecutive_protocol_errors} consecutive invalid model "
            f"responses (last: {kind})"
        )

        return Command(update={"messages": messages}, goto=END)

    return Command(update={"messages": messages}, goto="decide")


def _record_usage(ctx: BaselineAContext, response: Any) -> dict | None:
    usage = getattr(response, "usage_metadata", None)

    if not usage:
        ctx.llm_calls_without_usage += 1
        return None

    ctx.input_tokens += usage.get("input_tokens", 0)
    ctx.output_tokens += usage.get("output_tokens", 0)

    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------


def observe_node(
    state: BaselineAState,
    runtime: Runtime[BaselineAContext],
) -> Command:
    ctx = runtime.context

    try:
        result = _browser_call(ctx, GetPageStateRequest(tool="get_page_state"))
    except Exception as exc:
        return _crash(ctx, "browser", exc)

    # The observation itself is traced once, below.
    ctx.tracer.write(
        event="observe",
        run_id=ctx.run_id,
        payload=result.model_copy(update={"data": None}),
    )

    observation = result.data

    if not result.ok or not isinstance(observation, Observation):
        ctx.status = "failed"
        ctx.error = (
            f"observation failed: {result.error_type} {result.message}"
        )

        return Command(
            update={
                "messages": [
                    SystemMessage(
                        content=(
                            "BROWSER OBSERVATION FAILED\n"
                            f"error_type={result.error_type}\n"
                            f"message={result.message}"
                        )
                    )
                ]
            },
            goto=END,
        )

    ctx.tracer.write(
        event="observation_captured",
        run_id=ctx.run_id,
        payload=observation,
    )

    return Command(
        update={"messages": [build_observation_message(observation)]},
        goto="decide",
    )


def decide_node(
    state: BaselineAState,
    runtime: Runtime[BaselineAContext],
) -> Command:
    ctx = runtime.context

    ctx.llm_calls += 1
    started = time.perf_counter()

    try:
        response = ctx.model.invoke(state["messages"])
    except Exception as exc:
        ctx.llm_seconds += time.perf_counter() - started
        return _crash(ctx, "llm", exc)

    elapsed = time.perf_counter() - started
    ctx.llm_seconds += elapsed

    ctx.tracer.write(
        event="llm_decision",
        run_id=ctx.run_id,
        payload={
            "llm_call": ctx.llm_calls,
            "seconds": round(elapsed, 3),
            "usage": _record_usage(ctx, response),
            "response": response,
        },
    )

    if not isinstance(response, AIMessage):
        ctx.status = "failed"
        ctx.error = "the model returned an unexpected message type"

        return Command(goto=END)

    calls = response.tool_calls

    if len(calls) == 1:
        return Command(update={"messages": [response]}, goto="act")

    if not calls:
        return _protocol_error(
            ctx,
            "no_tool_call",
            "the reply had no tool call",
            [
                response,
                HumanMessage(
                    content=(
                        "Your reply had no tool call. Respond with exactly "
                        "one tool call."
                    )
                ),
            ],
        )

    # Every tool call needs a result, or the next request is invalid.
    return _protocol_error(
        ctx,
        "multiple_tool_calls",
        f"the reply had {len(calls)} tool calls",
        [
            response,
            *[
                ToolMessage(
                    content=(
                        "Exactly one tool call per turn is required. None "
                        "of your tool calls were executed."
                    ),
                    tool_call_id=call["id"],
                )
                for call in calls
            ],
        ],
    )


def act_node(
    state: BaselineAState,
    runtime: Runtime[BaselineAContext],
) -> Command:
    ctx = runtime.context

    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage) or len(
        last_message.tool_calls
    ) != 1:
        ctx.status = "failed"
        ctx.error = "act was reached without exactly one tool call"

        return Command(goto=END)

    call = last_message.tool_calls[0]
    name, args, call_id = call["name"], call["args"], call["id"]

    try:
        request = parse_agent_tool_call(name, args)
    except Exception as exc:
        ctx.tracer.write(
            event="tool_call_validation_failed",
            run_id=ctx.run_id,
            payload={"tool": name, "args": args, "error": str(exc)},
        )

        return _protocol_error(
            ctx,
            "invalid_tool_call",
            str(exc),
            [
                ToolMessage(
                    content=f"Invalid tool call: {exc}",
                    tool_call_id=call_id,
                )
            ],
        )

    ctx.consecutive_protocol_errors = 0

    if isinstance(request, FinishRequest):
        ctx.status = "success_claimed"
        ctx.claim = request.claim

        ctx.tracer.write(
            event="finish_claimed",
            run_id=ctx.run_id,
            payload={"step": ctx.steps_used, "claim": request.claim},
        )

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            "finish accepted as the agent's completion "
                            f"claim.\nclaim: {request.claim}"
                        ),
                        tool_call_id=call_id,
                    )
                ]
            },
            goto=END,
        )

    if isinstance(request, GiveUpRequest):
        ctx.status = "gave_up"
        ctx.give_up_reason = request.reason

        ctx.tracer.write(
            event="give_up",
            run_id=ctx.run_id,
            payload={"step": ctx.steps_used, "reason": request.reason},
        )

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"give_up accepted.\nreason: {request.reason}",
                        tool_call_id=call_id,
                    )
                ]
            },
            goto=END,
        )

    # A browser action: this is what the budget counts.
    if ctx.steps_used >= ctx.task.max_steps:
        ctx.status = "budget_exhausted"

        ctx.tracer.write(
            event="budget_exhausted",
            run_id=ctx.run_id,
            payload={"steps_used": ctx.steps_used, "requested_tool": name},
        )

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            "Execution budget exhausted before this action "
                            "could execute."
                        ),
                        tool_call_id=call_id,
                    )
                ]
            },
            goto=END,
        )

    ctx.steps_used += 1

    ctx.tracer.write(
        event="action_selected",
        run_id=ctx.run_id,
        payload={"step": ctx.steps_used, "tool": name, "args": args},
    )

    started = time.perf_counter()

    try:
        result = _browser_call(ctx, request)
    except Exception as exc:
        return _crash(ctx, "browser", exc)

    ctx.tracer.write(
        event="tool_result",
        run_id=ctx.run_id,
        payload={
            "step": ctx.steps_used,
            "tool": name,
            "seconds": round(time.perf_counter() - started, 3),
            "result": result,
        },
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=result.model_dump_json(exclude_none=True),
                    tool_call_id=call_id,
                )
            ]
        },
        goto="observe",
    )


def build_baseline_a_graph():
    builder = StateGraph(
        BaselineAState,
        context_schema=BaselineAContext,
    )

    builder.add_node("observe", observe_node)
    builder.add_node("decide", decide_node)
    builder.add_node("act", act_node)

    # Every node routes itself with Command(goto=...).
    builder.add_edge(START, "observe")

    return builder.compile()