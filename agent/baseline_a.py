"""
Runs Baseline A on one task: a plain LLM browser loop where the model alone
decides whether the task is complete.

`run_baseline_a` always returns a result. A model or browser exception, a
failed first page load, or a loop that ends without a verdict are all
recorded in `status` / `error`, and the trace always ends with `run_finished`.

The agent is only ever given the task's agent view (instruction, start page,
step budget). Pass a full TaskSpec and it is reduced to that view here, so
the oracle's expected state and the injected faults cannot reach the model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import BaselineAContext, build_baseline_a_graph
from agent.prompts.baseline_a import BASELINE_A_SYSTEM_PROMPT
from agent.tool_definitions import LLM_TOOL_DEFINITIONS
from agent.trace import JsonlTraceWriter
from benchmark.schema import AgentTaskView, TaskSpec
from tools.runtime import BrowserToolRuntime
from tools.schemas import NavigateRequest


@dataclass(frozen=True)
class BaselineARunResult:
    run_id: str
    status: str
    steps_used: int
    llm_calls: int
    final_messages: list

    claim: str | None = None
    give_up_reason: str | None = None
    error: str | None = None

    protocol_errors: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    # LLM calls whose response carried no token usage (the totals above
    # undercount by that many calls).
    llm_calls_without_usage: int = 0

    llm_seconds: float = 0.0
    browser_seconds: float = 0.0
    wall_seconds: float = 0.0

    @property
    def declared_success(self) -> bool:
        """The agent said it was done. This is a claim, not a result."""

        return self.status == "success_claimed"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def prompt_fingerprint() -> str:
    """Short hash of the system prompt and tool schemas, for the trace."""

    material = BASELINE_A_SYSTEM_PROMPT + json.dumps(
        LLM_TOOL_DEFINITIONS, sort_keys=True
    )

    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def run_baseline_a(
    *,
    task: TaskSpec | AgentTaskView,
    model: BaseChatModel,
    browser_runtime: BrowserToolRuntime,
    trace_path: str | Path,
    start_url: str,
    run_metadata: dict[str, Any] | None = None,
    max_protocol_errors: int = 3,
) -> BaselineARunResult:
    view = task.agent_view() if isinstance(task, TaskSpec) else task

    run_id = uuid4().hex
    tracer = JsonlTraceWriter(trace_path)

    context = BaselineAContext(
        task=view,
        model=model,
        browser_runtime=browser_runtime,
        tracer=tracer,
        run_id=run_id,
        max_protocol_errors=max_protocol_errors,
    )

    started = time.perf_counter()

    tracer.write(
        event="run_started",
        run_id=run_id,
        payload={
            "system": "baseline_a",
            "task_id": view.task_id,
            "start_url": start_url,
            "max_steps": view.max_steps,
            "prompt_sha256": prompt_fingerprint(),
            "metadata": run_metadata,
        },
    )

    final_messages: list = []

    try:
        nav_started = time.perf_counter()
        start_result = browser_runtime.execute(
            NavigateRequest(tool="navigate", url=start_url)
        )
        context.browser_seconds += time.perf_counter() - nav_started

        tracer.write(
            event="initial_navigation",
            run_id=run_id,
            payload=start_result,
        )

        if not start_result.ok:
            context.status = "failed"
            context.error = (
                "initial navigation failed: "
                f"{start_result.error_type} {start_result.message}"
            )
        else:
            result = build_baseline_a_graph().invoke(
                {
                    "messages": [
                        SystemMessage(content=BASELINE_A_SYSTEM_PROMPT),
                        HumanMessage(
                            content=(
                                "Complete this web task:\n\n"
                                f"{view.instruction}"
                            )
                        ),
                    ]
                },
                context=context,
                config={"recursion_limit": 10 * (view.max_steps + 2) + 50},
            )
            final_messages = result["messages"]

    except Exception as exc:
        context.status = "error"
        context.error = f"runner: {type(exc).__name__}: {exc}"

        tracer.write(
            event="run_error",
            run_id=run_id,
            payload={"stage": "runner", "error": context.error},
        )

    if context.status == "running":
        context.status = "failed"
        context.error = context.error or "the run ended without a result"

    wall = time.perf_counter() - started

    tracer.write(
        event="run_finished",
        run_id=run_id,
        payload={
            "status": context.status,
            "steps_used": context.steps_used,
            "llm_calls": context.llm_calls,
            "protocol_errors": context.protocol_errors,
            "input_tokens": context.input_tokens,
            "output_tokens": context.output_tokens,
            "llm_calls_without_usage": context.llm_calls_without_usage,
            "llm_seconds": round(context.llm_seconds, 3),
            "browser_seconds": round(context.browser_seconds, 3),
            "wall_seconds": round(wall, 3),
            "claim": context.claim,
            "give_up_reason": context.give_up_reason,
            "error": context.error,
        },
    )

    return BaselineARunResult(
        run_id=run_id,
        status=context.status,
        steps_used=context.steps_used,
        llm_calls=context.llm_calls,
        final_messages=final_messages,
        claim=context.claim,
        give_up_reason=context.give_up_reason,
        error=context.error,
        protocol_errors=context.protocol_errors,
        input_tokens=context.input_tokens,
        output_tokens=context.output_tokens,
        llm_calls_without_usage=context.llm_calls_without_usage,
        llm_seconds=context.llm_seconds,
        browser_seconds=context.browser_seconds,
        wall_seconds=wall,
    )