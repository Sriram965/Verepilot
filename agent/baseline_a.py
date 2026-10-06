from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from agent.graph import (
    BaselineAContext,
    build_baseline_a_graph,
)
from agent.prompts.baseline_a import (
    BASELINE_A_SYSTEM_PROMPT,
)
from agent.trace import JsonlTraceWriter
from benchmark.schema import TaskSpec
from tools.runtime import BrowserToolRuntime
from tools.schemas import NavigateRequest


@dataclass(frozen=True)
class BaselineARunResult:
    run_id: str
    status: str
    steps_used: int
    llm_calls: int
    final_messages: list


def run_baseline_a(
    *,
    task: TaskSpec,
    model: BaseChatModel,
    browser_runtime: BrowserToolRuntime,
    trace_path: str | Path,
    start_url: str,
) -> BaselineARunResult:

    run_id = uuid4().hex

    tracer = JsonlTraceWriter(
        trace_path
    )

    context = BaselineAContext(
        task=task,
        model=model,
        browser_runtime=browser_runtime,
        tracer=tracer,
        run_id=run_id,
    )

    tracer.write(
        event="run_started",
        run_id=run_id,
        payload={
            "task_id": task.task_id,
            "start_url": start_url,
            "max_steps": task.max_steps,
        },
    )

    start_result = browser_runtime.execute(
        NavigateRequest(
            tool="navigate",
            url=start_url,
        )
    )

    tracer.write(
        event="initial_navigation",
        run_id=run_id,
        payload=start_result,
    )

    if not start_result.ok:
        context.status = "failed"

        tracer.write(
            event="run_finished",
            run_id=run_id,
            payload={
                "status": context.status,
                "steps_used": context.steps_used,
                "llm_calls": context.llm_calls,
            },
        )

        return BaselineARunResult(
            run_id=run_id,
            status=context.status,
            steps_used=context.steps_used,
            llm_calls=context.llm_calls,
            final_messages=[],
        )

    graph = build_baseline_a_graph()

    initial_messages = [
        SystemMessage(
            content=BASELINE_A_SYSTEM_PROMPT
        ),
        HumanMessage(
            content=(
                "Complete this web task:\n\n"
                f"{task.instruction}"
            )
        ),
    ]

    result = graph.invoke(
        {
            "messages": initial_messages
        },
        context=context,
    )

    tracer.write(
        event="run_finished",
        run_id=run_id,
        payload={
            "status": context.status,
            "steps_used": context.steps_used,
            "llm_calls": context.llm_calls,
        },
    )

    return BaselineARunResult(
        run_id=run_id,
        status=context.status,
        steps_used=context.steps_used,
        llm_calls=context.llm_calls,
        final_messages=result[
            "messages"
        ],
    )
