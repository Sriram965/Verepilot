from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from agent.baseline_a import run_baseline_a
from benchmark.runner import load_task
from benchmark.schema import TaskSpec
from environments.sites.shop import ShopEnvironment
from observation import ObservationExtractor
from tools.browser_backend import BrowserBackend
from tools.runtime import build_browser_tool_runtime


class ScriptedToolModel:
    """
    Minimal fake chat model used to test the Baseline A
    LangGraph execution loop without making an API call.

    First response:
        click Laptop_A's Add to cart button

    Second response:
        finish

    This lets us test graph control flow independently
    from a real LLM provider.
    """

    def __init__(self) -> None:
        self.responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "click",
                        "args": {
                            "element_id": "el_003",
                            "observation_id": (
                                "obs_000001"
                            ),
                        },
                        "id": "call_001",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "finish",
                        "args": {
                            "claim": (
                                "Laptop_A was added "
                                "to the cart."
                            )
                        },
                        "id": "call_002",
                    }
                ],
            ),
        ]

    def invoke(self, messages):
        if not self.responses:
            raise AssertionError(
                "Fake model received more calls "
                "than expected."
            )

        return self.responses.pop(0)


def make_runtime(
    environment: ShopEnvironment,
    backend: BrowserBackend,
    page,
):
    extractor = ObservationExtractor()

    return build_browser_tool_runtime(
        page=page,
        extractor=extractor,
        allowed_origins=frozenset(
            {environment.base_url}
        ),
    )


def test_baseline_a_executes_browser_tool_and_finishes(
    tmp_path: Path,
):
    task = load_task(
        Path("benchmark/tasks/cart_001.yaml")
    )

    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            result = run_baseline_a(
                task=task,
                model=ScriptedToolModel(),
                browser_runtime=runtime,
                trace_path=(
                    tmp_path / "run.jsonl"
                ),
                start_url=(
                    environment.base_url
                    + "/products"
                ),
            )

            assert (
                result.status
                == "success_claimed"
            )

            assert (
                result.steps_used
                == 2
            )

            assert (
                result.llm_calls
                == 2
            )

            cookies = page.context.cookies()

            session_cookie = next(
                cookie["value"]
                for cookie in cookies
                    if cookie["name"]
                    == "vp_session"
            )

            assert (
                environment.state.get_cart(
                    session_cookie
                )
                == {
                    "Laptop_A": 1
                }
            )


def test_baseline_a_records_jsonl_trace(
    tmp_path: Path,
):
    task = load_task(
        Path("benchmark/tasks/cart_001.yaml")
    )

    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            trace_path = (
                tmp_path / "trace.jsonl"
            )

            run_baseline_a(
                task=task,
                model=ScriptedToolModel(),
                browser_runtime=runtime,
                trace_path=trace_path,
                start_url=(
                    environment.base_url
                    + "/products"
                ),
            )

    assert trace_path.exists()

    lines = (
        trace_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert lines

    records = [
        json.loads(line)
        for line in lines
    ]

    events = [
        record["event"]
        for record in records
    ]

    assert (
        events[0]
        == "run_started"
    )

    assert (
        "initial_navigation"
        in events
    )

    assert (
        "observation_captured"
        in events
    )

    assert (
        "llm_decision"
        in events
    )

    assert (
        "action_selected"
        in events
    )

    assert (
        "tool_result"
        in events
    )

    assert (
        "finish_claimed"
        in events
    )

    assert (
        events[-1]
        == "run_finished"
    )


def test_baseline_a_stops_at_step_budget(
    tmp_path: Path,
):
    original_task = load_task(
        Path("benchmark/tasks/cart_001.yaml")
    )

    task = TaskSpec.model_validate(
        {
            **original_task.model_dump(
                mode="json"
            ),
            "max_steps": 1,
        }
    )

    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            result = run_baseline_a(
                task=task,
                model=ScriptedToolModel(),
                browser_runtime=runtime,
                trace_path=(
                    tmp_path / "budget.jsonl"
                ),
                start_url=(
                    environment.base_url
                    + "/products"
                ),
            )

            assert (
                result.status
                == "budget_exhausted"
            )

            assert (
                result.steps_used
                == 1
            )

            assert (
                result.llm_calls
                == 1
            )