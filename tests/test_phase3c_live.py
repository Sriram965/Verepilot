from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.baseline_a import run_baseline_a
from agent.llm import LLMConfig, create_bound_llm
from benchmark.runner import load_task
from environments.sites.shop import ShopEnvironment
from observation import ObservationExtractor
from tools.browser_backend import BrowserBackend
from tools.runtime import build_browser_tool_runtime


def make_runtime(
    environment: ShopEnvironment,
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


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"),
    reason="Set RUN_LIVE_TESTS=1 to run live LLM tests.",
)
def test_phase3c_real_llm_end_to_end(
    tmp_path: Path,
) -> None:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        pytest.fail(
            "OPENAI_API_KEY must be set for the live LLM test."
        )

    model_name = os.getenv("VERIPILOT_MODEL")

    if not model_name:
        pytest.fail(
            "VERIPILOT_MODEL must be set for the live LLM test."
        )

    task = load_task(
        Path("benchmark/tasks/cart_001.yaml")
    )

    model = create_bound_llm(
        LLMConfig(
            model=model_name,
        )
    )

    trace_path = (
        tmp_path / "phase3c_live.jsonl"
    )

    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            runtime = make_runtime(
                environment,
                page,
            )

            result = run_baseline_a(
                task=task,
                browser_runtime=runtime,
                model=model,
                start_url=f"{environment.base_url}{task.start_url}",
                trace_path=trace_path,
            )

            assert result.status == (
                "success_claimed"
            )

            assert result.steps_used >= 1
            assert result.llm_calls >= 1

            cookies = page.context.cookies()

            session_cookie = next(
                (
                    cookie["value"]
                    for cookie in cookies
                    if cookie["name"] == "vp_session"
                ),
                None,
            )

            assert session_cookie is not None

            assert environment.state.get_cart(
                session_cookie
            ) == {
                "Laptop_A": 1
            }

            assert trace_path.exists()
            assert trace_path.stat().st_size > 0