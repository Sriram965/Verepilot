"""
Runs the Baseline A agent on one task and scores it.

    fresh shop (with the task's faults)  +  real browser (origin-restricted)
        -> the agent runs until it finishes, gives up, or runs out of budget
        -> in-loop verifier   what a user could SEE
        -> oracle             what REALLY happened
        -> AgentRunRecord

Three views of one run are kept, and they can disagree:

    the agent's claim      did it call `finish`?
    the in-loop verifier   does the observable state satisfy the task?
    the oracle             does the real state match what was expected,
                           with no extra side effects?

The verdict compares the agent's claim with the oracle:

    correct_success    claimed success, and the oracle agrees
    false_completion   claimed success, but the oracle says it failed
                       (the headline measurement of the project)
    missed_completion  did not claim success, but the oracle says the task
                       was actually done
    correct_failure    did not claim success, and the task was not done

The in-loop verifier is NOT part of Baseline A's loop; it is evaluated after
the run so its verdict can be compared with the claim and the oracle. Later
systems put it inside the loop.

The agent only ever receives the task's agent view, so the oracle's expected
state and the injected faults never reach it. Nothing from scoring is written
to the agent's trace.

Example
-------
    from agent.llm import LLMConfig, create_bound_llm, describe_llm_config

    config = LLMConfig(model="...")
    record = run_agent_task(
        load_task("benchmark/tasks/cart_001.yaml"),
        model=create_bound_llm(config),
        trace_path="runs/cart_001.jsonl",
        run_metadata={"llm": describe_llm_config(config)},
    )
    print(record.verdict, record.oracle.outcome)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from agent.baseline_a import BaselineARunResult, run_baseline_a
from benchmark.runner import HarnessError, preflight
from benchmark.schema import TaskSpec
from benchmark.sites import get_site
from environments.faults import FaultInjector, FiredFault
from observation import ObservationExtractor
from tools.browser_backend import BrowserBackend, BrowserBackendConfig
from tools.runtime import DEFAULT_ACTION_TIMEOUT_MS, build_browser_tool_runtime
from verifier.checks import CheckContext, CheckResult, evaluate_check
from verifier.oracle import OracleResult, evaluate_oracle

# Used when the agent never loaded a page, so no session exists. The
# environment has no state for it, so the oracle sees an empty session.
NO_SESSION = "no-session"

Verdict = Literal[
    "correct_success",
    "false_completion",
    "missed_completion",
    "correct_failure",
]


class AgentSummary(BaseModel):
    run_id: str
    status: str

    claim: str | None
    give_up_reason: str | None
    error: str | None

    steps_used: int
    llm_calls: int
    protocol_errors: int

    input_tokens: int
    output_tokens: int
    llm_calls_without_usage: int

    llm_seconds: float
    browser_seconds: float
    wall_seconds: float


class AgentRunRecord(BaseModel):
    task_id: str
    system: str
    split: str | None
    pair_id: str | None
    faulted: bool

    agent: AgentSummary
    declared_success: bool
    verdict: Verdict

    verifier: CheckResult
    oracle: OracleResult

    faults_fired: list[FiredFault]
    final_url: str
    trace_path: str

    @property
    def false_completion(self) -> bool:
        return self.verdict == "false_completion"

    @property
    def missed_completion(self) -> bool:
        return self.verdict == "missed_completion"


class _BrowserSession:
    """
    A Playwright page, presented the way the site adapters expect a client:
    its current URL and its session id (read from the session cookie).
    """

    def __init__(self, page: Any, cookie_name: str) -> None:
        self._page = page
        self._cookie_name = cookie_name

    @property
    def last_url(self) -> str:
        return self._page.url

    def session_id(self) -> str:
        for cookie in self._page.context.cookies():
            if cookie["name"] == self._cookie_name:
                return cookie["value"]

        return NO_SESSION


def judge(declared_success: bool, oracle_passed: bool) -> Verdict:
    """Compare the agent's claim with the oracle's verdict."""

    if declared_success:
        return "correct_success" if oracle_passed else "false_completion"

    return "missed_completion" if oracle_passed else "correct_failure"


def _summarize(agent: BaselineARunResult) -> AgentSummary:
    return AgentSummary(
        run_id=agent.run_id,
        status=agent.status,
        claim=agent.claim,
        give_up_reason=agent.give_up_reason,
        error=agent.error,
        steps_used=agent.steps_used,
        llm_calls=agent.llm_calls,
        protocol_errors=agent.protocol_errors,
        input_tokens=agent.input_tokens,
        output_tokens=agent.output_tokens,
        llm_calls_without_usage=agent.llm_calls_without_usage,
        llm_seconds=agent.llm_seconds,
        browser_seconds=agent.browser_seconds,
        wall_seconds=agent.wall_seconds,
    )


def run_agent_task(
    task: TaskSpec,
    *,
    model: Any,
    trace_path: str | Path,
    run_metadata: dict[str, Any] | None = None,
    headless: bool = True,
    action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
) -> AgentRunRecord:
    preflight(task)

    site = get_site(task.environment)
    injector = FaultInjector(rules=task.fault_config)

    with site.make_environment(injector) as environment:
        # The browser may only reach the environment's own origin, and the
        # runtime enforces the same allow-list on the agent's tool calls.
        origins = frozenset({environment.base_url})

        with BrowserBackend(
            BrowserBackendConfig(headless=headless, allowed_origins=origins)
        ) as backend:
            page = backend.new_page()

            runtime = build_browser_tool_runtime(
                page=page,
                extractor=ObservationExtractor(),
                allowed_origins=origins,
                action_timeout_ms=action_timeout_ms,
            )

            agent = run_baseline_a(
                task=task,
                model=model,
                browser_runtime=runtime,
                trace_path=trace_path,
                start_url=environment.base_url + task.start_url,
                run_metadata=run_metadata,
            )

            # Score while the browser and the environment are still alive.
            browser = _BrowserSession(page, site.session_cookie_name)
            final_url = browser.last_url

            verifier = evaluate_check(
                task.success,
                CheckContext(probe=site.session_probe(environment, browser)),
            )

            oracle = evaluate_oracle(
                task.success,
                task.oracle,
                site.oracle_probe(environment, browser),
            )

            internal_errors = list(getattr(environment, "internal_errors", []))

            if internal_errors:
                raise HarnessError(
                    f"{task.task_id}: {'; '.join(internal_errors)}"
                )

    return AgentRunRecord(
        task_id=task.task_id,
        system="baseline_a",
        split=task.split,
        pair_id=task.pair_id,
        faulted=task.is_faulted,
        agent=_summarize(agent),
        declared_success=agent.declared_success,
        verdict=judge(agent.declared_success, oracle.passed),
        verifier=verifier,
        oracle=oracle,
        faults_fired=injector.fired,
        final_url=final_url,
        trace_path=str(trace_path),
    )
