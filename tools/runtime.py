"""
Validated browser-tool execution.

The runtime receives a typed tool request, validates it, runs it in the
browser, and returns a typed ToolResult. It does not contain any LLM logic,
planning, verification, or recovery policy.

Every action (navigate, click, type, press, go_back) takes the same path:

    validate  ->  note the page's state  ->  run the Playwright action
        ->  let the page settle  ->  read what happened  ->  judge

so there is exactly one place that decides what a result looks like.

What the result reports
-----------------------
The ToolEffect describes the REAL browser state after the action, including
when the action failed (for example, navigating to a 404 page leaves the
browser on that page, and the effect says so).

If an action tries to take the page outside the allowed origins, the request
is blocked, the result is a POLICY_VIOLATION, and the browser is returned to
the page it was on (Chrome would otherwise leave it on an error page).

An action returns ok=False when:

- validation rejected it (nothing happened in the browser);
- the action itself failed (timeout, element covered by an overlay...);
- the action tried to navigate outside the allowed origins; the request was
  blocked (POLICY_VIOLATION). Other blocked requests, such as a tracking
  pixel on a normal page, are listed in the effect but do not fail the action;
- the page it landed on answered HTTP 4xx/5xx (NAVIGATION_ERROR). For a
  mutating action the message says the outcome is uncertain: the click did
  happen, so the server may or may not have applied the change. Verify
  before repeating it.

The effect of the latest action is also passed to the next observation as
`previous_action_effect`, so the model can see what its last action did.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page

from observation.extract import ObservationExtractor
from observation.models import Observation
from tools.page_events import attach_page_events, wait_until_settled
from tools.schemas import (
    TOOL_SIDE_EFFECTS,
    ClickRequest,
    GetPageStateRequest,
    GoBackRequest,
    NavigateRequest,
    PressRequest,
    SideEffectClass,
    ToolEffect,
    ToolErrorType,
    ToolRequest,
    ToolResult,
    TypeRequest,
)
from tools.validate import (
    BrowserToolPolicy,
    ValidationFailure,
    validate_click_request,
    validate_go_back_request,
    validate_navigate_request,
    validate_press_request,
    validate_type_request,
)

DEFAULT_ACTION_TIMEOUT_MS = 5_000

_Error = tuple[ToolErrorType, str]


class _ActionRefused(Exception):
    """An action decided, at run time, that it cannot be performed."""

    def __init__(self, error_type: ToolErrorType, message: str) -> None:
        super().__init__(message)

        self.error_type = error_type
        self.message = message


class BrowserToolRuntime:
    def __init__(
        self,
        *,
        page: Page,
        extractor: ObservationExtractor,
        policy: BrowserToolPolicy,
        action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
    ) -> None:
        self.page = page
        self.extractor = extractor
        self.policy = policy
        self.action_timeout_ms = action_timeout_ms

        self.events = attach_page_events(page)

        self.current_observation: Observation | None = None
        self.last_action: dict[str, Any] | None = None

        self._handlers: dict[type, Callable[[Any], ToolResult]] = {
            NavigateRequest: self._navigate,
            GetPageStateRequest: self._get_page_state,
            ClickRequest: self._click,
            TypeRequest: self._type,
            PressRequest: self._press,
            GoBackRequest: self._go_back,
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        handler = self._handlers.get(type(request))

        if handler is None:
            raise TypeError(f"Unsupported tool request: {type(request)!r}")

        result = handler(request)

        if not isinstance(request, GetPageStateRequest):
            self.last_action = self._summarize(request.tool, result)

        return result

    # -- tools -------------------------------------------------------------

    def _get_page_state(self, request: GetPageStateRequest) -> ToolResult:
        before_url = self.page.url

        try:
            observation = self.extractor.capture(
                self.page,
                previous_action_effect=self.last_action,
            )
        except PlaywrightError as exc:
            return self._rejected(
                ToolErrorType.TOOL_EXECUTION_ERROR,
                f"Failed to capture page state: {self._short(exc)}",
                before_url,
            )

        self.current_observation = observation

        after_url = self.page.url

        return ToolResult(
            ok=True,
            effect=self._effect(before_url, after_url, dom_changed=False),
            data=observation,
        )

    def _navigate(self, request: NavigateRequest) -> ToolResult:
        return self._run(
            request.tool,
            validate=lambda: validate_navigate_request(request, self.policy),
            act=lambda _: self.page.goto(
                request.url, wait_until="domcontentloaded"
            ),
            failure_type=ToolErrorType.NAVIGATION_ERROR,
            label="Navigation",
            uses_observation=False,
        )

    def _click(self, request: ClickRequest) -> ToolResult:
        return self._run(
            request.tool,
            validate=lambda: validate_click_request(
                request,
                observation=self.current_observation,
                page=self.page,
                extractor=self.extractor,
            ),
            act=lambda locator: locator.click(timeout=self.action_timeout_ms),
            failure_type=ToolErrorType.TOOL_EXECUTION_ERROR,
            label="Click",
        )

    def _type(self, request: TypeRequest) -> ToolResult:
        return self._run(
            request.tool,
            validate=lambda: validate_type_request(
                request,
                observation=self.current_observation,
                page=self.page,
                extractor=self.extractor,
                policy=self.policy,
            ),
            act=lambda locator: locator.fill(
                request.text, timeout=self.action_timeout_ms
            ),
            failure_type=ToolErrorType.TOOL_EXECUTION_ERROR,
            label="Type",
        )

    def _press(self, request: PressRequest) -> ToolResult:
        return self._run(
            request.tool,
            validate=lambda: validate_press_request(
                request,
                observation=self.current_observation,
                page=self.page,
                extractor=self.extractor,
                policy=self.policy,
            ),
            act=lambda _: self.page.keyboard.press(request.key),
            failure_type=ToolErrorType.TOOL_EXECUTION_ERROR,
            label="Key press",
        )

    def _go_back(self, request: GoBackRequest) -> ToolResult:
        return self._run(
            request.tool,
            validate=lambda: validate_go_back_request(
                request,
                observation=self.current_observation,
                page=self.page,
                extractor=self.extractor,
            ),
            act=self._back,
            failure_type=ToolErrorType.NAVIGATION_ERROR,
            label="Go back",
        )

    def _back(self, _: Any) -> None:
        before_url = self.page.url
        response = self.page.go_back(wait_until="domcontentloaded")

        if response is None and self.page.url == before_url:
            raise _ActionRefused(
                ToolErrorType.NAVIGATION_ERROR,
                "There is no previous page to go back to.",
            )

    # -- the one shared path for actions ------------------------------------

    def _run(
        self,
        tool: str,
        *,
        validate: Callable[[], Any],
        act: Callable[[Any], Any],
        failure_type: ToolErrorType,
        label: str,
        uses_observation: bool = True,
    ) -> ToolResult:
        before_url = self.page.url

        try:
            target = validate()
        except ValidationFailure as exc:
            return self._rejected(exc.error_type, exc.message, before_url)

        if uses_observation:
            # Validation just proved the live page still matches it.
            assert self.current_observation is not None
            before_version: str | None = self.current_observation.page_version
        else:
            before_version = self._page_version()

        self.events.reset()
        error: _Error | None = None

        try:
            act(target)
        except _ActionRefused as exc:
            error = (exc.error_type, exc.message)
        except PlaywrightError as exc:
            error = self._classify(exc, failure_type, label)

        self._settle()
        restored = self._restore_if_stranded(before_url)

        effect = self._effect(
            before_url,
            self.page.url,
            dom_changed=before_version != self._page_version(),
        )

        error = self._judge(tool, effect, error, restored)

        if error is None:
            return ToolResult(ok=True, effect=effect)

        return ToolResult(
            ok=False,
            error_type=error[0],
            message=error[1],
            effect=effect,
        )

    def _judge(
        self,
        tool: str,
        effect: ToolEffect,
        error: _Error | None,
        restored: bool,
    ) -> _Error | None:
        """Decide the final error, if any. Blocked > action error > HTTP."""

        blocked = self.events.blocked_navigations

        if blocked:
            message = (
                "The action tried to navigate outside the allowed origins "
                f"({blocked[0]}); the request was blocked."
            )

            if restored:
                message += " The browser was returned to the previous page."

            return ToolErrorType.POLICY_VIOLATION, message

        if error is not None:
            return error

        status = effect.http_status

        if status is not None and status >= 400:
            message = f"The server answered HTTP {status}."

            if TOOL_SIDE_EFFECTS[tool] is SideEffectClass.MUTATING:
                message += (
                    " The action was performed, so the server may or may "
                    "not have applied the change. Check the page state "
                    "before repeating a mutating action."
                )

            return ToolErrorType.NAVIGATION_ERROR, message

        return None

    # -- helpers -------------------------------------------------------------

    def _effect(
        self,
        before_url: str,
        after_url: str,
        *,
        dom_changed: bool,
    ) -> ToolEffect:
        return ToolEffect(
            before_url=before_url,
            after_url=after_url,
            url_changed=before_url != after_url,
            dom_changed=dom_changed,
            dialog_appeared=bool(self.events.dialogs),
            dialog_messages=list(self.events.dialogs),
            http_status=self.events.main_frame_status,
            blocked_requests=list(self.events.blocked_requests),
        )

    def _rejected(
        self,
        error_type: ToolErrorType,
        message: str,
        url: str,
    ) -> ToolResult:
        """A failure where nothing happened in the browser."""

        return ToolResult(
            ok=False,
            error_type=error_type,
            message=message,
            effect=ToolEffect(
                before_url=url,
                after_url=url,
                url_changed=False,
                dom_changed=False,
                dialog_appeared=False,
            ),
        )

    def _settle(self) -> None:
        """Wait for a navigation the action started to finish loading."""

        wait_until_settled(self.page, self.events)

    def _restore_if_stranded(self, before_url: str) -> bool:
        """
        A blocked navigation leaves Chrome on its own error page. Put the
        browser back where it was so the agent can carry on. Returns True if
        a restore was needed.
        """

        if not (
            self.events.blocked_navigations
            and self.page.url.startswith("chrome-error:")
        ):
            return False

        try:
            self.page.go_back(wait_until="domcontentloaded")

            if self.page.url != before_url:
                self.page.goto(before_url, wait_until="domcontentloaded")
        except PlaywrightError:
            return False

        # The restore is not the agent's navigation: forget its status.
        self.events.main_frame_status = None

        return True

    def _page_version(self) -> str | None:
        try:
            return self.extractor.compute_page_version(self.page)
        except PlaywrightError:
            return None  # mid-navigation: counts as changed

    @staticmethod
    def _short(exc: Exception) -> str:
        """First line of a Playwright error (the rest is a call log)."""

        lines = str(exc).strip().splitlines()

        return (lines[0] if lines else type(exc).__name__)[:300]

    def _classify(
        self,
        exc: PlaywrightError,
        failure_type: ToolErrorType,
        label: str,
    ) -> _Error:
        if "intercepts pointer events" in str(exc):
            return (
                ToolErrorType.ELEMENT_NOT_ACTIONABLE,
                f"{label} failed: another element is covering the target "
                "(possibly an overlay or popup).",
            )

        return failure_type, f"{label} failed: {self._short(exc)}"

    @staticmethod
    def _summarize(tool: str, result: ToolResult) -> dict[str, Any]:
        return {
            "tool": tool,
            "ok": result.ok,
            "error_type": (
                result.error_type.value if result.error_type else None
            ),
            "message": result.message,
            **result.effect.model_dump(mode="json"),
        }


def build_browser_tool_runtime(
    *,
    page: Page,
    extractor: ObservationExtractor,
    allowed_origins: frozenset[str],
    action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
) -> BrowserToolRuntime:
    return BrowserToolRuntime(
        page=page,
        extractor=extractor,
        policy=BrowserToolPolicy(allowed_origins=allowed_origins),
        action_timeout_ms=action_timeout_ms,
    )