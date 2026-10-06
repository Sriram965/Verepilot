from __future__ import annotations

from typing import Any

from playwright.sync_api import (
    Error as PlaywrightError,
    Page,
)

from observation.extract import (
    ObservationExtractor,
)
from observation.models import Observation
from tools.schemas import (
    ClickRequest,
    GetPageStateRequest,
    GoBackRequest,
    NavigateRequest,
    PressRequest,
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


class BrowserToolRuntime:
    """
    Validated browser-tool execution layer.

    Responsibilities:

    - receive typed tool requests
    - validate requests
    - enforce observation freshness
    - resolve observation-scoped elements
    - enforce browser policy
    - execute validated Playwright actions
    - return typed ToolResults

    This class does not contain:
    - LLM logic
    - planning
    - LangGraph
    - verification
    - recovery policy
    """

    def __init__(
        self,
        *,
        page: Page,
        extractor: ObservationExtractor,
        policy: BrowserToolPolicy,
    ) -> None:

        self.page = page
        self.extractor = extractor
        self.policy = policy

        self.current_observation: (
            Observation | None
        ) = None

    def execute(
        self,
        request: ToolRequest,
    ) -> ToolResult:

        if isinstance(
            request,
            NavigateRequest,
        ):
            return self._navigate(request)

        if isinstance(
            request,
            GetPageStateRequest,
        ):
            return self._get_page_state(request)

        if isinstance(
            request,
            ClickRequest,
        ):
            return self._click(request)

        if isinstance(
            request,
            TypeRequest,
        ):
            return self._type(request)

        if isinstance(
            request,
            PressRequest,
        ):
            return self._press(request)

        if isinstance(
            request,
            GoBackRequest,
        ):
            return self._go_back(request)

        raise TypeError(
            f"Unsupported tool request: "
            f"{type(request)!r}"
        )

    def _get_page_state(
        self,
        request: GetPageStateRequest,
    ) -> ToolResult:

        del request

        before_url = self.page.url

        try:
            observation = (
                self.extractor.capture(
                    self.page
                )
            )

            self.current_observation = (
                observation
            )

        except PlaywrightError as exc:
            return self._failure(
                error_type=(
                    ToolErrorType.TOOL_EXECUTION_ERROR
                ),
                message=(
                    "Failed to capture "
                    f"page state: {exc}"
                ),
                before_url=before_url,
            )

        return ToolResult(
            ok=True,
            effect=ToolEffect(
                before_url=before_url,
                after_url=self.page.url,
                url_changed=(
                    before_url
                    != self.page.url
                ),
                dom_changed=False,
                dialog_appeared=False,
            ),
            data=observation,
        )

    def _navigate(
        self,
        request: NavigateRequest,
    ) -> ToolResult:

        before_url = self.page.url

        try:
            validate_navigate_request(
                request,
                self.policy,
            )

        except ValidationFailure as exc:
            return self._failure(
                error_type=exc.error_type,
                message=exc.message,
                before_url=before_url,
            )

        before_version = (
            self.extractor.compute_page_version(
                self.page
            )
        )

        try:
            response = (
                self.page.goto(
                    request.url,
                    wait_until="domcontentloaded",
                )
            )

            if response is not None:
                status = response.status

                if status >= 400:
                    return self._failure(
                        error_type=(
                            ToolErrorType.NAVIGATION_ERROR
                        ),
                        message=(
                            "Navigation returned "
                            f"HTTP {status}."
                        ),
                        before_url=before_url,
                    )

        except PlaywrightError as exc:
            return self._failure(
                error_type=(
                    ToolErrorType.NAVIGATION_ERROR
                ),
                message=(
                    f"Navigation failed: {exc}"
                ),
                before_url=before_url,
            )

        after_url = self.page.url

        after_version = (
            self.extractor.compute_page_version(
                self.page
            )
        )

        return ToolResult(
            ok=True,
            effect=ToolEffect(
                before_url=before_url,
                after_url=after_url,
                url_changed=(
                    before_url != after_url
                ),
                dom_changed=(
                    before_version
                    != after_version
                ),
                dialog_appeared=False,
            ),
        )

    def _click(
        self,
        request: ClickRequest,
    ) -> ToolResult:

        before_url = self.page.url

        try:
            locator = (
                validate_click_request(
                    request,
                    observation=(
                        self.current_observation
                    ),
                    page=self.page,
                    extractor=self.extractor,
                )
            )

        except ValidationFailure as exc:
            return self._failure(
                error_type=exc.error_type,
                message=exc.message,
                before_url=before_url,
            )

        before_version = (
            self.extractor.compute_page_version(
                self.page
            )
        )

        try:
            locator.click()

        except PlaywrightError as exc:
            return self._failure(
                error_type=(
                    ToolErrorType.TOOL_EXECUTION_ERROR
                ),
                message=(
                    f"Click failed: {exc}"
                ),
                before_url=before_url,
            )

        after_url = self.page.url

        after_version = (
            self.extractor.compute_page_version(
                self.page
            )
        )

        return ToolResult(
            ok=True,
            effect=ToolEffect(
                before_url=before_url,
                after_url=after_url,
                url_changed=(
                    before_url
                    != after_url
                ),
                dom_changed=(
                    before_version
                    != after_version
                ),
                dialog_appeared=False,
            ),
        )

    def _type(
        self,
        request: TypeRequest,
    ) -> ToolResult:

        before_url = self.page.url

        try:
            locator = (
                validate_type_request(
                    request,
                    observation=(
                        self.current_observation
                    ),
                    page=self.page,
                    extractor=self.extractor,
                    policy=self.policy,
                )
            )

        except ValidationFailure as exc:
            return self._failure(
                error_type=exc.error_type,
                message=exc.message,
                before_url=before_url,
            )

        before_version = (
            self.extractor.compute_page_version(
                self.page
            )
        )

        try:
            locator.fill(
                request.text
            )

        except PlaywrightError as exc:
            return self._failure(
                error_type=(
                    ToolErrorType.TOOL_EXECUTION_ERROR
                ),
                message=(
                    f"Type operation failed: "
                    f"{exc}"
                ),
                before_url=before_url,
            )

        after_url = self.page.url

        after_version = (
            self.extractor.compute_page_version(
                self.page
            )
        )

        return ToolResult(
            ok=True,
            effect=ToolEffect(
                before_url=before_url,
                after_url=after_url,
                url_changed=(
                    before_url
                    != after_url
                ),
                dom_changed=(
                    before_version
                    != after_version
                ),
                dialog_appeared=False,
            ),
        )

    def _press(
        self,
        request: PressRequest,
    ) -> ToolResult:

        before_url = self.page.url

        try:
            validate_press_request(
                request,
                observation=(
                    self.current_observation
                ),
                page=self.page,
                extractor=self.extractor,
                policy=self.policy,
            )

        except ValidationFailure as exc:
            return self._failure(
                error_type=exc.error_type,
                message=exc.message,
                before_url=before_url,
            )

        before_version = (
            self.extractor.compute_page_version(
                self.page
            )
        )

        try:
            self.page.keyboard.press(
                request.key
            )

        except PlaywrightError as exc:
            return self._failure(
                error_type=(
                    ToolErrorType.TOOL_EXECUTION_ERROR
                ),
                message=(
                    f"Press operation failed: "
                    f"{exc}"
                ),
                before_url=before_url,
            )

        after_url = self.page.url

        after_version = (
            self.extractor.compute_page_version(
                self.page
            )
        )

        return ToolResult(
            ok=True,
            effect=ToolEffect(
                before_url=before_url,
                after_url=after_url,
                url_changed=(
                    before_url
                    != after_url
                ),
                dom_changed=(
                    before_version
                    != after_version
                ),
                dialog_appeared=False,
            ),
        )

    def _go_back(
        self,
        request: GoBackRequest,
    ) -> ToolResult:

        before_url = self.page.url

        try:
            validate_go_back_request(
                request,
                observation=(
                    self.current_observation
                ),
                page=self.page,
                extractor=self.extractor,
            )

        except ValidationFailure as exc:
            return self._failure(
                error_type=exc.error_type,
                message=exc.message,
                before_url=before_url,
            )

        before_version = (
            self.extractor.compute_page_version(
                self.page
            )
        )

        try:
            self.page.go_back(
                wait_until="domcontentloaded"
            )

        except PlaywrightError as exc:
            return self._failure(
                error_type=(
                    ToolErrorType.NAVIGATION_ERROR
                ),
                message=(
                    f"Go-back operation failed: "
                    f"{exc}"
                ),
                before_url=before_url,
            )

        after_url = self.page.url

        after_version = (
            self.extractor.compute_page_version(
                self.page
            )
        )

        return ToolResult(
            ok=True,
            effect=ToolEffect(
                before_url=before_url,
                after_url=after_url,
                url_changed=(
                    before_url
                    != after_url
                ),
                dom_changed=(
                    before_version
                    != after_version
                ),
                dialog_appeared=False,
            ),
        )

    @staticmethod
    def _failure(
        *,
        error_type: ToolErrorType,
        message: str,
        before_url: str,
    ) -> ToolResult:

        return ToolResult(
            ok=False,
            error_type=error_type,
            message=message,
            effect=ToolEffect(
                before_url=before_url,
                after_url=before_url,
                url_changed=False,
                dom_changed=False,
                dialog_appeared=False,
            ),
        )


def build_browser_tool_runtime(
    *,
    page: Page,
    extractor: ObservationExtractor,
    allowed_origins: frozenset[str],
) -> BrowserToolRuntime:

    return BrowserToolRuntime(
        page=page,
        extractor=extractor,
        policy=BrowserToolPolicy(
            allowed_origins=allowed_origins
        ),
    )
