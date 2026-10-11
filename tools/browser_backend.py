"""
Direct Playwright backend.

Owns the browser, one isolated context, and its pages. Pages it creates are
watched (see tools/page_events.py) so the tool runtime can report dialogs,
HTTP status, and blocked requests.

Origin enforcement: if `allowed_origins` is set, EVERY request the browser
makes (link clicks, form posts, redirects, scripts, images) must go to an
allowed origin, or it is aborted and recorded. Validating only the
`navigate` tool would let a click on an external link leave the benchmark
sites. Pass the same set to the runtime's policy.

This layer contains no LLM logic, tool validation, observation extraction,
verification or recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Request,
    Route,
    sync_playwright,
)
from playwright.sync_api import Error as PlaywrightError

from tools.origins import origin_of
from tools.page_events import PageEvents, attach_page_events, events_for


@dataclass(frozen=True)
class BrowserBackendConfig:
    headless: bool = True
    timeout_ms: int = 10_000
    navigation_timeout_ms: int = 15_000

    # Exact origins like "http://127.0.0.1:5000". None means no restriction.
    allowed_origins: frozenset[str] | None = None


class BrowserBackend:
    def __init__(self, config: BrowserBackendConfig | None = None) -> None:
        self.config = config if config is not None else BrowserBackendConfig()

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    def start(self) -> None:
        if self._playwright is not None:
            raise RuntimeError("BrowserBackend is already started")

        self._playwright = sync_playwright().start()

        try:
            self._browser = self._playwright.chromium.launch(
                headless=self.config.headless
            )
            self._context = self._browser.new_context()
            self._context.set_default_timeout(self.config.timeout_ms)
            self._context.set_default_navigation_timeout(
                self.config.navigation_timeout_ms
            )

            if self.config.allowed_origins is not None:
                self._context.route("**/*", self._enforce_origin)

        except Exception:
            self.close()
            raise

    @property
    def browser(self) -> Browser:
        if self._browser is None:
            raise RuntimeError("BrowserBackend has not been started")
        return self._browser

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("BrowserBackend has not been started")
        return self._context

    def new_page(self) -> Page:
        page = self.context.new_page()

        page.set_default_timeout(self.config.timeout_ms)
        page.set_default_navigation_timeout(self.config.navigation_timeout_ms)
        attach_page_events(page)

        return page

    def events(self, page: Page) -> PageEvents:
        return events_for(page)

    def navigate(self, page: Page, url: str):
        """Low-level navigation. Tool-level validation belongs elsewhere."""
        return page.goto(url, wait_until="domcontentloaded")

    def close(self) -> None:
        """Close in reverse order; one failure must not leak the rest."""

        try:
            if self._context is not None:
                self._context.close()
        finally:
            self._context = None
            try:
                if self._browser is not None:
                    self._browser.close()
            finally:
                self._browser = None
                try:
                    if self._playwright is not None:
                        self._playwright.stop()
                finally:
                    self._playwright = None

    def __enter__(self) -> "BrowserBackend":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _enforce_origin(self, route: Route, request: Request) -> None:
        try:
            if origin_of(request.url) in self.config.allowed_origins:
                route.continue_()
                return

            page = request.frame.page
            events = events_for(page)
            events.blocked_requests.append(request.url)

            if (
                request.is_navigation_request()
                and request.frame == page.main_frame
            ):
                events.blocked_navigations.append(request.url)

            route.abort("blockedbyclient")
        except PlaywrightError:
            pass  # the page or context closed while the request was in flight