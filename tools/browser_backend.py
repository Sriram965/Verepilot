from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)


@dataclass(frozen=True)
class BrowserBackendConfig:
    """
    Configuration for the direct Playwright backend.

    This is internal runtime configuration.
    It is not an agent-facing tool schema.
    """

    headless: bool = True
    timeout_ms: int = 10_000
    navigation_timeout_ms: int = 15_000


class BrowserBackend:
    """
    Direct Playwright browser backend.

    Responsibilities at this phase:

    - start Playwright
    - launch Chromium
    - create an isolated browser context
    - create pages
    - navigate pages
    - close browser resources

    This layer intentionally does NOT contain:
    - LLM logic
    - tool validation
    - observation extraction
    - verifier logic
    - recovery logic
    """

    def __init__(
        self,
        config: BrowserBackendConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else BrowserBackendConfig()
        )

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    def start(self) -> None:
        if self._playwright is not None:
            raise RuntimeError(
                "BrowserBackend is already started"
            )

        self._playwright = sync_playwright().start()

        try:
            self._browser = (
                self._playwright.chromium.launch(
                    headless=self.config.headless
                )
            )

            self._context = (
                self._browser.new_context()
            )

            self._context.set_default_timeout(
                self.config.timeout_ms
            )

            self._context.set_default_navigation_timeout(
                self.config.navigation_timeout_ms
            )

        except Exception:
            self.close()
            raise

    @property
    def browser(self) -> Browser:
        if self._browser is None:
            raise RuntimeError(
                "BrowserBackend has not been started"
            )

        return self._browser

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError(
                "BrowserBackend has not been started"
            )

        return self._context

    def new_page(self) -> Page:
        page = self.context.new_page()

        page.set_default_timeout(
            self.config.timeout_ms
        )

        page.set_default_navigation_timeout(
            self.config.navigation_timeout_ms
        )

        return page

    def navigate(
        self,
        page: Page,
        url: str,
    ):
        """
        Low-level browser navigation.

        Tool-level validation belongs elsewhere.
        """
        return page.goto(
            url,
            wait_until="domcontentloaded",
        )

    def close(self) -> None:
        """
        Close resources in reverse ownership order.
        """

        if self._context is not None:
            self._context.close()
            self._context = None

        if self._browser is not None:
            self._browser.close()
            self._browser = None

        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def __enter__(self) -> "BrowserBackend":
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        self.close()
