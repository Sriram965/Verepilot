"""
What the browser did on a page, recorded for the tool runtime.

Four things are tracked since the last `reset()`:

- dialogs          alert/confirm/prompt/beforeunload messages. Every dialog is
                   dismissed (the safe choice: a confirm is never accepted on
                   the agent's behalf).
- main_frame_status  HTTP status of the last main-frame navigation response
                   (None if the page did not navigate).
- blocked_requests requests refused because their origin is not allowed
                   (recorded by the browser backend's origin enforcement).
                   `blocked_navigations` is the subset that tried to take the
                   page itself somewhere else; only those make an action a
                   policy violation (a blocked tracking pixel on a page the
                   agent legitimately loaded does not).
- pending_navigations  main-frame navigations that have started but not yet
                   committed or failed. `wait_until_settled` waits for this
                   to reach zero, so an action's result is read AFTER the
                   page it triggered has loaded, not while it is mid-flight.

`attach_page_events(page)` works on any Playwright page and is idempotent, so
the backend and the runtime can both call it without double-recording.
"""

from __future__ import annotations

import time
import weakref
from dataclasses import dataclass, field

from playwright.sync_api import Dialog, Frame
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, Request, Response

# Pause after an action so a navigation it just triggered has time to start.
_START_GRACE_MS = 50


@dataclass
class PageEvents:
    dialogs: list[str] = field(default_factory=list)
    blocked_requests: list[str] = field(default_factory=list)
    blocked_navigations: list[str] = field(default_factory=list)
    main_frame_status: int | None = None
    pending_navigations: int = 0
    attached: bool = False

    def reset(self) -> None:
        self.dialogs.clear()
        self.blocked_requests.clear()
        self.blocked_navigations.clear()
        self.main_frame_status = None
        self.pending_navigations = 0


_REGISTRY: "weakref.WeakKeyDictionary[Page, PageEvents]" = (
    weakref.WeakKeyDictionary()
)


def events_for(page: Page) -> PageEvents:
    """The events record for `page` (created empty if needed)."""

    events = _REGISTRY.get(page)

    if events is None:
        events = _REGISTRY[page] = PageEvents()

    return events


def attach_page_events(page: Page) -> PageEvents:
    """Start recording dialogs and navigation status on `page`."""

    events = events_for(page)

    if events.attached:
        return events

    events.attached = True
    page.on("dialog", lambda dialog: _on_dialog(events, dialog))
    page.on(
        "response",
        lambda response: _on_response(page, events, response),
    )
    page.on("request", lambda request: _on_request(page, events, request))
    page.on(
        "requestfailed",
        lambda request: _on_request_failed(page, events, request),
    )
    page.on(
        "framenavigated",
        lambda frame: _on_frame_navigated(page, events, frame),
    )

    return events


def wait_until_settled(
    page: Page,
    events: PageEvents,
    *,
    timeout_ms: int = 3_000,
) -> None:
    """
    Wait for any navigation the last action started to finish loading.

    Bounded by `timeout_ms`; never raises (a closed page just returns).
    """

    try:
        page.wait_for_timeout(_START_GRACE_MS)

        deadline = time.monotonic() + timeout_ms / 1000

        while events.pending_navigations > 0 and time.monotonic() < deadline:
            page.wait_for_timeout(25)

        remaining_ms = max(100, int((deadline - time.monotonic()) * 1000))
        page.wait_for_load_state("domcontentloaded", timeout=remaining_ms)
    except PlaywrightError:
        pass


def _on_dialog(events: PageEvents, dialog: Dialog) -> None:
    events.dialogs.append(dialog.message)

    try:
        dialog.dismiss()
    except PlaywrightError:
        pass  # already handled


def _on_response(page: Page, events: PageEvents, response: Response) -> None:
    try:
        request = response.request

        if request.is_navigation_request() and request.frame == page.main_frame:
            events.main_frame_status = response.status
    except PlaywrightError:
        pass  # the page closed while the response was in flight


def _is_main_frame_navigation(page: Page, request: Request) -> bool:
    try:
        return (
            request.is_navigation_request()
            and request.frame == page.main_frame
        )
    except PlaywrightError:
        return False


def _on_request(page: Page, events: PageEvents, request: Request) -> None:
    # A redirect chain is one navigation: count only its first request.
    if _is_main_frame_navigation(page, request) and request.redirected_from is None:
        events.pending_navigations += 1


def _on_request_failed(page: Page, events: PageEvents, request: Request) -> None:
    if _is_main_frame_navigation(page, request):
        events.pending_navigations = max(0, events.pending_navigations - 1)


def _on_frame_navigated(page: Page, events: PageEvents, frame: Frame) -> None:
    if frame == page.main_frame:
        events.pending_navigations = max(0, events.pending_navigations - 1)