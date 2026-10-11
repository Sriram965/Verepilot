"""
Validation for browser tool calls.

Every tool call goes through three layers before anything touches the
browser:

1. Schema     (Pydantic, in tools/schemas.py) - shape and types.
2. Semantic   - does the call make sense for the CURRENT page? The
                observation id and page version must still match, the
                element must exist, be visible, enabled, and be the right
                kind of element for the action.
3. Policy     - only allowed origins, allowed keys, bounded text.

A failed check raises `ValidationFailure` carrying a typed error, which the
runtime turns into a failed ToolResult. Nothing in this module may raise
anything else: a malformed URL, a closed page or a bad selector all become a
ValidationFailure, never an uncaught exception that would kill the agent
loop.

Origin policy applies to the `navigate` tool here. The browser backend
enforces the same allow-list on every request the browser makes, so a click
on an external link is stopped too.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator, Page

from observation.extract import ObservationExtractor
from observation.models import InteractiveElement, Observation
from tools.origins import origin_of
from tools.schemas import (
    ClickRequest,
    GoBackRequest,
    NavigateRequest,
    PressRequest,
    ToolErrorType,
    TypeRequest,
)

DEFAULT_ALLOWED_KEYS = frozenset(
    {
        "Backspace",
        "Tab",
        "Enter",
        "Shift",
        "Control",
        "Alt",
        "Meta",
        "Escape",
        "ArrowDown",
        "ArrowUp",
        "ArrowLeft",
        "ArrowRight",
        "End",
        "Home",
        "Insert",
        "Delete",
        "PageDown",
        "PageUp",
        "Space",
        *{f"F{i}" for i in range(1, 13)},
        *{f"Digit{i}" for i in range(10)},
        *{f"Key{chr(code)}" for code in range(ord("A"), ord("Z") + 1)},
    }
)

# <input> types the agent may type into. These are the ones a person types
# text or a date into, and that Playwright's fill() accepts. Checkboxes,
# radios, buttons, file pickers, sliders and colour pickers are operated by
# clicking, not typing.
TYPEABLE_INPUT_TYPES = frozenset(
    {
        "text",
        "search",
        "email",
        "url",
        "tel",
        "password",
        "number",
        "date",
        "time",
        "datetime-local",
        "month",
        "week",
    }
)


@dataclass(frozen=True)
class BrowserToolPolicy:
    """
    Runtime policy for browser tool execution.

    The allowed origins are exact origins ("http://127.0.0.1:5000"), not
    arbitrary domains or URL prefixes.
    """

    allowed_origins: frozenset[str]

    max_text_chars: int = 2_000

    allowed_keys: frozenset[str] = DEFAULT_ALLOWED_KEYS


class ValidationFailure(Exception):
    def __init__(
        self,
        error_type: ToolErrorType,
        message: str,
    ) -> None:
        super().__init__(message)

        self.error_type = error_type
        self.message = message


# --------------------------------------------------------------------------
# Policy: origins
# --------------------------------------------------------------------------


def parse_tool_url_origin(url: str) -> str:
    """
    The "scheme://host:port" origin of `url`, or a ValidationFailure.

    Never raises anything else, including for malformed URLs such as an
    out-of-range port.
    """

    parsed = urlparse(url)

    if parsed.username or parsed.password:
        raise ValidationFailure(
            ToolErrorType.POLICY_VIOLATION,
            "URLs containing credentials are not allowed.",
        )

    if not parsed.scheme:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            "URL must contain a scheme.",
        )

    if not parsed.hostname:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            "URL must contain a hostname.",
        )

    try:
        parsed.port
    except ValueError:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            "URL contains an invalid port.",
        )

    origin = origin_of(url)

    if origin is None:
        raise ValidationFailure(
            ToolErrorType.POLICY_VIOLATION,
            "Only HTTP and HTTPS URLs are supported.",
        )

    return origin


def validate_allowed_origin(
    url: str,
    policy: BrowserToolPolicy,
) -> None:
    origin = parse_tool_url_origin(url)

    if origin not in policy.allowed_origins:
        raise ValidationFailure(
            ToolErrorType.POLICY_VIOLATION,
            f"Origin '{origin}' is not allowed by the browser policy.",
        )


# --------------------------------------------------------------------------
# Semantic: observation and elements
# --------------------------------------------------------------------------


def validate_observation(
    *,
    observation: Observation | None,
    requested_observation_id: str,
    page: Page,
    extractor: ObservationExtractor,
) -> None:
    """The action must refer to the observation that is still current."""

    if observation is None:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            "An observation is required before this action.",
        )

    if observation.observation_id != requested_observation_id:
        raise ValidationFailure(
            ToolErrorType.STALE_OBSERVATION,
            "The supplied observation_id does not match the current "
            "observation.",
        )

    try:
        live_page_version = extractor.compute_page_version(page)
    except PlaywrightError:
        if page.is_closed():
            raise ValidationFailure(
                ToolErrorType.TOOL_EXECUTION_ERROR,
                "The browser page is closed.",
            )

        raise ValidationFailure(
            ToolErrorType.STALE_OBSERVATION,
            "The page could not be read (it may be navigating). "
            "Request a new observation.",
        )

    if live_page_version != observation.page_version:
        raise ValidationFailure(
            ToolErrorType.STALE_OBSERVATION,
            "The browser page changed after the observation was captured.",
        )


def find_element(
    observation: Observation,
    element_id: str,
) -> InteractiveElement:
    for candidate in observation.interactive_elements:
        if candidate.element_id == element_id:
            return candidate

    raise ValidationFailure(
        ToolErrorType.VALIDATION_ERROR,
        f"Element '{element_id}' was not present in the observation.",
    )


def locate_element(
    element: InteractiveElement,
    page: Page,
) -> Locator:
    """
    Check the element as observed AND as it is in the live page, and return
    a locator for it. Observation ids are only valid for one observation,
    so the element must still resolve to exactly one visible, enabled node.
    """

    element_id = element.element_id

    if not element.visible:
        raise ValidationFailure(
            ToolErrorType.ELEMENT_NOT_ACTIONABLE,
            f"Element '{element_id}' was not visible in the observation.",
        )

    if not element.enabled:
        raise ValidationFailure(
            ToolErrorType.ELEMENT_NOT_ACTIONABLE,
            f"Element '{element_id}' was disabled in the observation.",
        )

    try:
        locator = page.locator(element.selector)

        if locator.count() != 1:
            raise ValidationFailure(
                ToolErrorType.ELEMENT_NOT_ACTIONABLE,
                f"Element '{element_id}' does not resolve to exactly one "
                "current DOM element.",
            )

        if not locator.is_visible():
            raise ValidationFailure(
                ToolErrorType.ELEMENT_NOT_ACTIONABLE,
                f"Element '{element_id}' is not currently visible.",
            )

        if not locator.is_enabled():
            raise ValidationFailure(
                ToolErrorType.ELEMENT_NOT_ACTIONABLE,
                f"Element '{element_id}' is not currently enabled.",
            )
    except PlaywrightError:
        raise ValidationFailure(
            ToolErrorType.ELEMENT_NOT_ACTIONABLE,
            f"Element '{element_id}' could not be located in the page.",
        )

    return locator


def resolve_element(
    *,
    element_id: str,
    observation: Observation,
    page: Page,
) -> Locator:
    return locate_element(find_element(observation, element_id), page)


def validate_type_target(*, element: InteractiveElement) -> None:
    """Only real text-entry fields can be typed into."""

    if element.tag == "input":
        supported = element.input_type in TYPEABLE_INPUT_TYPES
    elif element.tag == "textarea":
        supported = True
    else:
        supported = element.role == "textbox"  # e.g. contenteditable

    if not supported:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            f"Element '{element.element_id}' is not a supported text input.",
        )


# --------------------------------------------------------------------------
# Policy: text and keys
# --------------------------------------------------------------------------


def validate_text(
    text: str,
    policy: BrowserToolPolicy,
) -> None:
    if len(text) > policy.max_text_chars:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            "Text exceeds the maximum allowed length of "
            f"{policy.max_text_chars}.",
        )

    for char in text:
        if ord(char) < 32 and char not in {"\n", "\t"}:
            raise ValidationFailure(
                ToolErrorType.VALIDATION_ERROR,
                "Text contains unsupported control characters.",
            )


def validate_key(
    key: str,
    policy: BrowserToolPolicy,
) -> None:
    if not key:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            "Key cannot be empty.",
        )

    parts = key.split("+")

    if any(not part for part in parts):
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            f"Invalid key expression: '{key}'.",
        )

    for part in parts:
        if part in policy.allowed_keys or (
            len(part) == 1 and part.isprintable()
        ):
            continue

        raise ValidationFailure(
            ToolErrorType.POLICY_VIOLATION,
            f"Key '{part}' is not allowed by the keyboard policy.",
        )


# --------------------------------------------------------------------------
# Per-tool entry points (called by the runtime)
# --------------------------------------------------------------------------


def validate_navigate_request(
    request: NavigateRequest,
    policy: BrowserToolPolicy,
) -> None:
    validate_allowed_origin(request.url, policy)


def validate_click_request(
    request: ClickRequest,
    *,
    observation: Observation | None,
    page: Page,
    extractor: ObservationExtractor,
) -> Locator:
    validate_observation(
        observation=observation,
        requested_observation_id=request.observation_id,
        page=page,
        extractor=extractor,
    )

    assert observation is not None

    return resolve_element(
        element_id=request.element_id,
        observation=observation,
        page=page,
    )


def validate_type_request(
    request: TypeRequest,
    *,
    observation: Observation | None,
    page: Page,
    extractor: ObservationExtractor,
    policy: BrowserToolPolicy,
) -> Locator:
    validate_observation(
        observation=observation,
        requested_observation_id=request.observation_id,
        page=page,
        extractor=extractor,
    )

    assert observation is not None

    element = find_element(observation, request.element_id)

    validate_type_target(element=element)
    validate_text(request.text, policy)

    return locate_element(element, page)


def validate_press_request(
    request: PressRequest,
    *,
    observation: Observation | None,
    page: Page,
    extractor: ObservationExtractor,
    policy: BrowserToolPolicy,
) -> None:
    validate_observation(
        observation=observation,
        requested_observation_id=request.observation_id,
        page=page,
        extractor=extractor,
    )

    validate_key(request.key, policy)


def validate_go_back_request(
    request: GoBackRequest,
    *,
    observation: Observation | None,
    page: Page,
    extractor: ObservationExtractor,
) -> None:
    validate_observation(
        observation=observation,
        requested_observation_id=request.observation_id,
        page=page,
        extractor=extractor,
    )