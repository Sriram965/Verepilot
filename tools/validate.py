from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from observation.extract import ObservationExtractor
from observation.models import (
    InteractiveElement,
    Observation,
)

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
        *{
            f"F{i}"
            for i in range(1, 13)
        },
        *{
            f"Digit{i}"
            for i in range(10)
        },
        *{
            f"Key{chr(code)}"
            for code in range(
                ord("A"),
                ord("Z") + 1,
            )
        },
    }
)


@dataclass(frozen=True)
class BrowserToolPolicy:
    """
    Runtime policy for browser tool execution.

    The allowed origins are exact origins, not
    arbitrary domains or URL prefixes.
    """

    allowed_origins: frozenset[str]

    max_text_chars: int = 2_000

    allowed_keys: frozenset[str] = (
        DEFAULT_ALLOWED_KEYS
    )


class ValidationFailure(Exception):
    def __init__(
        self,
        error_type: ToolErrorType,
        message: str,
    ) -> None:
        super().__init__(message)

        self.error_type = error_type
        self.message = message


def parse_tool_url_origin(
    url: str,
) -> str:

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

    if parsed.port is None:
        default_port = {
            "http": 80,
            "https": 443,
        }.get(parsed.scheme)

        if default_port is None:
            raise ValidationFailure(
                ToolErrorType.POLICY_VIOLATION,
                (
                    "Only HTTP and HTTPS URLs "
                    "are supported."
                ),
            )

        port = default_port

    else:
        port = parsed.port

    return (
        f"{parsed.scheme}://"
        f"{parsed.hostname}:{port}"
    )


def validate_allowed_origin(
    url: str,
    policy: BrowserToolPolicy,
) -> None:

    origin = parse_tool_url_origin(url)

    if origin not in policy.allowed_origins:
        raise ValidationFailure(
            ToolErrorType.POLICY_VIOLATION,
            (
                f"Origin '{origin}' is not "
                "allowed by the browser policy."
            ),
        )


def validate_observation(
    *,
    observation: Observation | None,
    requested_observation_id: str,
    page,
    extractor: ObservationExtractor,
) -> None:

    if observation is None:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            (
                "An observation is required "
                "before this action."
            ),
        )

    if (
        observation.observation_id
        != requested_observation_id
    ):
        raise ValidationFailure(
            ToolErrorType.STALE_OBSERVATION,
            (
                "The supplied observation_id "
                "does not match the current "
                "observation."
            ),
        )

    live_page_version = (
        extractor.compute_page_version(page)
    )

    if (
        live_page_version
        != observation.page_version
    ):
        raise ValidationFailure(
            ToolErrorType.STALE_OBSERVATION,
            (
                "The browser page changed "
                "after the observation was "
                "captured."
            ),
        )


def resolve_element(
    *,
    element_id: str,
    observation: Observation,
    page,
) -> object:

    element: InteractiveElement | None = None

    for candidate in (
        observation.interactive_elements
    ):
        if candidate.element_id == element_id:
            element = candidate
            break

    if element is None:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            (
                f"Element '{element_id}' was "
                "not present in the observation."
            ),
        )

    if not element.visible:
        raise ValidationFailure(
            ToolErrorType.ELEMENT_NOT_ACTIONABLE,
            (
                f"Element '{element_id}' was "
                "not visible in the observation."
            ),
        )

    if not element.enabled:
        raise ValidationFailure(
            ToolErrorType.ELEMENT_NOT_ACTIONABLE,
            (
                f"Element '{element_id}' was "
                "disabled in the observation."
            ),
        )

    locator = page.locator(
        element.selector
    )

    if locator.count() != 1:
        raise ValidationFailure(
            ToolErrorType.ELEMENT_NOT_ACTIONABLE,
            (
                f"Element '{element_id}' does "
                "not resolve to exactly one "
                "current DOM element."
            ),
        )

    if not locator.is_visible():
        raise ValidationFailure(
            ToolErrorType.ELEMENT_NOT_ACTIONABLE,
            (
                f"Element '{element_id}' is "
                "not currently visible."
            ),
        )

    if not locator.is_enabled():
        raise ValidationFailure(
            ToolErrorType.ELEMENT_NOT_ACTIONABLE,
            (
                f"Element '{element_id}' is "
                "not currently enabled."
            ),
        )

    return locator


def validate_type_target(
    *,
    element: InteractiveElement,
) -> None:

    if (
        element.tag not in {
            "input",
            "textarea",
        }
        and element.role != "textbox"
    ):
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            (
                f"Element '{element.element_id}' "
                "is not a supported text input."
            ),
        )


def validate_text(
    text: str,
    policy: BrowserToolPolicy,
) -> None:

    if len(text) > policy.max_text_chars:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            (
                "Text exceeds the maximum "
                f"allowed length of "
                f"{policy.max_text_chars}."
            ),
        )

    for char in text:
        if (
            ord(char) < 32
            and char not in {
                "\n",
                "\t",
            }
        ):
            raise ValidationFailure(
                ToolErrorType.VALIDATION_ERROR,
                (
                    "Text contains unsupported "
                    "control characters."
                ),
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
        if (
            part in policy.allowed_keys
            or len(part) == 1
            and part.isprintable()
        ):
            continue

        raise ValidationFailure(
            ToolErrorType.POLICY_VIOLATION,
            (
                f"Key '{part}' is not "
                "allowed by the keyboard policy."
            ),
        )


def validate_navigate_request(
    request: NavigateRequest,
    policy: BrowserToolPolicy,
) -> None:

    validate_allowed_origin(
        request.url,
        policy,
    )


def validate_click_request(
    request: ClickRequest,
    *,
    observation: Observation | None,
    page,
    extractor: ObservationExtractor,
) -> object:

    validate_observation(
        observation=observation,
        requested_observation_id=(
            request.observation_id
        ),
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
    page,
    extractor: ObservationExtractor,
    policy: BrowserToolPolicy,
) -> object:

    validate_observation(
        observation=observation,
        requested_observation_id=(
            request.observation_id
        ),
        page=page,
        extractor=extractor,
    )

    assert observation is not None

    target = None

    for element in (
        observation.interactive_elements
    ):
        if (
            element.element_id
            == request.element_id
        ):
            target = element
            break

    if target is None:
        raise ValidationFailure(
            ToolErrorType.VALIDATION_ERROR,
            (
                f"Element '{request.element_id}' "
                "was not present in the "
                "observation."
            ),
        )

    validate_type_target(
        element=target
    )

    validate_text(
        request.text,
        policy,
    )

    return resolve_element(
        element_id=request.element_id,
        observation=observation,
        page=page,
    )


def validate_press_request(
    request: PressRequest,
    *,
    observation: Observation | None,
    page,
    extractor: ObservationExtractor,
    policy: BrowserToolPolicy,
) -> None:

    validate_observation(
        observation=observation,
        requested_observation_id=(
            request.observation_id
        ),
        page=page,
        extractor=extractor,
    )

    validate_key(
        request.key,
        policy,
    )


def validate_go_back_request(
    request: GoBackRequest,
    *,
    observation: Observation | None,
    page,
    extractor: ObservationExtractor,
) -> None:

    validate_observation(
        observation=observation,
        requested_observation_id=(
            request.observation_id
        ),
        page=page,
        extractor=extractor,
    )
