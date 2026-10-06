from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from typing import Any

from playwright.sync_api import Page

from observation.models import (
    InteractiveElement,
    Observation,
    TruncationInfo,
)


INTERACTIVE_SELECTOR = """
a[href],
button,
input:not([type="hidden"]),
textarea,
select,
summary,
[role="button"],
[role="link"],
[role="textbox"],
[role="checkbox"],
[role="radio"],
[role="combobox"],
[role="listbox"],
[role="tab"],
[role="menuitem"],
[contenteditable="true"]
""".replace(
    "\n",
    " ",
).strip()


ELEMENT_METADATA_SCRIPT = """
element => {
    const clean = value => {
        if (value === null || value === undefined) {
            return null;
        }

        const normalized = String(value)
            .replace(/\\s+/g, " ")
            .trim();

        return normalized || null;
    };

    const tag = element.tagName.toLowerCase();

    const explicitRole = (
        element.getAttribute("role")
        || null
    );

    let inputType = null;

    if (tag === "input") {
        inputType = (
            element.getAttribute("type")
            || "text"
        ).toLowerCase();
    }

    let role = explicitRole;

    if (role === null) {
        if (tag === "a") {
            role = "link";
        } else if (tag === "button") {
            role = "button";
        } else if (tag === "textarea") {
            role = "textbox";
        } else if (tag === "select") {
            role = element.multiple
                ? "listbox"
                : "combobox";
        } else if (tag === "summary") {
            role = "button";
        } else if (tag === "input") {
            const roleMap = {
                checkbox: "checkbox",
                radio: "radio",
                button: "button",
                submit: "button",
                reset: "button",
                image: "button"
            };

            role = (
                roleMap[inputType]
                || "textbox"
            );
        } else if (element.isContentEditable) {
            role = "textbox";
        }
    }

    const ariaLabel = clean(
        element.getAttribute("aria-label")
    );

    const labelledBy = (
        element.getAttribute("aria-labelledby")
        || ""
    );

    let labelledByText = null;

    if (labelledBy) {
        const parts = [];

        for (const id of labelledBy.split(/\\s+/)) {
            const labelledElement =
                document.getElementById(id);

            if (labelledElement) {
                const text = clean(
                    labelledElement.textContent
                );

                if (text) {
                    parts.push(text);
                }
            }
        }

        labelledByText = clean(
            parts.join(" ")
        );
    }

    let labelText = null;

    if (element.labels) {
        const labels = Array.from(
            element.labels
        )
            .map(label => clean(label.textContent))
            .filter(Boolean);

        if (labels.length > 0) {
            labelText = clean(
                labels.join(" ")
            );
        }
    }

    const placeholder = clean(
        element.getAttribute("placeholder")
    );

    const title = clean(
        element.getAttribute("title")
    );

    const alt = clean(
        element.getAttribute("alt")
    );

    const innerText = clean(
        element.innerText
    );

    let value = null;

    if (
        (
            tag === "input"
            || tag === "textarea"
            || tag === "select"
        )
        && inputType !== "password"
    ) {
        value = clean(element.value);
    }

    let name = null;

    if (ariaLabel) {
        name = ariaLabel;
    } else if (labelledByText) {
        name = labelledByText;
    } else if (labelText) {
        name = labelText;
    } else if (innerText) {
        name = innerText;
    } else if (placeholder) {
        name = placeholder;
    } else if (title) {
        name = title;
    } else {
        name = value;
    }

    let href = null;

    if (tag === "a") {
        href = element.href || null;
    }

    let checked = null;

    if (
        inputType === "checkbox"
        || inputType === "radio"
        || role === "checkbox"
        || role === "radio"
    ) {
        checked = Boolean(element.checked);
    }

    let selected = null;

    if (tag === "option") {
        selected = Boolean(element.selected);
    }

    const fieldName = clean(
        element.getAttribute("name")
    );

    return {
        tag,
        role,
        name,
        field_name: fieldName,
        input_type: inputType,
        value,
        placeholder,
        href,
        checked,
        selected,
    };
}
"""


DOM_SELECTOR_SCRIPT = """
element => {
    const escapeIdentifier = value => {
        if (window.CSS && CSS.escape) {
            return CSS.escape(value);
        }

        return String(value).replace(
            /([ !"#$%&'()*+,./:;<=>?@[\\\\\\\\\\]^`{|}~])/g,
            "\\\\$1"
        );
    };

    const id = element.getAttribute("id");

    if (id) {
        const idSelector = "#" + escapeIdentifier(id);

        try {
            if (
                document.querySelectorAll(
                    idSelector
                ).length === 1
            ) {
                return idSelector;
            }
        } catch (_) {
            // Fall through to DOM path.
        }
    }

    const parts = [];
    let current = element;

    while (
        current
        && current.nodeType === Node.ELEMENT_NODE
    ) {
        let selector = current.tagName.toLowerCase();

        const parent = current.parentElement;

        if (!parent) {
            parts.unshift(selector);
            break;
        }

        const sameTagSiblings = Array.from(
            parent.children
        ).filter(
            child =>
                child.tagName
                === current.tagName
        );

        if (sameTagSiblings.length > 1) {
            const index =
                sameTagSiblings.indexOf(current)
                + 1;

            selector += `:nth-of-type(${index})`;
        }

        parts.unshift(selector);

        const candidate = parts.join(" > ");

        try {
            if (
                document.querySelectorAll(
                    candidate
                ).length === 1
            ) {
                return candidate;
            }
        } catch (_) {
            // Continue walking toward the root.
        }

        current = parent;
    }

    return parts.join(" > ");
}
"""


@dataclass(frozen=True)
class ObservationExtractorConfig:
    """
    Limits for the model-facing observation.

    Page versioning is calculated from the full,
    untruncated evidence before these limits are applied.
    """

    max_visible_text_chars: int = 8_000
    max_accessibility_chars: int = 12_000
    max_interactive_elements: int = 100


class ObservationExtractor:
    """
    Converts a Playwright Page into a structured
    VeriPilot Observation.

    This class only observes browser state.
    It does not execute browser actions.
    """

    def __init__(
        self,
        config: ObservationExtractorConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ObservationExtractorConfig()
        )

        self._observation_counter = itertools.count(
            start=1
        )

    def capture(
        self,
        page: Page,
        previous_action_effect: dict[str, Any]
        | None = None,
    ) -> Observation:

        url = page.url
        title = page.title()

        raw_visible_text = (
            self._get_visible_text(page)
        )

        raw_accessibility_tree = (
            self._get_accessibility_tree(page)
        )

        raw_elements = (
            self._get_interactive_elements(page)
        )

        page_version = (
            self._compute_page_version(
                url=url,
                title=title,
                visible_text=raw_visible_text,
                accessibility_tree=(
                    raw_accessibility_tree
                ),
                interactive_elements=raw_elements,
            )
        )

        observation_id = (
            "obs_"
            f"{next(self._observation_counter):06d}"
        )

        (
            visible_text,
            visible_text_truncated,
        ) = self._truncate_text(
            raw_visible_text,
            self.config.max_visible_text_chars,
        )

        (
            accessibility_tree,
            accessibility_tree_truncated,
        ) = self._truncate_text(
            raw_accessibility_tree,
            self.config.max_accessibility_chars,
        )

        (
            interactive_elements,
            interactive_elements_truncated,
        ) = self._limit_interactive_elements(
            raw_elements
        )

        truncation = TruncationInfo(
            visible_text_truncated=(
                visible_text_truncated
            ),
            accessibility_tree_truncated=(
                accessibility_tree_truncated
            ),
            interactive_elements_truncated=(
                interactive_elements_truncated
            ),
            original_interactive_element_count=(
                len(raw_elements)
            ),
        )

        return Observation(
            observation_id=observation_id,
            page_version=page_version,
            url=url,
            title=title,
            visible_text=visible_text,
            accessibility_tree=(
                accessibility_tree
            ),
            interactive_elements=(
                interactive_elements
            ),
            previous_action_effect=(
                previous_action_effect
            ),
            truncation=truncation,
        )

    def compute_page_version(
        self,
        page: Page,
    ) -> str:
        """
        Compute the current browser-state version
        without creating a new Observation.

        This is used by tool validation to determine
        whether the page changed since the agent's
        observation.

        It intentionally does not increment the
        observation counter.
        """

        url = page.url
        title = page.title()

        visible_text = (
            self._get_visible_text(page)
        )

        accessibility_tree = (
            self._get_accessibility_tree(page)
        )

        interactive_elements = (
            self._get_interactive_elements(page)
        )

        return self._compute_page_version(
            url=url,
            title=title,
            visible_text=visible_text,
            accessibility_tree=(
                accessibility_tree
            ),
            interactive_elements=(
                interactive_elements
            ),
        )

    def _get_visible_text(
        self,
        page: Page,
    ) -> str:

        body = page.locator("body")

        if body.count() == 0:
            return ""

        return self._normalize_text(
            body.inner_text()
        )

    def _get_accessibility_tree(
        self,
        page: Page,
    ) -> str:

        snapshot = page.aria_snapshot()

        if snapshot is None:
            return ""

        return snapshot.strip()

    def _get_interactive_elements(
        self,
        page: Page,
    ) -> list[InteractiveElement]:

        locator = page.locator(
            INTERACTIVE_SELECTOR
        )

        elements: list[InteractiveElement] = []

        for index in range(locator.count()):
            current = locator.nth(index)

            if not current.is_visible():
                continue

            metadata = current.evaluate(
                ELEMENT_METADATA_SCRIPT
            )

            selector = current.evaluate(
                DOM_SELECTOR_SCRIPT
            )

            element_id = (
                f"el_{len(elements) + 1:03d}"
            )

            elements.append(
                InteractiveElement(
                    element_id=element_id,
                    selector=selector,
                    tag=metadata["tag"],
                    role=metadata["role"],
                    name=metadata["name"],
                    field_name=metadata[
                        "field_name"
                    ],
                    input_type=metadata[
                        "input_type"
                    ],
                    value=metadata["value"],
                    placeholder=metadata[
                        "placeholder"
                    ],
                    href=metadata["href"],
                    checked=metadata["checked"],
                    selected=metadata["selected"],
                    visible=True,
                    enabled=current.is_enabled(),
                )
            )

        return elements

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        lines = []

        for line in text.splitlines():
            cleaned = re.sub(
                r"\s+",
                " ",
                line,
            ).strip()

            if cleaned:
                lines.append(cleaned)

        return "\n".join(lines)

    @staticmethod
    def _truncate_text(
        text: str,
        limit: int,
    ) -> tuple[str, bool]:

        if limit < 1:
            raise ValueError(
                "observation text limits "
                "must be positive"
            )

        if len(text) <= limit:
            return text, False

        suffix = "\n...[truncated]"

        available = max(
            0,
            limit - len(suffix),
        )

        return (
            text[:available] + suffix,
            True,
        )

    def _limit_interactive_elements(
        self,
        elements: list[InteractiveElement],
    ) -> tuple[
        list[InteractiveElement],
        bool,
    ]:

        limit = (
            self.config.max_interactive_elements
        )

        if limit < 1:
            raise ValueError(
                "max_interactive_elements "
                "must be positive"
            )

        if len(elements) <= limit:
            return elements, False

        return (
            elements[:limit],
            True,
        )

    @staticmethod
    def _compute_page_version(
        *,
        url: str,
        title: str,
        visible_text: str,
        accessibility_tree: str,
        interactive_elements: list[
            InteractiveElement
        ],
    ) -> str:

        element_data = [
            element.model_dump(
                mode="json",
                exclude={
                    "element_id",
                    "selector",
                    "visible",
                },
            )
            for element in interactive_elements
        ]

        canonical = {
            "url": url,
            "title": title,
            "visible_text": visible_text,
            "accessibility_tree": (
                accessibility_tree
            ),
            "interactive_elements": element_data,
        }

        payload = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(
            payload
        ).hexdigest()
