"""
Turns a Playwright page into a structured VeriPilot Observation.

Speed: everything about the page's elements (visibility, enabled state, name,
role, selector...) is collected in ONE browser call. Asking the browser about
each element separately costs several round trips per element, which made a
120-element page take about 5 seconds to observe.

Page version: `page_version` is a fingerprint of url, title, visible text and
the interactive elements (everything except the element ids, selectors and
visibility flags, which are derived). The runtime recomputes it before and
after every action, so it deliberately leaves out the accessibility tree,
which is the slowest part to read and is derived from the same DOM. The tree
is still included in the Observation shown to the model.

Limits: elements inside shadow roots and iframes are not observed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from typing import Any

from playwright.sync_api import Error as PlaywrightError
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


# One call collects the title, the visible text and every visible interactive
# element. It reuses the two scripts above, so each element is described
# exactly as before.
PAGE_SCRIPT = (
    "(selector) => {\n"
    "  const metadata = " + ELEMENT_METADATA_SCRIPT + ";\n"
    "  const selectorFor = " + DOM_SELECTOR_SCRIPT + ";\n"
    + """
  // Same meaning as Playwright's is_visible(): rendered (this also rules out
  // content inside a closed <details>), not visibility:hidden, non-empty box.
  const isVisible = element => {
    if (!element.checkVisibility({ visibilityProperty: true })) {
      return false;
    }

    const rect = element.getBoundingClientRect();

    return rect.width > 0 && rect.height > 0;
  };

  // Same meaning as Playwright's is_enabled().
  const isEnabled = element => (
    !element.matches(":disabled")
    && !element.closest('[aria-disabled="true"]')
  );

  const elements = [];

  for (const element of document.querySelectorAll(selector)) {
    if (!isVisible(element)) {
      continue;
    }

    const described = metadata(element);
    described.selector = selectorFor(element);
    described.enabled = isEnabled(element);
    elements.push(described);
  }

  return {
    title: document.title,
    text: document.body ? document.body.innerText : "",
    elements,
  };
}
"""
)
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


@dataclass(frozen=True)
class _PageRead:
    url: str
    title: str
    visible_text: str
    interactive_elements: list[InteractiveElement]


class ObservationExtractor:
    """
    Converts a Playwright Page into a structured VeriPilot Observation.

    This class only observes browser state. It does not execute actions.
    """

    def __init__(
        self,
        config: ObservationExtractorConfig | None = None,
    ) -> None:
        self.config = (
            config if config is not None else ObservationExtractorConfig()
        )
        self._observation_counter = itertools.count(start=1)

    def capture(
        self,
        page: Page,
        previous_action_effect: dict[str, Any] | None = None,
    ) -> Observation:
        read = self._read_page(page)
        raw_accessibility_tree = self._get_accessibility_tree(page)

        page_version = self._compute_page_version(
            url=read.url,
            title=read.title,
            visible_text=read.visible_text,
            interactive_elements=read.interactive_elements,
        )

        observation_id = f"obs_{next(self._observation_counter):06d}"

        visible_text, visible_text_truncated = self._truncate_text(
            read.visible_text,
            self.config.max_visible_text_chars,
        )
        accessibility_tree, accessibility_tree_truncated = (
            self._truncate_text(
                raw_accessibility_tree,
                self.config.max_accessibility_chars,
            )
        )
        interactive_elements, interactive_elements_truncated = (
            self._limit_interactive_elements(read.interactive_elements)
        )

        return Observation(
            observation_id=observation_id,
            page_version=page_version,
            url=read.url,
            title=read.title,
            visible_text=visible_text,
            accessibility_tree=accessibility_tree,
            interactive_elements=interactive_elements,
            previous_action_effect=previous_action_effect,
            truncation=TruncationInfo(
                visible_text_truncated=visible_text_truncated,
                accessibility_tree_truncated=accessibility_tree_truncated,
                interactive_elements_truncated=interactive_elements_truncated,
                original_interactive_element_count=len(
                    read.interactive_elements
                ),
            ),
        )

    def compute_page_version(self, page: Page) -> str:
        """
        Current browser-state version, without creating an Observation or
        advancing the observation counter. Used by tool validation to detect
        that the page changed since the agent last looked.
        """

        read = self._read_page(page)

        return self._compute_page_version(
            url=read.url,
            title=read.title,
            visible_text=read.visible_text,
            interactive_elements=read.interactive_elements,
        )

    def _read_page(self, page: Page) -> _PageRead:
        raw = self._evaluate_page(page)

        elements = [
            InteractiveElement(
                element_id=f"el_{index:03d}",
                visible=True,
                **element,
            )
            for index, element in enumerate(raw["elements"], start=1)
        ]

        return _PageRead(
            url=page.url,
            title=raw["title"],
            visible_text=self._normalize_text(raw["text"]),
            interactive_elements=elements,
        )

    @staticmethod
    def _evaluate_page(page: Page) -> dict[str, Any]:
        """Run the page script; retry once if a navigation interrupts it."""

        for attempt in range(2):
            try:
                return page.evaluate(PAGE_SCRIPT, INTERACTIVE_SELECTOR)
            except PlaywrightError as exc:
                interrupted = "context was destroyed" in str(exc).lower()

                if attempt == 1 or not interrupted:
                    raise

                page.wait_for_load_state("domcontentloaded")

        raise AssertionError("unreachable")

    @staticmethod
    def _get_accessibility_tree(page: Page) -> str:
        snapshot = page.aria_snapshot()

        return snapshot.strip() if snapshot else ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = []

        for line in text.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip()

            if cleaned:
                lines.append(cleaned)

        return "\n".join(lines)

    @staticmethod
    def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
        if limit < 1:
            raise ValueError("observation text limits must be positive")

        if len(text) <= limit:
            return text, False

        suffix = "\n...[truncated]"
        available = max(0, limit - len(suffix))

        return text[:available] + suffix, True

    def _limit_interactive_elements(
        self,
        elements: list[InteractiveElement],
    ) -> tuple[list[InteractiveElement], bool]:
        limit = self.config.max_interactive_elements

        if limit < 1:
            raise ValueError("max_interactive_elements must be positive")

        if len(elements) <= limit:
            return elements, False

        return elements[:limit], True

    @staticmethod
    def _compute_page_version(
        *,
        url: str,
        title: str,
        visible_text: str,
        interactive_elements: list[InteractiveElement],
    ) -> str:
        canonical = {
            "url": url,
            "title": title,
            "visible_text": visible_text,
            "interactive_elements": [
                element.model_dump(
                    mode="json",
                    exclude={"element_id", "selector", "visible"},
                )
                for element in interactive_elements
            ],
        }

        payload = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()