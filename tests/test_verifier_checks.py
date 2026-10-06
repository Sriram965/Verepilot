from dataclasses import dataclass

from verifier.checks import (
    CheckContext,
    evaluate_check,
)
from verifier.spec import (
    AllCheck,
    AnyCheck,
    CartContains,
    ElementText,
    FormSubmitted,
    NotCheck,
    UrlMatches,
)


@dataclass
class FakeProbe:
    url: str
    cart_state: dict[str, int]
    elements: dict[str, str]
    submitted: dict[str, str] | None

    def current_url(self) -> str:
        return self.url

    def cart(self) -> dict[str, int]:
        return dict(self.cart_state)

    def element_text(
        self,
        selector: str,
    ) -> str | None:
        return self.elements.get(selector)

    def submitted_form(
        self,
    ) -> dict[str, str] | None:
        return (
            None
            if self.submitted is None
            else dict(self.submitted)
        )


def context(
    *,
    url="http://localhost/products",
    cart=None,
    elements=None,
    submitted=None,
) -> CheckContext:

    return CheckContext(
        probe=FakeProbe(
            url=url,
            cart_state=cart or {},
            elements=elements or {},
            submitted=submitted,
        )
    )


def test_element_text_passes_when_text_matches():
    result = evaluate_check(
        ElementText(
            type="element_text",
            selector="[data-testid='message']",
            expected_text="Success",
        ),
        context(
            elements={
                "[data-testid='message']": "Success"
            }
        ),
    )

    assert result.passed is True
    assert result.check == "ElementText"


def test_element_text_fails_when_text_does_not_match():
    result = evaluate_check(
        ElementText(
            type="element_text",
            selector="[data-testid='message']",
            expected_text="Success",
        ),
        context(
            elements={
                "[data-testid='message']": "Failed"
            }
        ),
    )

    assert result.passed is False


def test_form_submitted_passes_when_required_fields_match():
    result = evaluate_check(
        FormSubmitted(
            type="form_submitted",
            fields={
                "email": "test@example.com",
                "name": "Sri",
            },
        ),
        context(
            submitted={
                "email": "test@example.com",
                "name": "Sri",
                "message": "Hello",
            }
        ),
    )

    assert result.passed is True
    assert result.check == "FormSubmitted"


def test_form_submitted_fails_when_required_field_is_wrong():
    result = evaluate_check(
        FormSubmitted(
            type="form_submitted",
            fields={
                "email": "test@example.com",
            },
        ),
        context(
            submitted={
                "email": "wrong@example.com",
            }
        ),
    )

    assert result.passed is False


def test_form_submitted_fails_when_nothing_was_submitted():
    result = evaluate_check(
        FormSubmitted(
            type="form_submitted",
            fields={
                "email": "test@example.com",
            },
        ),
        context(
            submitted=None
        ),
    )

    assert result.passed is False


def test_nested_all_any_and_not():
    spec = AllCheck(
        type="all",
        checks=[
            CartContains(
                type="cart_contains",
                product_id="Laptop_A",
                quantity=1,
            ),
            AnyCheck(
                type="any",
                checks=[
                    UrlMatches(
                        type="url_matches",
                        pattern=r"/products$",
                    ),
                    ElementText(
                        type="element_text",
                        selector="#status",
                        expected_text="Ready",
                    ),
                ],
            ),
            NotCheck(
                type="not",
                check=CartContains(
                    type="cart_contains",
                    product_id="Laptop_B",
                    quantity=1,
                ),
            ),
        ],
    )

    result = evaluate_check(
        spec,
        context(
            cart={
                "Laptop_A": 1
            },
            elements={
                "#status": "Ready"
            },
        ),
    )

    assert result.passed is True
    assert result.check == "All"
