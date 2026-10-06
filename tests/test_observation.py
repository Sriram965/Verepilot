from environments.sites.shop import ShopEnvironment
from observation.extract import (
    ObservationExtractor,
    ObservationExtractorConfig,
)
from tools.browser_backend import BrowserBackend


def test_observation_contains_core_browser_state():
    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            extractor = ObservationExtractor()

            observation = extractor.capture(
                page
            )

            assert (
                observation.url
                == page.url
            )

            assert (
                observation.title
                == "VeriPilot Local Shop"
            )

            assert (
                "Laptop A"
                in observation.visible_text
            )

            assert (
                "Laptop A"
                in observation.accessibility_tree
            )

            assert (
                len(
                    observation.interactive_elements
                )
                > 0
            )


def test_interactive_elements_have_unique_observation_scoped_ids():
    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            extractor = ObservationExtractor()

            observation = extractor.capture(
                page
            )

            ids = [
                element.element_id
                for element
                in observation.interactive_elements
            ]

            assert len(ids) == len(
                set(ids)
            )

            assert ids[0] == "el_001"


def test_shop_observation_contains_useful_element_metadata():
    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            extractor = ObservationExtractor()

            observation = extractor.capture(
                page
            )

            buttons = [
                element
                for element
                in observation.interactive_elements
                if element.role == "button"
            ]

            assert len(buttons) >= 3

            add_to_cart_buttons = [
                button
                for button in buttons
                if button.name
                == "Add to cart"
            ]

            assert (
                len(add_to_cart_buttons)
                == 3
            )

            quantity_inputs = [
                element
                for element
                in observation.interactive_elements
                if element.tag == "input"
            ]

            assert (
                len(quantity_inputs)
                == 3
            )


def test_same_page_has_same_page_version_but_new_observation_id():
    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            extractor = ObservationExtractor()

            first = extractor.capture(
                page
            )

            second = extractor.capture(
                page
            )

            assert (
                first.observation_id
                != second.observation_id
            )

            assert (
                first.page_version
                == second.page_version
            )


def test_navigation_changes_page_version():
    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            extractor = ObservationExtractor()

            products = extractor.capture(
                page
            )

            backend.navigate(
                page,
                environment.base_url
                + "/cart",
            )

            cart = extractor.capture(
                page
            )

            assert (
                products.observation_id
                != cart.observation_id
            )

            assert (
                products.page_version
                != cart.page_version
            )

            assert products.url != cart.url


def test_previous_action_effect_is_preserved():
    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            extractor = ObservationExtractor()

            effect = {
                "tool": "navigate",
                "ok": True,
                "url_changed": True,
                "dom_changed": True,
            }

            observation = extractor.capture(
                page,
                previous_action_effect=effect,
            )

            assert (
                observation.previous_action_effect
                == effect
            )


def test_observation_limits_are_explicit():
    config = ObservationExtractorConfig(
        max_visible_text_chars=40,
        max_accessibility_chars=50,
        max_interactive_elements=2,
    )

    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            extractor = ObservationExtractor(
                config
            )

            observation = extractor.capture(
                page
            )

            assert (
                len(observation.visible_text)
                <= 40
            )

            assert (
                len(
                    observation.accessibility_tree
                )
                <= 50
            )

            assert (
                len(
                    observation.interactive_elements
                )
                == 2
            )

            assert (
                observation.truncation
                .visible_text_truncated
                is True
            )

            assert (
                observation.truncation
                .accessibility_tree_truncated
                is True
            )

            assert (
                observation.truncation
                .interactive_elements_truncated
                is True
            )

            assert (
                observation.truncation
                .original_interactive_element_count
                > 2
            )


def test_interactive_element_selectors_ground_exact_elements():
    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            extractor = ObservationExtractor()

            observation = extractor.capture(
                page
            )

            add_to_cart_buttons = [
                element
                for element
                in observation.interactive_elements
                if element.role == "button"
                and element.name
                == "Add to cart"
            ]

            assert (
                len(add_to_cart_buttons)
                == 3
            )

            for element in add_to_cart_buttons:
                matches = page.locator(
                    element.selector
                )

                assert matches.count() == 1

                assert (
                    matches.is_visible()
                    is True
                )

            first = add_to_cart_buttons[0]
            second = add_to_cart_buttons[1]
            third = add_to_cart_buttons[2]

            assert (
                first.selector
                != second.selector
            )

            assert (
                second.selector
                != third.selector
            )

            assert (
                first.selector
                != third.selector
            )
