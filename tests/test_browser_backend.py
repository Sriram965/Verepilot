from environments.sites.shop import ShopEnvironment
from tools.browser_backend import (
    BrowserBackend,
    BrowserBackendConfig,
)


def test_browser_backend_starts_and_navigates():
    with ShopEnvironment() as environment:
        with BrowserBackend() as backend:
            page = backend.new_page()

            response = backend.navigate(
                page,
                environment.base_url + "/products",
            )

            assert response is not None
            assert response.status == 200

            assert (
                page.title()
                == "VeriPilot Local Shop"
            )

            assert (
                "Laptop A"
                in page.locator("body").inner_text()
            )


def test_browser_backend_defaults_to_headless():
    config = BrowserBackendConfig()

    assert config.headless is True


def test_browser_context_is_isolated_between_backends():
    with ShopEnvironment() as environment:
        with BrowserBackend() as backend_a:
            page_a = backend_a.new_page()

            backend_a.navigate(
                page_a,
                environment.base_url + "/products",
            )

            cookies_a = (
                backend_a.context.cookies()
            )

            session_a = next(
                cookie["value"]
                for cookie in cookies_a
                if cookie["name"]
                == "vp_session"
            )

        with BrowserBackend() as backend_b:
            page_b = backend_b.new_page()

            backend_b.navigate(
                page_b,
                environment.base_url + "/products",
            )

            cookies_b = (
                backend_b.context.cookies()
            )

            session_b = next(
                cookie["value"]
                for cookie in cookies_b
                if cookie["name"]
                == "vp_session"
            )

    assert session_a != session_b


def test_browser_backend_can_be_closed_explicitly():
    backend = BrowserBackend()

    backend.start()

    assert backend.browser is not None
    assert backend.context is not None

    backend.close()

    try:
        _ = backend.browser
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "browser should be unavailable "
            "after close"
        )
