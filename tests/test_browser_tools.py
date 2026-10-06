from environments.sites.shop import (
    ShopEnvironment,
)
from observation import ObservationExtractor
from tools.runtime import (
    build_browser_tool_runtime,
)
from tools.schemas import (
    ClickRequest,
    GetPageStateRequest,
    GoBackRequest,
    NavigateRequest,
    PressRequest,
    ToolErrorType,
    TypeRequest,
)


def make_runtime(
    environment: ShopEnvironment,
    backend,
    page,
):
    extractor = ObservationExtractor()

    runtime = build_browser_tool_runtime(
        page=page,
        extractor=extractor,
        allowed_origins=frozenset(
            {
                environment.base_url
            }
        ),
    )

    return runtime


def test_get_page_state_creates_current_observation():
    with ShopEnvironment() as environment:
        from tools.browser_backend import (
            BrowserBackend,
        )

        with BrowserBackend() as backend:
            page = backend.new_page()

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            result = runtime.execute(
                GetPageStateRequest(
                    tool="get_page_state"
                )
            )

            assert result.ok is True
            assert (
                result.data
                is not None
            )


def test_navigate_rejects_external_origin():
    with ShopEnvironment() as environment:
        from tools.browser_backend import (
            BrowserBackend,
        )

        with BrowserBackend() as backend:
            page = backend.new_page()

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            result = runtime.execute(
                NavigateRequest(
                    tool="navigate",
                    url="https://example.com/",
                )
            )

            assert result.ok is False
            assert (
                result.error_type
                == ToolErrorType.POLICY_VIOLATION
            )


def test_click_requires_current_observation():
    with ShopEnvironment() as environment:
        from tools.browser_backend import (
            BrowserBackend,
        )

        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            result = runtime.execute(
                ClickRequest(
                    tool="click",
                    element_id="el_003",
                    observation_id=(
                        "obs_000001"
                    ),
                )
            )

            assert result.ok is False
            assert (
                result.error_type
                == ToolErrorType.VALIDATION_ERROR
            )


def test_click_with_valid_observation_executes():
    with ShopEnvironment() as environment:
        from tools.browser_backend import (
            BrowserBackend,
        )

        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            observation_result = (
                runtime.execute(
                    GetPageStateRequest(
                        tool="get_page_state"
                    )
                )
            )

            observation = (
                observation_result.data
            )

            result = runtime.execute(
                ClickRequest(
                    tool="click",
                    element_id="el_003",
                    observation_id=(
                        observation.observation_id
                    ),
                )
            )

            assert result.ok is True
            assert (
                result.effect.url_changed
                is True
            )
            assert (
                page.url.endswith("/cart")
            )


def test_old_observation_is_rejected_after_navigation():
    with ShopEnvironment() as environment:
        from tools.browser_backend import (
            BrowserBackend,
        )

        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            observation_result = (
                runtime.execute(
                    GetPageStateRequest(
                        tool="get_page_state"
                    )
                )
            )

            observation = (
                observation_result.data
            )

            page.goto(
                environment.base_url
                + "/cart"
            )

            result = runtime.execute(
                ClickRequest(
                    tool="click",
                    element_id="el_003",
                    observation_id=(
                        observation.observation_id
                    ),
                )
            )

            assert result.ok is False
            assert (
                result.error_type
                == ToolErrorType.STALE_OBSERVATION
            )


def test_invalid_element_id_is_rejected():
    with ShopEnvironment() as environment:
        from tools.browser_backend import (
            BrowserBackend,
        )

        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            observation_result = (
                runtime.execute(
                    GetPageStateRequest(
                        tool="get_page_state"
                    )
                )
            )

            observation = (
                observation_result.data
            )

            result = runtime.execute(
                ClickRequest(
                    tool="click",
                    element_id="el_999",
                    observation_id=(
                        observation.observation_id
                    ),
                )
            )

            assert result.ok is False
            assert (
                result.error_type
                == ToolErrorType.VALIDATION_ERROR
            )


def test_type_fills_text_input():
    with ShopEnvironment() as environment:
        from tools.browser_backend import (
            BrowserBackend,
        )

        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            observation_result = (
                runtime.execute(
                    GetPageStateRequest(
                        tool="get_page_state"
                    )
                )
            )

            observation = (
                observation_result.data
            )

            result = runtime.execute(
                TypeRequest(
                    tool="type",
                    element_id="el_002",
                    text="3",
                    observation_id=(
                        observation.observation_id
                    ),
                )
            )

            assert result.ok is True

            value = page.locator(
                "input[name='quantity']"
            ).first.input_value()

            assert value == "3"


def test_same_observation_cannot_be_used_after_type():
    with ShopEnvironment() as environment:
        from tools.browser_backend import (
            BrowserBackend,
        )

        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            observation_result = (
                runtime.execute(
                    GetPageStateRequest(
                        tool="get_page_state"
                    )
                )
            )

            observation = (
                observation_result.data
            )

            first = runtime.execute(
                TypeRequest(
                    tool="type",
                    element_id="el_002",
                    text="3",
                    observation_id=(
                        observation.observation_id
                    ),
                )
            )

            assert first.ok is True

            second = runtime.execute(
                TypeRequest(
                    tool="type",
                    element_id="el_002",
                    text="4",
                    observation_id=(
                        observation.observation_id
                    ),
                )
            )

            assert second.ok is False
            assert (
                second.error_type
                == ToolErrorType.STALE_OBSERVATION
            )


def test_press_rejects_disallowed_key():
    with ShopEnvironment() as environment:
        from tools.browser_backend import (
            BrowserBackend,
        )

        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            observation_result = (
                runtime.execute(
                    GetPageStateRequest(
                        tool="get_page_state"
                    )
                )
            )

            observation = (
                observation_result.data
            )

            result = runtime.execute(
                PressRequest(
                    tool="press",
                    key="Control+DefinitelyNotAKey",
                    observation_id=(
                        observation.observation_id
                    ),
                )
            )

            assert result.ok is False
            assert (
                result.error_type
                == ToolErrorType.POLICY_VIOLATION
            )


def test_go_back_requires_observation_after_navigation():
    with ShopEnvironment() as environment:
        from tools.browser_backend import (
            BrowserBackend,
        )

        with BrowserBackend() as backend:
            page = backend.new_page()

            backend.navigate(
                page,
                environment.base_url
                + "/products",
            )

            runtime = make_runtime(
                environment,
                backend,
                page,
            )

            runtime.execute(
                GetPageStateRequest(
                    tool="get_page_state"
                )
            )

            runtime.execute(
                NavigateRequest(
                    tool="navigate",
                    url=(
                        environment.base_url
                        + "/cart"
                    ),
                )
            )

            old_observation = (
                runtime.current_observation
            )

            result = runtime.execute(
                GoBackRequest(
                    tool="go_back",
                    observation_id=(
                        old_observation.observation_id
                    ),
                )
            )

            assert result.ok is False
            assert (
                result.error_type
                == ToolErrorType.STALE_OBSERVATION
            )
