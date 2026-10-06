from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)

from agent.tool_calls import (
    parse_browser_tool_call,
)
from agent.tool_definitions import (
    LLM_TOOL_DEFINITIONS,
)
from tools.schemas import (
    ClickRequest,
    NavigateRequest,
)


def test_all_expected_tools_are_exposed_to_llm():
    names = {
        definition["function"]["name"]
        for definition
        in LLM_TOOL_DEFINITIONS
    }

    assert names == {
        "navigate",
        "get_page_state",
        "click",
        "type",
        "press",
        "go_back",
        "finish",
        "give_up",
    }


def test_click_tool_call_maps_to_existing_request_model():
    request = parse_browser_tool_call(
        "click",
        {
            "element_id": "el_003",
            "observation_id": "obs_000001",
        },
    )

    assert isinstance(
        request,
        ClickRequest,
    )

    assert (
        request.element_id
        == "el_003"
    )

    assert (
        request.observation_id
        == "obs_000001"
    )


def test_navigate_tool_call_maps_to_existing_request_model():
    request = parse_browser_tool_call(
        "navigate",
        {
            "url": (
                "http://127.0.0.1:12345/products"
            ),
        },
    )

    assert isinstance(
        request,
        NavigateRequest,
    )

    assert (
        request.url.endswith("/products")
    )


def test_unknown_browser_tool_is_rejected():
    try:
        parse_browser_tool_call(
            "shell",
            {},
        )
    except ValueError as exc:
        assert (
            "not a browser tool"
            in str(exc)
        )
    else:
        raise AssertionError(
            "unknown browser tool should fail"
        )


def test_fake_model_produces_standardized_tool_call():
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "click",
                        "args": {
                            "element_id": "el_003",
                            "observation_id": (
                                "obs_000001"
                            ),
                        },
                        "id": "call_001",
                    }
                ],
            )
        ]
    )

    response = model.invoke(
        "perform the task"
    )

    assert len(response.tool_calls) == 1

    tool_call = response.tool_calls[0]

    assert (
        tool_call["name"]
        == "click"
    )

    assert (
        tool_call["args"]["element_id"]
        == "el_003"
    )

    assert (
        tool_call["args"]["observation_id"]
        == "obs_000001"
    )