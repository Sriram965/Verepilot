from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from agent.tool_definitions import LLM_TOOL_DEFINITIONS


@dataclass(frozen=True)
class LLMConfig:
    """
    Configuration for the Phase 3 LLM client.

    The model name is supplied explicitly so experiments
    do not silently change models.
    """

    model: str

    timeout: float = 60.0

    max_retries: int = 2


def create_llm(
    config: LLMConfig,
) -> ChatOpenAI:
    """
    Create the OpenAI chat-model adapter.

    API authentication is taken from OPENAI_API_KEY
    by the underlying LangChain OpenAI integration.
    """

    if not config.model.strip():
        raise ValueError(
            "LLM model name cannot be empty."
        )

    return ChatOpenAI(
        model=config.model,
        timeout=config.timeout,
        max_retries=config.max_retries,
        use_responses_api=True,
        output_version="responses/v1",
    )


def create_bound_llm(
    config: LLMConfig,
) -> ChatOpenAI:
    """
    Create an LLM configured with VeriPilot's
    model-facing browser/terminal tool schemas.

    Parallel tool calls are explicitly disabled.
    VeriPilot executes one action at a time.
    """

    model = create_llm(config)

    return model.bind_tools(
        LLM_TOOL_DEFINITIONS,
        parallel_tool_calls=False,
    )