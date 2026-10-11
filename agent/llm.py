"""
LLM client for the agent.

Everything an experiment might vary is explicit in `LLMConfig`, and every
field here really reaches the API request (checked against the payload
LangChain builds), so a run's configuration can be written into its trace
and compared across systems.

Notes:
- `model` has no default, so experiments never silently change models.
- `temperature=None` leaves the provider default. Some reasoning models
  reject a temperature, so it is only sent when you set it.
- `tool_choice="required"` makes the model call a tool on every turn, so a
  reply with no tool call (which the loop could only treat as an error) is
  not something the model can produce.
- Parallel tool calls are always off: VeriPilot runs one action at a time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from langchain_openai import ChatOpenAI

from agent.tool_definitions import LLM_TOOL_DEFINITIONS


@dataclass(frozen=True)
class LLMConfig:
    model: str

    timeout: float = 60.0
    max_retries: int = 2

    temperature: float | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    tool_choice: Literal["required", "auto"] = "required"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("LLM model name cannot be empty.")

        if self.timeout <= 0:
            raise ValueError("LLM timeout must be positive.")

        if self.max_retries < 0:
            raise ValueError("LLM max_retries cannot be negative.")

        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("LLM temperature must be between 0 and 2.")


def describe_llm_config(config: LLMConfig) -> dict[str, Any]:
    """The configuration as plain data, for the run trace."""

    return asdict(config)


def create_llm(config: LLMConfig) -> ChatOpenAI:
    """
    Create the OpenAI chat-model adapter. Authentication comes from the
    OPENAI_API_KEY environment variable.
    """

    options: dict[str, Any] = {}

    if config.temperature is not None:
        options["temperature"] = config.temperature

    if config.reasoning_effort is not None:
        options["reasoning_effort"] = config.reasoning_effort

    return ChatOpenAI(
        model=config.model,
        timeout=config.timeout,
        max_retries=config.max_retries,
        use_responses_api=True,
        output_version="responses/v1",
        **options,
    )


def create_bound_llm(config: LLMConfig):
    """The LLM with VeriPilot's browser and terminal tools attached."""

    return create_llm(config).bind_tools(
        LLM_TOOL_DEFINITIONS,
        parallel_tool_calls=False,
        tool_choice=config.tool_choice,
    )