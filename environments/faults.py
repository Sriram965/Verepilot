from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field


class FaultRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str
    occurrence: int = Field(default=1, ge=1)
    response_status: int = Field(
        default=500,
        ge=100,
        le=599,
    )
    response_body: str = "Injected deterministic fault"


@dataclass
class FaultInjector:
    rules: list[FaultRule] = field(default_factory=list)
    _counts: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self._counts.clear()

    def maybe_inject(
        self,
        event: str,
    ) -> FaultRule | None:

        self._counts[event] = (
            self._counts.get(event, 0) + 1
        )

        occurrence = self._counts[event]

        for rule in self.rules:
            if (
                rule.event == event
                and rule.occurrence == occurrence
            ):
                return rule

        return None
