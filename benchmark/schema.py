"""
Benchmark task contract for VeriPilot.

This module defines the canonical schema used to describe a benchmark task
and its evaluation configuration.

A TaskSpec is the benchmark-level representation of a task. It contains the
natural-language instruction, environment entry point, step budget,
declarative success specification, optional fault configuration, and
evaluation/oracle configuration.

Important boundary:
- TaskSpec is a benchmark/evaluation object, not necessarily the exact object
  exposed to the agent.
- The agent receives only the information allowed by its baseline/system.
- Hidden oracle configuration and authoritative evaluation state must never
  be exposed to the agent during execution.
- Fault configuration is consumed by the benchmark environment to inject
  deterministic, event-triggered faults; it is not an agent instruction.
- This schema must remain environment-agnostic and must not contain
  task-specific fields such as cart-specific state.

Keep this module limited to the benchmark contract. Environment behavior,
verification logic, fault execution, and oracle evaluation belong in their
respective modules.
"""


from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from environments.faults import FaultRule
from verifier.spec import CheckSpec


class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    no_collateral_changes: bool = True


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    instruction: str
    start_url: str
    max_steps: int = Field(gt=0)

    success: CheckSpec

    fault_config: list[FaultRule] = Field(
        default_factory=list
    )

    oracle: OracleConfig
