"""Typed agent-adapter protocol shared by runtime-facing adapters."""

from typing import Any, Mapping, Optional, Protocol

from .contracts import (
    AgentCallError,
    AgentCapabilities,
    AgentObservation,
    BillingStatus,
    BudgetDelta,
    ControlOutcome,
    Decision,
)


class AgentAdapter(Protocol):
    capabilities: AgentCapabilities

    def respond(
        self,
        task_id: str,
        observation: Optional[AgentObservation] = None,
    ) -> Mapping[str, Any]:
        ...

    def estimate_next_call(self, task_id: str) -> BudgetDelta:
        ...

    def apply_control(self, decision: Decision) -> ControlOutcome:
        ...
