"""Versioned contracts shared by the runtime, policy and replay layers."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from pydantic.dataclasses import dataclass


Action = Literal["continue", "replan", "route_cheaper", "stop"]
BillingStatus = Literal["measured", "estimated_from_provider_usage", "billed", "unknown"]
SideEffectLevel = Literal["L0", "L1", "L2", "L3"]


class BudgetLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: StrictInt = Field(gt=0)
    tokens: StrictInt = Field(gt=0)
    time_ms: StrictInt = Field(gt=0)
    cost_micros: StrictInt = Field(gt=0)


class BudgetDelta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: StrictInt = Field(default=0, ge=0)
    tokens: StrictInt = Field(default=0, ge=0)
    time_ms: StrictInt = Field(default=0, ge=0)
    cost_micros: StrictInt = Field(default=0, ge=0)


class AgentObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)
    schema_version: StrictInt = Field(default=2, ge=1)


class AgentCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    supports_replan: bool
    supports_route_cheaper: bool


class ControlOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    applied: bool
    action: Action
    reason_code: str = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class AgentCallError(Exception):
    code: str
    task_id: str
    billing_status: BillingStatus
    message: str

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return self.message


class TaskContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    workspace_root: str = Field(min_length=1)
    allowed_tools: List[str] = Field(min_length=1)
    tool_levels: Dict[str, SideEffectLevel] = Field(default_factory=dict)
    budgets: BudgetLimits
    success_condition: str = Field(min_length=1)
    scorer_version: str = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("allowed_tools")
    @classmethod
    def unique_tool_names(cls, value: List[str]) -> List[str]:
        if any(not name.strip() for name in value):
            raise ValueError("allowed_tools cannot contain empty names")
        if len(set(value)) != len(value):
            raise ValueError("allowed_tools must be unique")
        return value

    @model_validator(mode="after")
    def tool_levels_are_allowlisted(self) -> "TaskContract":
        unknown = set(self.tool_levels) - set(self.allowed_tools)
        if unknown:
            raise ValueError("tool_levels contains non-allowlisted tools: " + ", ".join(sorted(unknown)))
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    budget_window_steps: StrictInt = Field(gt=0)
    offline: bool = True
    seed: StrictInt = 0


class BudgetSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    limits: BudgetLimits
    steps_used: StrictInt = Field(default=0, ge=0)
    tokens_used: StrictInt = Field(default=0, ge=0)
    time_ms_used: StrictInt = Field(default=0, ge=0)
    cost_micros_used: StrictInt = Field(default=0, ge=0)

    @property
    def remaining_steps(self) -> int:
        return max(0, self.limits.steps - self.steps_used)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.limits.tokens - self.tokens_used)

    @property
    def remaining_time_ms(self) -> int:
        return max(0, self.limits.time_ms - self.time_ms_used)

    @property
    def remaining_cost_micros(self) -> int:
        return max(0, self.limits.cost_micros - self.cost_micros_used)

    @property
    def exhausted(self) -> bool:
        return any(
            (
                self.steps_used >= self.limits.steps,
                self.tokens_used >= self.limits.tokens,
                self.time_ms_used >= self.limits.time_ms,
                self.cost_micros_used >= self.limits.cost_micros,
            )
        )


class Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = Field(default=1, ge=1)
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence: StrictInt = Field(gt=0)
    event_type: str = Field(min_length=1)
    occurred_at: str = Field(min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)
    budget_delta: BudgetDelta = Field(default_factory=BudgetDelta)


class FeatureSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    score_slope: Optional[float] = None
    feedback_novelty: Optional[float] = None
    tool_repeat_rate: Optional[float] = None
    error_rate: Optional[float] = None
    progress_gap: Optional[float] = None
    context_pressure: Optional[float] = None
    budget_burn_rate: Optional[float] = None
    late_breakthrough_guard: bool = False
    success: bool = False
    safety_blocked: bool = False
    budget_exhausted: bool = False
    new_feedback: bool = False
    high_repeat_no_novelty: bool = False
    cheap_route_safe: bool = False
    can_replan: bool = False
    missing_reason: Dict[str, str] = Field(default_factory=dict)
    window_index: int = Field(default=0, ge=0)


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Action
    reason_code: str = Field(min_length=1)
    budget_window: StrictInt = Field(ge=0)
    policy_version: str = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunCard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    terminal_reason: str = Field(min_length=1)
    success: bool
    final_score: float = 0.0
    event_count: StrictInt = Field(ge=0)
    budget: BudgetSnapshot
    over_budget: bool = False
    over_budget_dimensions: List[str] = Field(default_factory=list)
    billing_status: BillingStatus = "measured"
    reserved_budget: BudgetDelta = Field(default_factory=BudgetDelta)
    observed_budget: BudgetDelta = Field(default_factory=BudgetDelta)
    provenance: Literal["offline", "live"] = "offline"
    journal_jsonl: str = ""
