"""Deterministic model and scorer fixtures; no network access."""

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .contracts import AgentCapabilities, AgentObservation, BudgetDelta, ControlOutcome, Decision, Event


class ScriptedLLM:
    capabilities = AgentCapabilities(supports_replan=True, supports_route_cheaper=False)

    def __init__(self, scripts: Mapping[str, Iterable[Mapping[str, object]]]) -> None:
        self._scripts: Dict[str, List[Mapping[str, object]]] = {
            task_id: list(actions) for task_id, actions in scripts.items()
        }
        self._positions: Dict[str, int] = {task_id: 0 for task_id in self._scripts}
        self.observations: List[AgentObservation] = []

    def respond(
        self,
        task_id: str,
        observation: Optional[AgentObservation] = None,
    ) -> Mapping[str, object]:
        if observation is not None:
            self.observations.append(observation)
        actions = self._scripts.get(task_id, [])
        position = self._positions.get(task_id, 0)
        if position >= len(actions):
            return {"kind": "script_exhausted"}
        self._positions[task_id] = position + 1
        return actions[position]

    def estimate_next_call(self, task_id: str) -> BudgetDelta:
        return BudgetDelta(steps=1)

    def apply_control(self, decision: Decision) -> ControlOutcome:
        return ControlOutcome(
            applied=decision.action in ("replan", "route_cheaper", "stop"),
            action=decision.action,
            reason_code=decision.reason_code,
            metadata=dict(decision.metadata),
        )


class FixtureScorer:
    def evaluate(self, events: Iterable[Event]) -> tuple:
        scores = [
            float(event.payload["score"])
            for event in events
            if event.event_type == "feedback" and "score" in event.payload
        ]
        score = max(scores) if scores else 0.0
        return score, score >= 1.0


class BusinessContractScorer:
    """Score explicit business evidence declared by a deterministic task contract."""

    scored = True

    def __init__(
        self,
        required_tools: Sequence[str] = (),
        required_feedback_ids: Sequence[str] = (),
        min_score: float = 1.0,
        version: str = "business-contract-v1",
    ) -> None:
        self.required_tools = tuple(str(name) for name in required_tools)
        self.required_feedback_ids = tuple(str(value) for value in required_feedback_ids)
        self.min_score = float(min_score)
        self.scorer_version = version

    def evaluate(self, events: Iterable[Event]) -> tuple:
        event_list = list(events)
        successful_tools = [
            str(event.payload.get("tool", ""))
            for event in event_list
            if event.event_type == "tool_result" and event.payload.get("ok") is True
        ]
        feedback_ids = {
            str(event.payload.get("feedback_id", ""))
            for event in event_list
            if event.event_type == "feedback"
        }
        scores = [
            float(event.payload["score"])
            for event in event_list
            if event.event_type == "feedback" and "score" in event.payload
        ]

        tool_index = 0
        matched_tools = 0
        for required in self.required_tools:
            try:
                tool_index = successful_tools.index(required, tool_index) + 1
            except ValueError:
                continue
            matched_tools += 1
        matched_feedback = sum(
            1 for feedback_id in self.required_feedback_ids if feedback_id in feedback_ids
        )
        criteria_count = len(self.required_tools) + len(self.required_feedback_ids)
        matched_count = matched_tools + matched_feedback
        evidence_score = matched_count / criteria_count if criteria_count else 0.0
        max_score = max(scores) if scores else 0.0
        success = (
            matched_count == criteria_count
            and bool(criteria_count)
            and max_score >= self.min_score
        )
        return evidence_score, success
