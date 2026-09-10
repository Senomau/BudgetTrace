"""Deterministic, auditable features extracted from event facts."""

import math
from typing import Dict, Iterable, List, Optional, Set

from .contracts import BudgetLimits, Event, FeatureSnapshot


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def extract_features(
    events: Iterable[Event],
    score_history: List[float],
    budget: BudgetLimits,
    *,
    late_breakthrough_guard: bool = False,
    success: bool = False,
    safety_blocked: bool = False,
    can_replan: bool = False,
    cheap_route_safe: bool = False,
    window_index: int = 0,
    previous_feedback_ids: Optional[Set[str]] = None,
) -> FeatureSnapshot:
    """Build a feature snapshot without inspecting model hidden reasoning."""

    event_list = list(events)
    missing: Dict[str, str] = {}

    score_slope = None
    if len(score_history) >= 2:
        score_slope = (score_history[-1] - score_history[0]) / (len(score_history) - 1)
    else:
        missing["score_slope"] = "need_at_least_two_score_points"

    feedback_ids = [
        str(event.payload["feedback_id"])
        for event in event_list
        if event.event_type == "feedback" and event.payload.get("feedback_id")
    ]
    feedback_novelty = None
    if feedback_ids:
        baseline = previous_feedback_ids or set()
        candidates = [feedback_id for feedback_id in feedback_ids if feedback_id not in baseline]
        counts = {feedback_id: candidates.count(feedback_id) for feedback_id in set(candidates)}
        novel_ids = sum(1 for count in counts.values() if count == 1)
        feedback_novelty = novel_ids / len(feedback_ids)
    else:
        missing["feedback_novelty"] = "no_structured_feedback_events"

    tool_keys = [
        (str(event.payload["tool"]), str(event.payload["args_hash"]))
        for event in event_list
        if event.event_type == "tool_call"
        and event.payload.get("tool")
        and event.payload.get("args_hash")
    ]
    tool_repeat_rate = None
    if tool_keys:
        tool_repeat_rate = (len(tool_keys) - len(set(tool_keys))) / len(tool_keys)
    else:
        missing["tool_repeat_rate"] = "no_tool_call_events"

    action_events = [
        event
        for event in event_list
        if event.event_type in {"model_call", "tool_call", "tool_result", "error"}
    ]
    error_events = [
        event
        for event in event_list
        if event.event_type == "error" or event.payload.get("ok") is False
    ]
    error_rate = _ratio(len(error_events), len(action_events))
    if error_rate is None:
        missing["error_rate"] = "no_action_events"

    progress_gap = None
    if score_history:
        progress_gap = max(0.0, 1.0 - score_history[-1])
    else:
        missing["progress_gap"] = "no_score_points"

    context_ratios = []
    invalid_context = False
    for event in event_list:
        if "context_ratio" not in event.payload:
            continue
        try:
            context_ratios.append(float(event.payload["context_ratio"]))
        except (TypeError, ValueError):
            invalid_context = True
        else:
            if not math.isfinite(context_ratios[-1]):
                context_ratios.pop()
                invalid_context = True
    context_pressure = max(context_ratios) if context_ratios else None
    if context_pressure is None:
        missing["context_pressure"] = (
            "no_valid_context_ratio_events" if invalid_context else "no_context_ratio_events"
        )

    total_steps = sum(event.budget_delta.steps for event in event_list)
    budget_burn_rate = total_steps / budget.steps if budget.steps else None
    if budget_burn_rate is None:
        missing["budget_burn_rate"] = "budget_steps_is_zero"

    new_feedback = bool(feedback_novelty and feedback_novelty > 0)
    high_repeat_no_novelty = bool(
        tool_repeat_rate is not None and tool_repeat_rate >= 0.5 and not new_feedback
    )

    return FeatureSnapshot(
        score_slope=score_slope,
        feedback_novelty=feedback_novelty,
        tool_repeat_rate=tool_repeat_rate,
        error_rate=error_rate,
        progress_gap=progress_gap,
        context_pressure=context_pressure,
        budget_burn_rate=budget_burn_rate,
        late_breakthrough_guard=late_breakthrough_guard,
        success=success,
        safety_blocked=safety_blocked,
        budget_exhausted=any(
            (
                total_steps >= budget.steps,
                sum(event.budget_delta.tokens for event in event_list) >= budget.tokens,
                sum(event.budget_delta.time_ms for event in event_list) >= budget.time_ms,
                sum(event.budget_delta.cost_micros for event in event_list) >= budget.cost_micros,
            )
        ),
        new_feedback=new_feedback,
        high_repeat_no_novelty=high_repeat_no_novelty,
        cheap_route_safe=cheap_route_safe,
        can_replan=can_replan,
        missing_reason=missing,
        window_index=window_index,
    )
