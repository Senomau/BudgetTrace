"""Small deterministic policy comparison harness."""

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional

from .contracts import BudgetLimits, Decision, FeatureSnapshot, RunConfig, TaskContract
from .fixtures import BusinessContractScorer, FixtureScorer, ScriptedLLM
from .policy import BudgetController
from .runtime import Runtime
from .security import ToolRegistry

DEFAULT_POLICIES = ["fixed_steps", "fixed_budget", "naive_stop", "budgettrace_rules_v2"]


class _AlwaysContinueController(BudgetController):
    def __init__(self, policy_version: str) -> None:
        super().__init__(policy_version)
        self.stop_on_success = False

    def choose(self, snapshot: FeatureSnapshot) -> Decision:
        if snapshot.safety_blocked:
            return self._decision("stop", "safety_denied", snapshot)
        if snapshot.budget_exhausted:
            return self._decision("stop", "budget_exhausted", snapshot)
        return self._decision("continue", "fixed_policy_window", snapshot)


class _NaiveStopController(BudgetController):
    def choose(self, snapshot: FeatureSnapshot) -> Decision:
        if snapshot.safety_blocked:
            return self._decision("stop", "safety_denied", snapshot)
        if snapshot.success:
            return self._decision("stop", "success", snapshot)
        if snapshot.budget_exhausted:
            return self._decision("stop", "budget_exhausted", snapshot)
        if snapshot.window_index >= 1:
            return self._decision("stop", "naive_no_progress", snapshot)
        return self._decision("continue", "naive_first_window", snapshot)


def _controller_for(policy: str) -> BudgetController:
    if policy in {"fixed_steps", "fixed_budget"}:
        return _AlwaysContinueController(policy)
    if policy == "naive_stop":
        return _NaiveStopController(policy)
    if policy in {"budgettrace_rules_v1", "budgettrace_rules_v2"}:
        return BudgetController(policy)
    raise ValueError(f"unknown policy: {policy}")


def _safety_ceilings(limits: BudgetLimits) -> Dict[str, int]:
    return {
        "steps": limits.steps,
        "tokens": limits.tokens,
        "time_ms": limits.time_ms,
        "cost_micros": limits.cost_micros,
    }


def _active_limit(policy: str, limits: BudgetLimits) -> Dict[str, object]:
    if policy == "fixed_steps":
        return {"dimension": "steps", "value": limits.steps}
    if policy == "fixed_budget":
        return {"dimension": "cost_micros", "value": limits.cost_micros}
    return {"dimension": "adaptive_budget", "value": _safety_ceilings(limits)}


def _runtime_limits(policy: str, limits: BudgetLimits) -> BudgetLimits:
    if policy == "fixed_steps":
        return BudgetLimits(
            steps=limits.steps,
            tokens=max(limits.tokens, 10**12),
            time_ms=max(limits.time_ms, 10**12),
            cost_micros=max(limits.cost_micros, 10**12),
        )
    return limits


def _budget_use(card, active_limit: Mapping[str, object]) -> float:
    dimension = str(active_limit["dimension"])
    value = active_limit["value"]
    if not isinstance(value, int) or value <= 0:
        return card.budget.steps_used / card.budget.limits.steps
    used_by_dimension = {
        "steps": card.budget.steps_used,
        "tokens": card.budget.tokens_used,
        "time_ms": card.budget.time_ms_used,
        "cost_micros": card.budget.cost_micros_used,
    }
    return used_by_dimension[dimension] / value


def _run_task(
    task: Mapping[str, object], policy: str, seed: int, model_route: str = "scripted-v1", trial: int = 1
):
    task_id = str(task["task_id"])
    budgets = BudgetLimits.model_validate(
        task.get(
            "budgets",
            {"steps": 16, "tokens": 4000, "time_ms": 30_000, "cost_micros": 100_000},
        )
    )
    allowed_tools = list(task.get("allowed_tools", ["read_file"]))
    runtime_budgets = _runtime_limits(policy, budgets)
    success_contract = task.get("success_contract")
    if isinstance(success_contract, Mapping):
        scorer = BusinessContractScorer(
            required_tools=success_contract.get("required_tools", []),
            required_feedback_ids=success_contract.get("required_feedback_ids", []),
            min_score=float(success_contract.get("min_score", 1.0)),
        )
    else:
        scorer = FixtureScorer()
    contract = TaskContract(
        task_id=task_id,
        workspace_root=".",
        allowed_tools=allowed_tools,
        tool_levels={name: "L0" for name in allowed_tools},
        budgets=runtime_budgets,
        success_condition="score_at_least_one",
        scorer_version=str(getattr(scorer, "scorer_version", "fixture-v1")),
        metadata={
            "late_breakthrough_guard": bool(task.get("late_breakthrough", False)),
            "success_contract": dict(success_contract) if isinstance(success_contract, Mapping) else {},
        },
    )
    config = RunConfig(
        run_id=f"{task_id}-{policy}-{model_route}-{trial}",
        task_id=task_id,
        policy_version=policy,
        budget_window_steps=int(task.get("budget_window_steps", 2)),
        offline=True,
        seed=seed,
    )
    llm = ScriptedLLM({task_id: task.get("script", [])})
    tools = ToolRegistry()
    for name in allowed_tools:
        tools.register(name, "L0", lambda args: {"ok": True})
    return Runtime(_controller_for(policy)).run(contract, config, llm, tools, scorer)


def evaluate_suite(
    tasks: Iterable[Mapping[str, object]],
    policies: Optional[Iterable[str]] = None,
    seed: int = 0,
    manifest_revision: Optional[str] = None,
    *,
    trials: int = 1,
    model_routes: Optional[Iterable[str]] = None,
) -> List[Dict[str, object]]:
    if trials < 1:
        raise ValueError("trials must be positive")
    policy_names = list(policies or DEFAULT_POLICIES)
    route_names = list(model_routes or ["scripted-v1"])
    if not route_names or any(not str(route).strip() for route in route_names):
        raise ValueError("model_routes must contain at least one non-empty route")
    records: List[Dict[str, object]] = []
    for task in tasks:
        for policy in policy_names:
            for route in route_names:
                for trial in range(1, trials + 1):
                    trial_seed = seed + trial - 1
                    card = _run_task(task, policy, trial_seed, str(route), trial)
                    original_limits = BudgetLimits.model_validate(
                        task.get(
                            "budgets",
                            {
                                "steps": 16,
                                "tokens": 4000,
                                "time_ms": 30_000,
                                "cost_micros": 100_000,
                            },
                        )
                    )
                    active_limit = _active_limit(policy, original_limits)
                    record = {
                        "task_id": str(task["task_id"]),
                        "run_id": card.run_id,
                        "task_count": 1,
                        "policy": policy,
                        "policy_version": policy,
                        "model_route": str(route),
                        "trial": trial,
                        "scorer_version": str(
                            "business-contract-v1"
                            if isinstance(task.get("success_contract"), Mapping)
                            else "fixture-v1"
                        ),
                        "final_score": card.final_score,
                        "success": card.success,
                        "cost": card.budget.cost_micros_used,
                        "budget_use": _budget_use(card, active_limit),
                        "active_limit": active_limit,
                        "safety_ceilings": _safety_ceilings(original_limits),
                        "invalid_loop_rate": 0.0,
                        "false_stop": int(bool(task.get("late_breakthrough", False)) and not card.success),
                        "terminal_reason": card.terminal_reason,
                        "event_count": card.event_count,
                        "provenance": "offline",
                    }
                    if manifest_revision:
                        record["manifest_revision"] = manifest_revision
                    records.append(record)
    return records


def summarize_records(records: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Aggregate per-task rows while preserving explicit denominators."""

    grouped = defaultdict(list)
    for index, record in enumerate(records):
        normalized = dict(record)
        normalized.setdefault("task_id", f"row-{index}")
        normalized.setdefault("success", False)
        normalized.setdefault("final_score", 0.0)
        normalized.setdefault("cost", 0)
        normalized.setdefault("budget_use", 0.0)
        normalized.setdefault("invalid_loop_rate", 0.0)
        normalized.setdefault("false_stop", 0)
        normalized.setdefault("terminal_reason", "unknown")
        normalized.setdefault("model_route", "scripted-v1")
        normalized.setdefault("trial", 1)
        normalized.setdefault("scorer_version", "fixture-v1")
        grouped[(str(normalized["policy"]), str(normalized["model_route"]))].append(normalized)
    summaries: List[Dict[str, object]] = []
    for (policy, model_route), rows in sorted(grouped.items()):
        task_count = len({str(row["task_id"]) for row in rows})
        trial_count = len({int(row["trial"]) for row in rows})
        success_count = sum(bool(row["success"]) for row in rows)
        false_stop_count = sum(int(row["false_stop"]) for row in rows)
        total_cost = sum(int(row["cost"]) for row in rows)
        mean_score = sum(float(row["final_score"]) for row in rows) / len(rows)
        mean_budget_use = sum(float(row["budget_use"]) for row in rows) / len(rows)
        mean_invalid_loop = sum(float(row["invalid_loop_rate"]) for row in rows) / len(rows)
        reasons = defaultdict(int)
        for row in rows:
            reasons[str(row["terminal_reason"])] += 1
        summaries.append(
            {
                "policy": policy,
                "model_route": model_route,
                "trial_count": trial_count,
                "run_count": len(rows),
                "scorer_versions": sorted({str(row["scorer_version"]) for row in rows}),
                "task_count": task_count,
                "success_count": success_count,
                "success_rate": success_count / len(rows) if rows else 0.0,
                "mean_final_score": mean_score,
                "total_cost_micros": total_cost,
                "mean_cost_micros": total_cost / len(rows),
                "unit_success_cost_micros": (
                    total_cost / success_count if success_count else None
                ),
                "mean_budget_use": mean_budget_use,
                "mean_invalid_loop_rate": mean_invalid_loop,
                "false_stop_count": false_stop_count,
                "false_stop_rate": false_stop_count / len(rows) if rows else 0.0,
                "terminal_reasons": dict(sorted(reasons.items())),
                "provenance": "offline",
            }
        )
    return summaries
