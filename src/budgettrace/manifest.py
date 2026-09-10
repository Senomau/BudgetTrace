"""Dataset and experiment provenance helpers."""

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

DEFAULT_POLICIES = ["fixed_steps", "fixed_budget", "naive_stop", "budgettrace_rules_v2"]
DEFAULT_BUDGETS = {"steps": 16, "tokens": 4000, "time_ms": 30_000, "cost_micros": 100_000}


def build_manifest(
    tasks: Iterable[Mapping[str, object]],
    *,
    dataset_id: str = "budgettrace-mini-v1",
    policies: Optional[Iterable[str]] = None,
    seed: int = 0,
    trials: int = 1,
    model_routes: Optional[Iterable[str]] = None,
) -> dict:
    if trials < 1:
        raise ValueError("trials must be positive")
    task_list = list(tasks)
    route_list = [str(route) for route in (model_routes or ["scripted-v1"])]
    if not route_list or any(not route.strip() for route in route_list):
        raise ValueError("model_routes must contain at least one non-empty route")
    canonical = json.dumps(task_list, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    revision = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "task_ids": [str(task["task_id"]) for task in task_list],
        "policy_versions": list(policies or DEFAULT_POLICIES),
        "policy_active_limits": {
            "fixed_steps": {"dimension": "steps"},
            "fixed_budget": {"dimension": "cost_micros"},
            "naive_stop": {"dimension": "adaptive_budget"},
            "budgettrace_rules_v2": {"dimension": "adaptive_budget"},
        },
        "safety_ceilings_by_task": {
            str(task["task_id"]): dict(task.get("budgets", DEFAULT_BUDGETS)) for task in task_list
        },
        "seed": seed,
        "trials": trials,
        "model_routes": route_list,
        "scorer_versions": {
            str(task["task_id"]): (
                "business-contract-v1" if isinstance(task.get("success_contract"), Mapping) else "fixture-v1"
            )
            for task in task_list
        },
        "provenance": "offline",
    }


def load_tasks(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))
