"""Multi-route, multi-trial live episode matrix with an explicit scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .contracts import BudgetLimits
from .live_config import LiveConfig
from .live_episode import (
    DEFAULT_BUDGET_WINDOW_STEPS,
    DEFAULT_MAX_STEPS,
    run_live_episode,
)
from .live_routes import LiveRoute
from .live_smoke import DEFAULT_MAX_OUTPUT_TOKENS, Transport
from .public_replay import Tau3ReferenceTask
from .security import redact_secrets

POLICY_VERSION = "live-episode-matrix-v1"
PROVENANCE = "live-model-run"


def _limits(config: LiveConfig, max_steps: int, max_output_tokens: int) -> BudgetLimits:
    return BudgetLimits(
        steps=max_steps,
        tokens=max(max_output_tokens * max_steps, 100_000),
        time_ms=max(30_000, max_steps * 2 * config.timeout_seconds * 1000),
        cost_micros=max(1, int(max_output_tokens * config.price_output_per_mtok * max_steps)),
    )


def run_live_matrix(
    routes: Iterable[LiveRoute],
    tasks: List[Tau3ReferenceTask],
    *,
    trials: int,
    max_steps: int = DEFAULT_MAX_STEPS,
    budget_window_steps: int = DEFAULT_BUDGET_WINDOW_STEPS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    scorer_mode: str = "reference-actions",
    transport: Optional[Transport] = None,
    source_root: Optional[Path] = None,
) -> Dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be positive")
    if max_steps < 1 or budget_window_steps < 1 or budget_window_steps >= max_steps:
        raise ValueError("budget_window_steps must satisfy 1 <= value < max_steps")
    route_list = list(routes)
    if not route_list:
        raise ValueError("routes must not be empty")
    records: List[Dict[str, Any]] = []
    for route in route_list:
        for task in tasks:
            for trial in range(1, trials + 1):
                run_id = f"tau3-{task.domain}-{task.split}-{route.route_id}-episode-{task.task_id}-t{trial}"
                record = run_live_episode(
                    route.config,
                    task,
                    limits=_limits(route.config, max_steps, max_output_tokens),
                    max_output_tokens=max_output_tokens,
                    max_steps=max_steps,
                    budget_window_steps=budget_window_steps,
                    run_id=run_id,
                    scorer_mode=scorer_mode,
                    transport=transport,
                    source_root=source_root,
                )
                record["route_id"] = route.route_id
                record["trial"] = trial
                records.append(record)

    completed = [record for record in records if record.get("status") == "completed"]
    success_count = sum(bool(record.get("protocol_success")) for record in completed)
    false_stop_count = sum(bool(record.get("false_stop")) for record in completed)
    cost_micros_total = sum(int(record.get("cost_micros", 0)) for record in completed)
    reasons: Dict[str, int] = {}
    for record in records:
        reason = str(record.get("terminal_reason", record.get("status", "unknown")))
        reasons[reason] = reasons.get(reason, 0) + 1
    summary = {
        "policy_version": POLICY_VERSION,
        "provenance": PROVENANCE,
        "scored": scorer_mode == "reference-actions",
        "scorer_mode": scorer_mode,
        "scorer_version": (
            "reference-action-contract-v1" if scorer_mode == "reference-actions" else "unscored-v1"
        ),
        "route_count": len(route_list),
        "routes": [route.route_id for route in route_list],
        "task_count": len(tasks),
        "trials": trials,
        "episodes_planned": len(route_list) * len(tasks) * trials,
        "episodes_completed": len(completed),
        "success_count": success_count,
        "protocol_success_count": success_count,
        "success_rate": success_count / len(completed) if completed else 0.0,
        "false_stop_count": false_stop_count,
        "false_stop_rate": false_stop_count / len(completed) if completed else 0.0,
        "cost_micros_total": cost_micros_total,
        "cost_usd_estimate": round(cost_micros_total / 1_000_000, 6),
        "unit_success_cost_micros": cost_micros_total / success_count if success_count else None,
        "terminal_reasons": dict(sorted(reasons.items())),
        "disclaimer": "Scored against declared reference action contracts; this is a deterministic protocol proxy, not an upstream benchmark score or business database oracle. Cost is an estimate from environment-declared prices.",
    }
    return {"records": records, "summary": summary}


def write_live_matrix_reports(result: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    journals = output_dir / "journals"
    journals.mkdir(exist_ok=True)
    public_records = []
    for record in result["records"]:
        public = {key: value for key, value in record.items() if key != "journal_jsonl"}
        public_records.append(redact_secrets(public))
        journal = record.get("journal_jsonl")
        if journal:
            (journals / f"{record['route_id']}-task-{record['task_id']}-trial{record['trial']}.jsonl").write_text(
                redact_secrets(journal), encoding="utf-8"
            )
    (output_dir / "records.json").write_text(
        json.dumps(public_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(redact_secrets(result["summary"]), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = result["summary"]
    lines = [
        "# BudgetTrace live episode matrix",
        "",
        f"routes {summary['route_count']} · tasks {summary['task_count']} · trials {summary['trials']} · "
        f"completed {summary['episodes_completed']}/{summary['episodes_planned']}",
        "",
        f"success {summary['success_count']}/{summary['episodes_completed']} ({summary['success_rate']:.3f}) · "
        f"false stop {summary['false_stop_count']}/{summary['episodes_completed']} ({summary['false_stop_rate']:.3f}) · "
        f"cost ${summary['cost_usd_estimate']:.6f} · unit success cost {summary['unit_success_cost_micros']}",
        "",
        "| route | task | trial | success | false stop | steps | tokens | cost (USD est) | terminal |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in result["records"]:
        lines.append(
            f"| {record['route_id']} | {record['task_id']} | {record['trial']} | "
            f"{record.get('business_success')} | {record.get('false_stop')} | {record['steps_used']} | "
            f"{record['tokens_used']} | {record['cost_usd_estimate']:.6f} | {record['terminal_reason']} |"
        )
    lines.extend(["", f"> {summary['disclaimer']}", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
