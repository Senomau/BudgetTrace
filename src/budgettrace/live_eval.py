"""Live eval matrix: N audited tasks x K trials of real model calls.

Orchestrates repeated live calls (one per task-trial pair) on top of the
single-call primitives in ``live_smoke`` and adds matrix-level accounting:

- budget cap: once the cumulative estimated spend (USD-micros over completed
  calls) reaches ``budget_usd``, the remaining planned calls are NOT issued
  and are recorded as ``skipped_budget_cap``;
- fault tolerance: a failed call is recorded with ``status=error`` and the
  matrix continues with the next planned call;
- aggregate summary: planned/completed/failed/skipped call counts, token and
  cost totals over completed calls, and the truncation rate.

Like ``live_smoke`` this module is informational: no scoring protocol runs,
so output is NOT a benchmark score, not a success-rate measurement, and not
comparable to any leaderboard. Provenance is fixed to ``live-model-run``.
The API key only travels from the environment into the Authorization header
at call time; it never appears in records, journals, or reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import BudgetLimits
from .live_config import LiveConfig
from .live_smoke import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    PROVENANCE,
    SmokeError,
    Transport,
    chat_endpoint,
    run_live_smoke,
)
from .public_replay import Tau3ReferenceTask
from .security import redact_secrets

POLICY_VERSION = "live-eval-v1"

EVAL_DISCLAIMER = (
    "Live eval matrix (N tasks x K trials) over audited tau3-bench tasks: "
    "measures raw live-model behavior (tokens, latency, truncation, cost "
    "estimate) under the budget ledger, without a scoring protocol. It is not "
    "a benchmark score, not a success-rate measurement, and not comparable to "
    "any official leaderboard. Cost is an estimate computed from "
    "environment-declared prices; provider peak/off-peak and cache discounts "
    "are not distinguished."
)


def _matrix_run_id(task: Tau3ReferenceTask, trial: int) -> str:
    return f"tau3-{task.domain}-{task.split}-live-{task.task_id}-t{trial}"


def _skipped_record(config: LiveConfig, task: Tau3ReferenceTask, trial: int) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "run_id": _matrix_run_id(task, trial),
        "domain": task.domain,
        "split": task.split,
        "model": config.model,
        "trial": trial,
        "status": "skipped_budget_cap",
        "reason": "budget cap reached before this call was issued",
        "scored": False,
        "provenance": PROVENANCE,
    }


def _error_record(
    config: LiveConfig, task: Tau3ReferenceTask, trial: int, exc: SmokeError
) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "run_id": _matrix_run_id(task, trial),
        "domain": task.domain,
        "split": task.split,
        "model": config.model,
        "trial": trial,
        "status": "error",
        "error": exc.to_dict(),
        "scored": False,
        "provenance": PROVENANCE,
    }


def run_live_eval(
    config: LiveConfig,
    tasks: List[Tau3ReferenceTask],
    *,
    trials: int,
    limits: BudgetLimits,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    transport: Optional[Transport] = None,
    budget_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the task-trial matrix; only completed calls bill against the cap."""
    budget_micros = None if budget_usd is None else max(0, int(round(budget_usd * 1_000_000)))
    spent_micros = 0
    records: List[Dict[str, Any]] = []
    for task in tasks:
        for trial in range(1, trials + 1):
            if budget_micros is not None and spent_micros >= budget_micros:
                records.append(_skipped_record(config, task, trial))
                continue
            try:
                record = run_live_smoke(
                    config,
                    task,
                    limits=limits,
                    max_output_tokens=max_output_tokens,
                    transport=transport,
                    run_id=_matrix_run_id(task, trial),
                )
            except SmokeError as exc:
                records.append(_error_record(config, task, trial, exc))
                continue
            record["trial"] = trial
            records.append(record)
            spent_micros += record["cost_micros"]

    completed = [r for r in records if r["status"] == "completed"]
    failed = [r for r in records if r["status"] == "error"]
    skipped = [r for r in records if r["status"] == "skipped_budget_cap"]
    prompt_tokens = sum(r["tokens"]["prompt"] for r in completed)
    completion_tokens = sum(r["tokens"]["completion"] for r in completed)
    cost_micros_total = sum(r["cost_micros"] for r in completed)
    truncated_count = sum(1 for r in completed if r["truncated"])
    summary = {
        "policy_version": POLICY_VERSION,
        "provenance": PROVENANCE,
        "scored": False,
        "model": config.model,
        "endpoint": chat_endpoint(config.base_url),
        "task_count": len(tasks),
        "trials": trials,
        "calls_planned": len(tasks) * trials,
        "calls_completed": len(completed),
        "calls_failed": len(failed),
        "calls_skipped": len(skipped),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "cost_micros_total": cost_micros_total,
        "cost_usd_estimate": round(cost_micros_total / 1_000_000, 6),
        "truncated_count": truncated_count,
        "truncation_rate": (truncated_count / len(completed)) if completed else 0.0,
        "budget_usd_cap": budget_usd,
        "disclaimer": EVAL_DISCLAIMER,
    }
    return {"records": records, "summary": summary}


def eval_report_markdown(result: Dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# BudgetTrace live eval matrix",
        "",
        f"policy `{summary['policy_version']}` · model `{summary['model']}` · endpoint `{summary['endpoint']}`",
        "",
        f"{summary['task_count']} task(s) × {summary['trials']} trial(s) · planned {summary['calls_planned']} · "
        f"completed {summary['calls_completed']} · failed {summary['calls_failed']} · skipped {summary['calls_skipped']}",
        "",
        f"tokens {summary['tokens']['prompt']} + {summary['tokens']['completion']} = {summary['tokens']['total']} · "
        f"cost estimate ${summary['cost_usd_estimate']:.6f} · "
        f"truncated {summary['truncated_count']}/{summary['calls_completed']} (rate {summary['truncation_rate']:.0%})",
        "",
        "| task | trial | status | tokens | cost (USD est) | finish | truncated |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in result["records"]:
        if record["status"] == "completed":
            lines.append(
                f"| {record['task_id']} | {record['trial']} | {record['status']} | "
                f"{record['tokens']['total']} | {record['cost_usd_estimate']:.6f} | "
                f"{record['finish_reason']} | {record['truncated']} |"
            )
        elif record["status"] == "error":
            lines.append(
                f"| {record['task_id']} | {record['trial']} | error ({record['error']['code']}) | - | - | - | - |"
            )
        else:
            lines.append(
                f"| {record['task_id']} | {record['trial']} | {record['status']} | - | - | - | - |"
            )
    lines.extend([
        "",
        f"> {EVAL_DISCLAIMER}",
        "",
        "The API key never appears in this report: it only exists in the local process "
        "environment and is sent in the Authorization header at call time.",
    ])
    return "\n".join(lines) + "\n"


def write_eval_reports(result: Dict[str, Any], output_dir: Path) -> None:
    result = redact_secrets(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    journals_dir = output_dir / "journals"
    journals_dir.mkdir(exist_ok=True)

    public_records = [
        {key: value for key, value in record.items() if key != "journal_jsonl"}
        for record in result["records"]
    ]
    (output_dir / "records.json").write_text(
        json.dumps(public_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for record in result["records"]:
        if record.get("journal_jsonl"):
            journal_path = journals_dir / f"task-{record['task_id']}-trial{record['trial']}.jsonl"
            journal_path.write_text(record["journal_jsonl"], encoding="utf-8")
    (output_dir / "report.md").write_text(eval_report_markdown(result), encoding="utf-8")
