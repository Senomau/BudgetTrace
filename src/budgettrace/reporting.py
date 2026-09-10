"""Human- and machine-readable evaluation reports."""

import json
from pathlib import Path
from typing import Iterable, Mapping

from .evaluation import summarize_records


def records_to_markdown(records: Iterable[Mapping[str, object]]) -> str:
    rows = list(records)
    lines = [
        "# BudgetTrace offline evaluation",
        "",
        "| task | policy | route | trial | scorer | active limit | score | success | cost (micros) | false stop | provenance |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for record in rows:
        active_limit = record.get("active_limit", {})
        if isinstance(active_limit, Mapping):
            active_label = str(active_limit.get("dimension", ""))
        else:
            active_label = ""
        lines.append(
            "| {task_id} | {policy} | {model_route} | {trial} | {scorer_version} | {active_limit} | {final_score} | {success} | {cost} | {false_stop} | {provenance} |".format(
                task_id=record.get("task_id", ""),
                policy=record.get("policy", ""),
                model_route=record.get("model_route", "scripted-v1"),
                trial=record.get("trial", 1),
                scorer_version=record.get("scorer_version", "fixture-v1"),
                active_limit=active_label,
                final_score=record.get("final_score", ""),
                success=record.get("success", ""),
                cost=record.get("cost", ""),
                false_stop=record.get("false_stop", ""),
                provenance=record.get("provenance", ""),
            )
        )
    return "\n".join(lines) + "\n"


def write_json(records: Iterable[Mapping[str, object]], path: Path) -> None:
    path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2), encoding="utf-8")


def write_value(value: object, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(records: Iterable[Mapping[str, object]], path: Path) -> None:
    path.write_text(records_to_markdown(records), encoding="utf-8")


def summary_to_markdown(summary: Iterable[Mapping[str, object]]) -> str:
    lines = [
        "# BudgetTrace offline evaluation summary",
        "",
        "| policy | route | tasks | runs | trials | success rate | mean score | total cost (micros) | unit success cost (micros) | false-stop rate | provenance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in summary:
        lines.append(
            "| {policy} | {model_route} | {task_count} | {run_count} | {trial_count} | {success_rate:.3f} | {mean_final_score:.3f} | {total_cost_micros} | {unit_success_cost} | {false_stop_rate:.3f} | {provenance} |".format(
                unit_success_cost=(
                    "n/a"
                    if record.get("unit_success_cost_micros") is None
                    else f"{float(record['unit_success_cost_micros']):.1f}"
                ),
                **record,
            )
        )
    return "\n".join(lines) + "\n"


def write_report(records: Iterable[Mapping[str, object]], path: Path) -> None:
    rows = list(records)
    path.write_text(summary_to_markdown(summarize_records(rows)), encoding="utf-8")
