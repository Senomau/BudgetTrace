"""Command-line entry points for offline replay and evaluation."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .evaluation import evaluate_suite, summarize_records
from .contracts import BudgetLimits
from .live_config import (
    LiveConfigError,
    config_check_payload,
    load_live_config,
    load_optional_cheap_config,
)
from .live_episode import DEFAULT_BUDGET_WINDOW_STEPS, DEFAULT_MAX_STEPS, run_live_episode, write_episode_reports
from .live_eval import run_live_eval, write_eval_reports
from .live_matrix import run_live_matrix, write_live_matrix_reports
from .live_routes import load_route_manifest
from .live_smoke import (
    DEFAULT_LIMITS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    SmokeError,
    run_live_smoke,
    select_task,
    write_smoke_reports,
)
from .manifest import build_manifest
from .public_audit import audit_tau3_retail
from .public_replay import (
    AdapterError,
    load_tau3_reference_tasks,
    replay_tau3_reference,
    write_replay_reports,
)
from .reporting import write_json, write_report, write_value
from .replay import replay_run


def _print_json(payload) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        sys.stdout.write(text + "\n")
    except UnicodeEncodeError:
        sys.stdout.write(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _model_routes(value: str) -> List[str]:
    routes = [item.strip() for item in value.split(",")]
    if not routes or any(not item for item in routes):
        raise argparse.ArgumentTypeError("must be a comma-separated list of non-empty route IDs")
    return routes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="budgettrace")
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("jsonl", type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("tasks", type=Path)
    evaluate.add_argument("--manifest", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--trials", type=_positive_int, default=1)
    evaluate.add_argument("--model-routes", type=_model_routes, default=["scripted-v1"])
    report = subparsers.add_parser("report")
    report.add_argument("run_dir", type=Path)
    public_audit = subparsers.add_parser("public-audit")
    public_audit.add_argument("--source", type=Path, required=True)
    public_audit.add_argument("--manifest", type=Path, required=True)
    public_audit.add_argument("--observed-commit")
    public_audit.add_argument("--output", type=Path)
    public_replay = subparsers.add_parser("public-replay")
    public_replay.add_argument("--source", type=Path, required=True)
    public_replay.add_argument("--manifest", type=Path, required=True)
    public_replay.add_argument("--observed-commit")
    public_replay.add_argument("--output", type=Path)
    subparsers.add_parser("live-config-check")
    live_smoke = subparsers.add_parser("live-smoke")
    live_smoke.add_argument("--source", type=Path, required=True)
    live_smoke.add_argument("--manifest", type=Path, required=True)
    live_smoke.add_argument("--task", required=True)
    live_smoke.add_argument("--output", type=Path)
    live_smoke.add_argument("--max-output-tokens", type=int, default=None)
    live_eval = subparsers.add_parser("live-eval")
    live_eval.add_argument("--source", type=Path, required=True)
    live_eval.add_argument("--manifest", type=Path, required=True)
    live_eval.add_argument(
        "--tasks", default=None, help="comma-separated task ids; default: all audited tasks"
    )
    live_eval.add_argument("--trials", type=int, default=3)
    live_eval.add_argument("--output", type=Path)
    live_eval.add_argument("--max-output-tokens", type=int, default=None)
    live_eval.add_argument("--budget-usd", type=float, default=None)
    live_episode = subparsers.add_parser("live-episode")
    live_episode.add_argument("--source", type=Path, required=True)
    live_episode.add_argument("--manifest", type=Path, required=True)
    live_episode.add_argument("--task", required=True)
    live_episode.add_argument("--output", type=Path)
    live_episode.add_argument("--max-output-tokens", type=int, default=None)
    live_episode.add_argument("--max-steps", type=int, default=None)
    live_episode.add_argument("--budget-window-steps", type=int, default=None)
    live_episode.add_argument(
        "--scorer-mode", choices=["finish", "reference-actions"], default="finish"
    )
    live_matrix = subparsers.add_parser("live-matrix")
    live_matrix.add_argument("--source", type=Path, required=True)
    live_matrix.add_argument("--manifest", type=Path, required=True)
    live_matrix.add_argument("--routes", type=Path, required=True)
    live_matrix.add_argument("--tasks", default=None)
    live_matrix.add_argument("--trials", type=_positive_int, default=1)
    live_matrix.add_argument("--output", type=Path, required=True)
    live_matrix.add_argument("--max-output-tokens", type=_positive_int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    live_matrix.add_argument("--max-steps", type=_positive_int, default=DEFAULT_MAX_STEPS)
    live_matrix.add_argument("--budget-window-steps", type=_positive_int, default=DEFAULT_BUDGET_WINDOW_STEPS)
    live_matrix.add_argument(
        "--scorer-mode", choices=["finish", "reference-actions"], default="reference-actions"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        card = replay_run(args.jsonl.read_text(encoding="utf-8"))
        _print_json(card.model_dump(mode="json"))
        return 0
    if args.command == "evaluate":
        tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
        manifest = None
        if args.manifest:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        elif args.output:
            manifest = build_manifest(
                tasks,
                trials=args.trials,
                model_routes=args.model_routes,
            )
        records = evaluate_suite(
            tasks,
            manifest_revision=manifest.get("dataset_revision") if manifest else None,
            trials=args.trials,
            model_routes=args.model_routes,
        )
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(records, args.output / "records.json")
            write_json(summarize_records(records), args.output / "summary.json")
            write_value(manifest, args.output / "manifest.json")
            write_report(records, args.output / "report.md")
        _print_json(records)
        return 0
    if args.command == "report":
        records_path = args.run_dir / "records.json"
        records = json.loads(records_path.read_text(encoding="utf-8"))
        write_report(records, args.run_dir / "report.md")
        return 0
    if args.command == "public-audit":
        result = audit_tau3_retail(
            args.source,
            args.manifest,
            observed_commit=args.observed_commit,
        )
        payload = result.to_dict()
        if args.output:
            write_value(payload, args.output)
        _print_json(payload)
        return 0 if result.passed else 2
    if args.command == "public-replay":
        try:
            tasks = load_tau3_reference_tasks(
                args.source,
                args.manifest,
                observed_commit=args.observed_commit,
            )
        except AdapterError as exc:
            _print_json({"error": exc.to_dict(), "passed": False})
            return 2
        result = replay_tau3_reference(tasks)
        if args.output:
            write_replay_reports(result, args.output)
        _print_json(result.to_dict())
        return 0 if result.failed_count == 0 and result.task_count > 0 else 2
    if args.command == "live-config-check":
        try:
            payload = config_check_payload()
        except LiveConfigError as exc:
            _print_json({"error": exc.to_dict(), "passed": False})
            return 2
        _print_json(payload)
        return 0
    if args.command == "live-smoke":
        try:
            config = load_live_config()
        except LiveConfigError as exc:
            _print_json({"error": exc.to_dict(), "passed": False})
            return 2
        try:
            tasks = load_tau3_reference_tasks(args.source, args.manifest)
            task = select_task(tasks, args.task)
            max_output_tokens = (
                args.max_output_tokens
                if args.max_output_tokens is not None
                else DEFAULT_MAX_OUTPUT_TOKENS
            )
            limits = BudgetLimits(
                steps=DEFAULT_LIMITS["steps"],
                tokens=DEFAULT_LIMITS["tokens"],
                time_ms=DEFAULT_LIMITS["time_ms"],
                cost_micros=int(DEFAULT_LIMITS["tokens"] * config.price_output_per_mtok) + 1,
            )
            record = run_live_smoke(
                config, task, limits=limits, max_output_tokens=max_output_tokens
            )
        except (AdapterError, SmokeError) as exc:
            _print_json({"error": exc.to_dict(), "passed": False})
            return 2
        if args.output:
            write_smoke_reports(record, args.output)
        _print_json({k: v for k, v in record.items() if k != "journal_jsonl"})
        return 0
    if args.command == "live-eval":
        try:
            config = load_live_config()
        except LiveConfigError as exc:
            _print_json({"error": exc.to_dict(), "passed": False})
            return 2
        try:
            tasks = load_tau3_reference_tasks(args.source, args.manifest)
            if args.tasks:
                wanted = [part.strip() for part in str(args.tasks).split(",") if part.strip()]
                tasks = [select_task(tasks, task_id) for task_id in wanted]
            max_output_tokens = (
                args.max_output_tokens
                if args.max_output_tokens is not None
                else DEFAULT_MAX_OUTPUT_TOKENS
            )
            limits = BudgetLimits(
                steps=DEFAULT_LIMITS["steps"],
                tokens=DEFAULT_LIMITS["tokens"],
                time_ms=DEFAULT_LIMITS["time_ms"],
                cost_micros=int(DEFAULT_LIMITS["tokens"] * config.price_output_per_mtok) + 1,
            )
            result = run_live_eval(
                config,
                tasks,
                trials=args.trials,
                limits=limits,
                max_output_tokens=max_output_tokens,
                budget_usd=args.budget_usd,
            )
        except (AdapterError, SmokeError) as exc:
            _print_json({"error": exc.to_dict(), "passed": False})
            return 2
        if args.output:
            write_eval_reports(result, args.output)
        _print_json(result["summary"])
        return 0
    if args.command == "live-episode":
        try:
            config = load_live_config()
            cheap_config = load_optional_cheap_config()
        except LiveConfigError as exc:
            _print_json({"error": exc.to_dict(), "passed": False})
            return 2
        max_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
        budget_window_steps = (
            args.budget_window_steps
            if args.budget_window_steps is not None
            else DEFAULT_BUDGET_WINDOW_STEPS
        )
        if budget_window_steps < 1 or budget_window_steps >= max_steps:
            _print_json(
                {
                    "error": {
                        "code": "invalid_budget_window_steps",
                        "message": "budget-window-steps must satisfy 1 <= value < max-steps",
                        "variables": ["--budget-window-steps", "--max-steps"],
                    },
                    "passed": False,
                }
            )
            return 2
        try:
            tasks = load_tau3_reference_tasks(args.source, args.manifest)
            task = select_task(tasks, args.task)
            max_output_tokens = (
                args.max_output_tokens
                if args.max_output_tokens is not None
                else DEFAULT_MAX_OUTPUT_TOKENS
            )
            episode_time_ms = max(
                DEFAULT_LIMITS["time_ms"], max_steps * 2 * config.timeout_seconds * 1000
            )
            limits = BudgetLimits(
                steps=max_steps,
                tokens=DEFAULT_LIMITS["tokens"],
                time_ms=episode_time_ms,
                cost_micros=int(DEFAULT_LIMITS["tokens"] * config.price_output_per_mtok) + 1,
            )
            record = run_live_episode(
                config,
                task,
                limits=limits,
                max_output_tokens=max_output_tokens,
                cheap_config=cheap_config,
                max_steps=max_steps,
                budget_window_steps=budget_window_steps,
                source_root=args.source,
                scorer_mode=args.scorer_mode,
            )
        except (AdapterError, SmokeError) as exc:
            _print_json({"error": exc.to_dict(), "passed": False})
            return 2
        if args.output:
            write_episode_reports(record, args.output)
        _print_json({k: v for k, v in record.items() if k != "journal_jsonl"})
        return 0
    if args.command == "live-matrix":
        try:
            routes = load_route_manifest(args.routes)
            tasks = load_tau3_reference_tasks(args.source, args.manifest)
            if args.tasks:
                wanted = [part.strip() for part in str(args.tasks).split(",") if part.strip()]
                tasks = [select_task(tasks, task_id) for task_id in wanted]
            result = run_live_matrix(
                routes,
                tasks,
                trials=args.trials,
                max_steps=args.max_steps,
                budget_window_steps=args.budget_window_steps,
                max_output_tokens=args.max_output_tokens,
                scorer_mode=args.scorer_mode,
                source_root=args.source,
            )
        except (AdapterError, LiveConfigError, SmokeError, ValueError) as exc:
            payload = exc.to_dict() if hasattr(exc, "to_dict") else {"code": "invalid_live_matrix", "message": str(exc)}
            _print_json({"error": payload, "passed": False})
            return 2
        write_live_matrix_reports(result, args.output)
        _print_json(result["summary"])
        return 0
    raise AssertionError("unreachable")
