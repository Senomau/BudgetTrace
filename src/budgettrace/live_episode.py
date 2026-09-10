"""Multi-turn live episode: Runtime's agent loop driven by LiveLLM.

Upgrades the ×T per-episode cost projection into a measured run: a real model
executes the T-step agent loop, every step is billed from the provider's real
usage, and the standard BudgetLedger gates each step before transport while
recording observed usage after transport even when it overruns a budget.

The default episode is informational: no scoring protocol is applied,
``scored`` is False, and ``terminal_reason == "success"`` only means the
model emitted a finish action. An explicit reference-action mode scores the
declared action contract as a deterministic protocol proxy.

The API key only travels from the environment into the Authorization header
at call time; it is never written to journals, reports, or error payloads.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .contracts import BudgetLimits, RunConfig, TaskContract
from .live_agent import LiveLLM
from .live_config import LiveConfig
from .live_smoke import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    PROVENANCE,
    SmokeError,
    Transport,
    _http_transport,
    chat_endpoint,
    prompt_from_task,
)
from .public_replay import Tau3ReferenceTask
from .runtime import Runtime
from .security import ToolRegistry, redact_secrets
from .tau2_retail import (
    RETAIL_TOOL_ENVIRONMENT,
    RetailToolEnvironment,
    has_retail_data,
    register_retail_tools,
    retail_tool_schemas,
)

POLICY_VERSION = "live-episode-v1"
DEFAULT_MAX_STEPS = 8
DEFAULT_BUDGET_WINDOW_STEPS = 2
NOOP_TOOL = "noop"

DISCLAIMER = (
    "Live multi-turn episode: measures a real agent loop (steps, tokens, "
    "latency, cost estimate) under the budget ledger. It is not a benchmark "
    "score, not a success-rate measurement, and not comparable to any "
    "official leaderboard. 'success' means the model emitted a finish action; "
    "the episode is unscored. Cost is an estimate computed from "
    "environment-declared prices; provider peak/off-peak and cache discounts "
    "are not distinguished."
)


class FinishScorer:
    """Ends the episode when the model emits a finish action; never scores."""

    scored = False

    def evaluate(self, events: List[Any]) -> Tuple[float, bool]:
        for event in events:
            if event.event_type == "feedback" and event.payload.get("feedback_id") == "finish":
                return 0.0, True
        return 0.0, False


class ReferenceActionScorer:
    """Score declared business actions plus an explicit finish signal."""

    scored = True
    scorer_version = "reference-action-contract-v1"

    def __init__(self, action_names: List[str]) -> None:
        self.action_names = tuple(name for name in action_names if name)

    def evaluate(self, events: List[Any]) -> Tuple[float, bool]:
        successful_tools = [
            str(event.payload.get("tool", ""))
            for event in events
            if event.event_type == "tool_result" and event.payload.get("ok") is True
        ]
        matched = 0
        cursor = 0
        for expected in self.action_names:
            try:
                cursor = successful_tools.index(expected, cursor) + 1
            except ValueError:
                continue
            matched += 1
        finished = any(
            event.event_type == "feedback" and event.payload.get("feedback_id") == "finish"
            for event in events
        )
        denominator = len(self.action_names) + 1
        evidence = (matched + int(finished)) / denominator if denominator else 0.0
        return evidence, matched == len(self.action_names) and finished


def build_task_contract(
    task: Tau3ReferenceTask,
    limits: BudgetLimits,
    workspace_root: str,
    *,
    allowed_tools: Optional[List[str]] = None,
    tool_levels: Optional[Dict[str, str]] = None,
    tool_environment: str = "noop",
    success_condition: str = "model_emitted_finish",
    scorer_version: str = "unscored-v1",
) -> TaskContract:
    """Runtime contract for an unscored live episode."""
    tools = allowed_tools or [NOOP_TOOL]
    levels = tool_levels or {NOOP_TOOL: "L0"}
    return TaskContract(
        task_id=task.task_id,
        workspace_root=workspace_root,
        allowed_tools=tools,
        tool_levels=levels,
        budgets=limits,
        success_condition=success_condition,
        scorer_version=scorer_version,
        metadata={
            "provenance": PROVENANCE,
            "policy_version": POLICY_VERSION,
            "tool_environment": tool_environment,
        },
    )


def run_live_episode(
    config: LiveConfig,
    task: Tau3ReferenceTask,
    *,
    limits: BudgetLimits,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    cheap_config: Optional[LiveConfig] = None,
    transport: Optional[Transport] = None,
    run_id: Optional[str] = None,
    max_steps: Optional[int] = None,
    budget_window_steps: Optional[int] = None,
    workspace_root: str = ".",
    source_root: Optional[Path] = None,
    scorer_mode: str = "finish",
) -> Dict[str, Any]:
    send = transport if transport is not None else _http_transport
    run_id = run_id or f"tau3-{task.domain}-{task.split}-live-episode-{task.task_id}"
    steps = max_steps if max_steps is not None else DEFAULT_MAX_STEPS
    window_steps = (
        budget_window_steps if budget_window_steps is not None else DEFAULT_BUDGET_WINDOW_STEPS
    )
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tools = ToolRegistry()
    tool_environment = "noop"
    tool_instructions = None
    tool_schemas = None
    allowed_tools = [NOOP_TOOL]
    tool_levels = {NOOP_TOOL: "L0"}
    if source_root is not None and has_retail_data(source_root, task):
        retail = RetailToolEnvironment.from_source(source_root, task)
        registered = register_retail_tools(tools, retail)
        allowed_tools = registered.allowed_tools
        tool_levels = registered.tool_levels
        tool_environment = RETAIL_TOOL_ENVIRONMENT
        tool_instructions = retail.tool_prompt()
        tool_schemas = retail_tool_schemas()
    else:
        tools.register(NOOP_TOOL, "L0", lambda args: {"status": "accepted"})

    if scorer_mode == "finish":
        scorer: Any = FinishScorer()
        success_condition = "model_emitted_finish"
        scorer_version = "unscored-v1"
    elif scorer_mode == "reference-actions":
        scorer = ReferenceActionScorer([str(action.get("name", "")) for action in task.actions])
        success_condition = "reference_actions_and_finish"
        scorer_version = scorer.scorer_version
    else:
        raise ValueError(f"unknown scorer_mode: {scorer_mode}")

    contract = build_task_contract(
        task,
        limits,
        workspace_root,
        allowed_tools=allowed_tools,
        tool_levels=tool_levels,
        tool_environment=tool_environment,
        success_condition=success_condition,
        scorer_version=scorer_version,
    )
    run_config = RunConfig(
        run_id=run_id,
        task_id=task.task_id,
        policy_version=POLICY_VERSION,
        budget_window_steps=window_steps,
        offline=False,
    )
    llm = LiveLLM(
        config,
        task,
        limits=limits,
        transport=send,
        max_output_tokens=max_output_tokens,
        cheap_config=cheap_config,
        tool_instructions=tool_instructions,
        tool_schemas=tool_schemas,
    )

    card = Runtime().run(contract, run_config, llm, tools, scorer)
    prompt, prompt_source = prompt_from_task(task)
    model_finished = any(call["action_kind"] == "finish" for call in llm.calls)

    scored = bool(getattr(scorer, "scored", False))
    record = {
        "task_id": task.task_id,
        "run_id": run_id,
        "domain": task.domain,
        "split": task.split,
        "upstream_commit": task.upstream_commit,
        "model": config.model,
        "base_url": config.base_url,
        "endpoint": chat_endpoint(config.base_url),
        "price_input_per_mtok": config.price_input_per_mtok,
        "price_output_per_mtok": config.price_output_per_mtok,
        "prompt_source": prompt_source,
        "tool_environment": tool_environment,
        "tool_count": len(allowed_tools),
        "tool_protocol": "openai-compatible-native" if tool_schemas else "text-json",
        "policy_version": POLICY_VERSION,
        "budget_window_steps": window_steps,
        "started_at": started_at,
        "steps_used": card.budget.steps_used,
        "tokens_used": card.budget.tokens_used,
        "time_ms_used": card.budget.time_ms_used,
        "cost_micros": card.budget.cost_micros_used,
        "cost_usd_estimate": round(card.budget.cost_micros_used / 1_000_000, 6),
        "terminal_reason": card.terminal_reason,
        "over_budget": card.over_budget,
        "over_budget_dimensions": list(card.over_budget_dimensions),
        "billing_status": card.billing_status,
        "reserved_budget": card.reserved_budget.model_dump(mode="json"),
        "observed_budget": card.observed_budget.model_dump(mode="json"),
        "model_finished": model_finished,
        "scorer_mode": scorer_mode,
        "scorer_version": scorer_version,
        "score": card.final_score,
        "business_success": bool(card.success) if scored else None,
        "protocol_success": bool(card.success) if scored else None,
        "false_stop": bool(scored and model_finished and not card.success),
        "event_count": card.event_count,
        "journal_jsonl": card.journal_jsonl,
        "scored": scored,
        "status": "completed",
        "provenance": PROVENANCE,
        "disclaimer": (
            DISCLAIMER
            if not scored
            else "Live episode scored against the declared reference-action contract. This is a deterministic protocol proxy, not an upstream benchmark score or a business database oracle. Cost is an estimate from environment-declared prices."
        ),
    }
    return record


def write_episode_reports(record: Dict[str, Any], output_dir: Path) -> None:
    record = redact_secrets(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    journals_dir = output_dir / "journals"
    journals_dir.mkdir(exist_ok=True)

    summary = {key: value for key, value in record.items() if key != "journal_jsonl"}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (journals_dir / f"task-{record['task_id']}.jsonl").write_text(
        record["journal_jsonl"], encoding="utf-8"
    )
    (output_dir / "report.md").write_text(episode_report_markdown(record), encoding="utf-8")


def episode_report_markdown(record: Dict[str, Any]) -> str:
    finished = "yes" if record["model_finished"] else "no"
    lines = [
        "# BudgetTrace live multi-turn episode",
        "",
        f"task `{record['task_id']}` · model `{record['model']}` · endpoint `{record['endpoint']}` "
        f"· upstream commit `{record['upstream_commit']}`",
        "",
        f"steps: {record['steps_used']} · tokens: {record['tokens_used']} · "
        f"wall time: {record['time_ms_used']} ms · cost estimate ${record['cost_usd_estimate']:.6f} "
        f"(at ${record['price_input_per_mtok']}/1M in, ${record['price_output_per_mtok']}/1M out)",
        "",
        f"terminal reason: `{record['terminal_reason']}` · model finished: {finished} · "
        f"over budget: {record['over_budget']} · "
        f"provenance: `{record['provenance']}` · scored: {record['scored']} · "
        f"events: {record['event_count']}",
        "",
        f"> {DISCLAIMER}",
        "",
        "The API key never appears in this report: it only exists in the local process "
        "environment and is sent in the Authorization header at call time.",
    ]
    return "\n".join(lines) + "\n"
