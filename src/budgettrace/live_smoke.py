"""Single-call live model smoke over an OpenAI-compatible endpoint.

This module makes ONE real chat-completion call for one audited τ³-bench task
to prove the live path end to end: real tokens, real latency, real cost
estimate from environment-declared prices, budget enforcement through the
standard ledger, and a full event journal.

It is explicitly informational: a single call with no scoring protocol is NOT
a benchmark result, not a success-rate measurement, and not comparable to any
leaderboard. Output provenance is fixed to ``live-model-run``.

The API key only travels from the environment into the Authorization header at
call time; it is never written to journals, reports, or error payloads.
"""

from __future__ import annotations

import json
import http.client
import socket
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .contracts import BudgetDelta, BudgetLimits, Event
from .events import BudgetExceededError, BudgetLedger, EventJournal
from .live_config import LiveConfig
from .public_replay import Tau3ReferenceTask
from .security import redact_secrets

PROVENANCE = "live-model-run"
POLICY_VERSION = "live-smoke-v1"
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_LIMITS = {"steps": 80, "tokens": 200000, "time_ms": 120000}

SYSTEM_PROMPT = (
    "You are a customer-service agent for a retail store. "
    "Answer the user's request briefly and factually."
)

DISCLAIMER = (
    "Single-call live smoke: proves the live pipeline (tokens, latency, cost "
    "estimate, budget ledger) works against a real endpoint. It is not a "
    "benchmark score, not a success-rate measurement, and not comparable to "
    "any official leaderboard. Cost is an estimate computed from "
    "environment-declared prices; provider peak/off-peak and cache discounts "
    "are not distinguished."
)


class SmokeError(Exception):
    """Raised when the live smoke cannot proceed or violates a budget."""

    def __init__(self, code: str, message: str, task_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.task_id = task_id

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "task_id": self.task_id, "message": str(self)}


Transport = Callable[[LiveConfig, Dict[str, Any]], Dict[str, Any]]


def chat_endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def estimate_cost_micros(prompt_tokens: int, completion_tokens: int, price_input: float, price_output: float) -> int:
    """USD-micros for the call: (pt * pin + ct * pout) where prices are USD per 1M tokens."""
    return int(round(prompt_tokens * price_input + completion_tokens * price_output))


def _nested_instruction_parts(raw: Dict[str, Any]) -> List[str]:
    """Extract instructions from the τ³-bench nested shape user_scenario.instructions."""
    scenario = raw.get("user_scenario")
    instructions = scenario.get("instructions") if isinstance(scenario, dict) else None
    if not isinstance(instructions, dict):
        return []
    parts: List[str] = []
    task_instructions = instructions.get("task_instructions")
    if isinstance(task_instructions, str) and task_instructions.strip():
        parts.append(task_instructions)
    elif isinstance(task_instructions, list) and task_instructions:
        text = "\n".join(str(item) for item in task_instructions if str(item).strip())
        if text.strip():
            parts.append(text)
    reason = instructions.get("reason_for_call")
    if isinstance(reason, str) and reason.strip():
        parts.append("Reason for call: " + reason)
    known_info = instructions.get("known_info")
    unknown_info = instructions.get("unknown_info")
    known_text = _instruction_value_text(known_info)
    unknown_text = _instruction_value_text(unknown_info)
    if known_text:
        parts.append("Known information: " + known_text)
    if unknown_text:
        parts.append("Unknown information: " + unknown_text)
    if known_text or unknown_text:
        parts.append(
            "Do not invent unknown identifiers or personal details; ask the customer "
            "or use an available lookup path when information is missing."
        )
    return parts


def _instruction_value_text(value: Any) -> str:
    """Render a nested instruction field deterministically without inventing content."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(items)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return ""


def prompt_from_task(task: Tau3ReferenceTask) -> Tuple[str, str]:
    """Build the smoke prompt from the upstream task, recording which field fed it."""
    raw = task.raw
    instructions = raw.get("instructions")
    if isinstance(instructions, list) and instructions:
        return "\n".join(str(item) for item in instructions), "instructions"
    if isinstance(instructions, str) and instructions.strip():
        return instructions, "instructions"
    nested = _nested_instruction_parts(raw)
    if nested:
        return "\n\n".join(nested), "instructions"
    description = raw.get("description")
    if isinstance(description, str) and description.strip():
        return description, "description"
    return json.dumps(raw, ensure_ascii=False), "raw-json"


def select_task(tasks: List[Tau3ReferenceTask], task_id: str) -> Tau3ReferenceTask:
    for task in tasks:
        if task.task_id == str(task_id):
            return task
    raise SmokeError(
        "task_not_selected",
        f"task {task_id} is not part of the audited subset",
        str(task_id),
    )


def _is_transient(exc: BaseException) -> bool:
    """True for transport failures worth one retry.

    A local HTTPS proxy can occasionally terminate a CONNECT tunnel with an
    ``SSLEOFError`` before the provider returns an HTTP response.  Treat that
    specific EOF like a timeout; authentication and HTTP 4xx/5xx responses are
    still fail-fast and are never retried here.
    """
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, (socket.timeout, TimeoutError, ssl.SSLEOFError, http.client.RemoteDisconnected)):
        return True
    return "EOF occurred in violation of protocol" in str(reason)


def _http_transport(config: LiveConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
    auth_scheme = "Bearer"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"{auth_scheme} {config.api_key}",
    }
    request = urllib.request.Request(
        chat_endpoint(config.base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    for attempt in range(2):  # one automatic retry for transient transport errors
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            raise SmokeError("http_error", f"HTTP {exc.code}: {body}")
        except json.JSONDecodeError as exc:
            raise SmokeError("http_error", f"non-JSON response body: {exc}")
        except urllib.error.URLError as exc:
            if _is_transient(exc) and attempt == 0:
                continue
            raise SmokeError("http_error", f"connection failed: {exc.reason}")
        except OSError as exc:  # read timeouts surface as TimeoutError here
            if _is_transient(exc) and attempt == 0:
                continue
            raise SmokeError("http_error", f"transport failed: {exc}")
    raise SmokeError("http_error", "transport failed after retry")


def run_live_smoke(
    config: LiveConfig,
    task: Tau3ReferenceTask,
    *,
    limits: BudgetLimits,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    transport: Optional[Transport] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    send = transport if transport is not None else _http_transport
    prompt, prompt_source = prompt_from_task(task)
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_output_tokens,
    }
    run_id = run_id or f"tau3-{task.domain}-{task.split}-live-{task.task_id}"
    journal = EventJournal()
    ledger = BudgetLedger(limits)
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def append(event_type: str, event_payload: Dict[str, Any], delta: Optional[BudgetDelta] = None) -> None:
        sequence = len(journal.events()) + 1
        event = Event(
            event_id=f"{run_id}-e{sequence}",
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            occurred_at=started_at,
            payload=event_payload,
            budget_delta=delta or BudgetDelta(),
        )
        try:
            ledger.apply(event)
        except BudgetExceededError as exc:
            raise SmokeError("budget_exceeded", str(exc), task.task_id) from exc
        journal.append(event)

    append(
        "task_started",
        {
            "task_id": task.task_id,
            "policy_version": POLICY_VERSION,
            "provenance": PROVENANCE,
            "upstream_commit": task.upstream_commit,
            "split": task.split,
            "prompt_source": prompt_source,
            "model": config.model,
            "base_url": config.base_url,
            "endpoint": chat_endpoint(config.base_url),
            "price_input_per_mtok": config.price_input_per_mtok,
            "price_output_per_mtok": config.price_output_per_mtok,
            "budgets": limits.model_dump(mode="json"),
        },
    )

    call_start = time.perf_counter()
    try:
        response = send(config, payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        raise SmokeError("http_error", f"HTTP {exc.code}: {body}", task.task_id) from exc
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        raise SmokeError("http_error", f"transport failed: {exc}", task.task_id) from exc
    latency_ms = int((time.perf_counter() - call_start) * 1000)

    usage = response.get("usage")
    if not isinstance(usage, dict) or "prompt_tokens" not in usage or "completion_tokens" not in usage:
        raise SmokeError("response_missing_usage", "provider response has no token usage", task.task_id)
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    total_tokens = prompt_tokens + completion_tokens
    cost_micros = estimate_cost_micros(
        prompt_tokens, completion_tokens, config.price_input_per_mtok, config.price_output_per_mtok
    )

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content", ""))
    finish_reason = choice.get("finish_reason")
    truncated = finish_reason == "length"
    completion_details = usage.get("completion_tokens_details")
    reasoning_tokens = None
    if isinstance(completion_details, dict) and isinstance(completion_details.get("reasoning_tokens"), int):
        reasoning_tokens = completion_details["reasoning_tokens"]
    append(
        "model_call",
        {
            "model": config.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_micros": cost_micros,
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "truncated": truncated,
            "reasoning_tokens": reasoning_tokens,
            "response_id": response.get("id"),
            "content_preview": content[:500],
        },
        BudgetDelta(steps=1, tokens=total_tokens, time_ms=latency_ms, cost_micros=cost_micros),
    )
    append(
        "terminal",
        {
            "reason": "live_smoke_complete",
            "scored": False,
            "disclaimer": DISCLAIMER,
        },
    )

    return {
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
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
        "cost_micros": cost_micros,
        "cost_usd_estimate": round(cost_micros / 1_000_000, 6),
        "latency_ms": latency_ms,
        "response_id": response.get("id"),
        "finish_reason": finish_reason,
        "truncated": truncated,
        "reasoning_tokens": reasoning_tokens,
        "content_preview": content[:500],
        "event_count": len(journal.events()),
        "journal_jsonl": journal.to_jsonl(),
        "scored": False,
        "status": "completed",
        "provenance": PROVENANCE,
        "disclaimer": DISCLAIMER,
    }


def write_smoke_reports(record: Dict[str, Any], output_dir: Path) -> None:
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
    (output_dir / "report.md").write_text(smoke_report_markdown(record), encoding="utf-8")


def smoke_report_markdown(record: Dict[str, Any]) -> str:
    lines = [
        "# BudgetTrace live model smoke",
        "",
        f"task `{record['task_id']}` · model `{record['model']}` · endpoint `{record['endpoint']}` · upstream commit `{record['upstream_commit']}`",
        "",
        f"tokens: prompt {record['tokens']['prompt']} + completion {record['tokens']['completion']} = {record['tokens']['total']} · "
        f"latency {record['latency_ms']} ms · cost estimate ${record['cost_usd_estimate']:.6f} "
        f"(at ${record['price_input_per_mtok']}/1M in, ${record['price_output_per_mtok']}/1M out)",
        "",
    ]
    if record.get("truncated"):
        lines.append(
            f"WARNING: response truncated (finish_reason={record.get('finish_reason')}); "
            "reasoning models may need a larger --max-output-tokens budget."
        )
        lines.append("")
    lines.extend([
        f"provenance: `{record['provenance']}` · scored: {record['scored']} · events: {record['event_count']}",
        "",
        f"> {DISCLAIMER}",
        "",
        "The API key never appears in this report: it only exists in the local process "
        "environment and is sent in the Authorization header at call time.",
    ])
    return "\n".join(lines) + "\n"
