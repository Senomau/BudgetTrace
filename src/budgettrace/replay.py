"""Replay a stored JSONL trajectory without invoking model or tools."""

from .contracts import BudgetLimits, RunCard
from .events import BudgetLedger, EventJournal


def replay_run(jsonl_text: str) -> RunCard:
    journal = EventJournal.from_jsonl(jsonl_text)
    events = journal.events()
    if not events or events[0].event_type != "task_started":
        raise ValueError("replay requires a task_started event")
    started = events[0].payload
    try:
        limits = BudgetLimits.model_validate(started["budgets"])
    except (KeyError, ValueError) as exc:
        raise ValueError("task_started event must include budgets") from exc
    ledger = BudgetLedger(limits)
    for event in events:
        ledger.apply(event)
    terminal = next((event for event in reversed(events) if event.event_type == "terminal"), None)
    if terminal is None:
        raise ValueError("replay requires a terminal event")
    decision = next((event for event in events if event.event_type == "decision"), None)
    return RunCard(
        run_id=events[0].run_id,
        task_id=str(started["task_id"]),
        policy_version=str(
            started.get("policy_version")
            or (decision.payload.get("policy_version") if decision else "unknown")
        ),
        terminal_reason=str(terminal.payload["reason"]),
        success=bool(terminal.payload.get("success", False)),
        final_score=float(terminal.payload.get("score", 0.0)),
        event_count=len(events),
        budget=ledger.snapshot,
        provenance="offline",
        journal_jsonl=journal.to_jsonl(),
    )
