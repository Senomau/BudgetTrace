from budgettrace.contracts import BudgetDelta, BudgetLimits, Event
from budgettrace.replay import replay_run


def test_replay_reconstructs_budget_terminal_and_score_from_jsonl():
    events = [
        Event(
            event_id="run-1-e1",
            run_id="run-1",
            sequence=1,
            event_type="task_started",
            occurred_at="2026-09-07T00:00:00Z",
            payload={
                "task_id": "task-1",
                "policy_version": "rules-v1",
                "budgets": {"steps": 4, "tokens": 1000, "time_ms": 100, "cost_micros": 1000},
            },
        ),
        Event(
            event_id="run-1-e2",
            run_id="run-1",
            sequence=2,
            event_type="model_call",
            occurred_at="2026-09-07T00:00:00Z",
            payload={},
            budget_delta=BudgetDelta(steps=1, tokens=10, time_ms=1, cost_micros=2),
        ),
        Event(
            event_id="run-1-e3",
            run_id="run-1",
            sequence=3,
            event_type="terminal",
            occurred_at="2026-09-07T00:00:00Z",
            payload={"reason": "success", "success": True, "score": 1.0},
        ),
    ]
    text = "\n".join(event.model_dump_json() for event in events)

    card = replay_run(text)

    assert card.run_id == "run-1"
    assert card.task_id == "task-1"
    assert card.success is True
    assert card.final_score == 1.0
    assert card.budget.steps_used == 1
