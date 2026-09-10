from budgettrace.contracts import BudgetDelta, BudgetLimits, Event, RunConfig, TaskContract
from budgettrace.events import EventJournal
from budgettrace.features import extract_features
from budgettrace.fixtures import FixtureScorer, ScriptedLLM
from budgettrace.runtime import Runtime
from budgettrace.security import ToolRegistry


def _event(event_id, sequence, event_type, payload):
    return Event(
        event_id=event_id,
        run_id="run-1",
        sequence=sequence,
        event_type=event_type,
        occurred_at="2026-09-07T00:00:00Z",
        payload=payload,
        budget_delta=BudgetDelta(steps=1, tokens=10, time_ms=5, cost_micros=2),
    )


def _runtime_contract(tmp_path):
    return TaskContract(
        task_id="tool-chain",
        workspace_root=str(tmp_path),
        allowed_tools=["read_file"],
        tool_levels={"read_file": "L0"},
        budgets=BudgetLimits(steps=8, tokens=4000, time_ms=30_000, cost_micros=100_000),
        success_condition="score_at_least_one",
        scorer_version="fixture-v1",
    )


def _runtime_config():
    return RunConfig(
        run_id="run-tool-chain-features",
        task_id="tool-chain",
        policy_version="rules-v1",
        budget_window_steps=4,
        offline=True,
    )


def test_features_detect_repeated_tools_and_new_feedback():
    events = [
        _event("e1", 1, "tool_call", {"tool": "search", "args_hash": "a"}),
        _event("e2", 2, "tool_call", {"tool": "search", "args_hash": "a"}),
        _event("e3", 3, "feedback", {"feedback_id": "test-failed-1"}),
    ]

    snapshot = extract_features(
        events,
        score_history=[0.0, 0.2],
        budget=BudgetLimits(steps=8, tokens=1000, time_ms=1000, cost_micros=1000),
    )

    assert snapshot.tool_repeat_rate == 0.5
    assert snapshot.feedback_novelty == 1.0
    assert snapshot.score_slope == 0.2


def test_runtime_tool_journal_yields_non_missing_repeat_rate(tmp_path):
    llm = ScriptedLLM(
        {
            "tool-chain": [
                {"kind": "tool_call", "tool": "read_file", "args": {"path": "input.py"}},
                {"kind": "tool_call", "tool": "read_file", "args": {"path": "input.py"}},
                {"kind": "finish", "score": 1.0},
            ]
        }
    )
    tools = ToolRegistry()
    tools.register("read_file", "L0", lambda args: {"ok": True})

    card = Runtime().run(_runtime_contract(tmp_path), _runtime_config(), llm, tools, FixtureScorer())
    snapshot = extract_features(
        EventJournal.from_jsonl(card.journal_jsonl).events(),
        score_history=[1.0],
        budget=BudgetLimits(steps=8, tokens=4000, time_ms=30_000, cost_micros=100_000),
    )

    assert snapshot.tool_repeat_rate == 0.5
    assert "tool_repeat_rate" not in snapshot.missing_reason


def test_features_keep_missing_inputs_explicit():
    snapshot = extract_features(
        [],
        score_history=[],
        budget=BudgetLimits(steps=8, tokens=1000, time_ms=1000, cost_micros=1000),
    )

    assert snapshot.score_slope is None
    assert snapshot.feedback_novelty is None
    assert "score_slope" in snapshot.missing_reason
    assert "feedback_novelty" in snapshot.missing_reason


def test_features_mark_late_breakthrough_and_no_novelty_repeat():
    events = [
        _event("e1", 1, "tool_call", {"tool": "search", "args_hash": "a"}),
        _event("e2", 2, "tool_call", {"tool": "search", "args_hash": "a"}),
    ]

    snapshot = extract_features(
        events,
        score_history=[0.1, 0.1],
        budget=BudgetLimits(steps=8, tokens=1000, time_ms=1000, cost_micros=1000),
        late_breakthrough_guard=True,
    )

    assert snapshot.late_breakthrough_guard is True
    assert snapshot.new_feedback is False
    assert snapshot.high_repeat_no_novelty is True


def test_budget_exhaustion_checks_tokens_time_and_cost_not_only_steps():
    event = Event(
        event_id="e1",
        run_id="run-1",
        sequence=1,
        event_type="model_call",
        occurred_at="2026-09-07T00:00:00Z",
        payload={},
        budget_delta=BudgetDelta(steps=1, tokens=1000, time_ms=10, cost_micros=10),
    )
    snapshot = extract_features(
        [event],
        score_history=[0.0],
        budget=BudgetLimits(steps=8, tokens=1000, time_ms=1000, cost_micros=1000),
    )

    assert snapshot.budget_exhausted is True


def test_malformed_context_ratio_is_missing_not_fatal():
    event = _event("e1", 1, "tool_call", {"tool": "search", "args_hash": "a", "context_ratio": "bad"})

    snapshot = extract_features(
        [event],
        score_history=[0.0],
        budget=BudgetLimits(steps=8, tokens=1000, time_ms=1000, cost_micros=1000),
    )

    assert snapshot.context_pressure is None
    assert "context_pressure" in snapshot.missing_reason


def test_malformed_tool_event_does_not_count_as_repeat():
    event = _event("e1", 1, "tool_call", {"tool": "search"})

    snapshot = extract_features(
        [event],
        score_history=[0.0],
        budget=BudgetLimits(steps=8, tokens=1000, time_ms=1000, cost_micros=1000),
    )

    assert snapshot.tool_repeat_rate is None
    assert "tool_repeat_rate" in snapshot.missing_reason


def test_repeated_feedback_id_is_not_new_feedback():
    events = [
        _event("e1", 1, "feedback", {"feedback_id": "same-error"}),
        _event("e2", 2, "feedback", {"feedback_id": "same-error"}),
    ]

    snapshot = extract_features(
        events,
        score_history=[0.0, 0.0],
        budget=BudgetLimits(steps=8, tokens=1000, time_ms=1000, cost_micros=1000),
    )

    assert snapshot.feedback_novelty == 0.0
    assert snapshot.new_feedback is False


def test_previous_feedback_ids_are_not_reported_as_new():
    event = _event("e1", 1, "feedback", {"feedback_id": "old-error"})

    snapshot = extract_features(
        [event],
        score_history=[0.0],
        budget=BudgetLimits(steps=8, tokens=1000, time_ms=1000, cost_micros=1000),
        previous_feedback_ids={"old-error"},
    )

    assert snapshot.feedback_novelty == 0.0
    assert snapshot.new_feedback is False


def test_non_finite_context_ratio_is_missing():
    event = _event("e1", 1, "tool_call", {"tool": "search", "args_hash": "a", "context_ratio": "nan"})

    snapshot = extract_features(
        [event],
        score_history=[0.0],
        budget=BudgetLimits(steps=8, tokens=1000, time_ms=1000, cost_micros=1000),
    )

    assert snapshot.context_pressure is None
    assert "context_pressure" in snapshot.missing_reason
