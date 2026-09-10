import pytest


def _event(event_id, sequence, steps, tokens):
    from budgettrace.contracts import BudgetDelta, Event

    return Event(
        event_id=event_id,
        run_id="run-1",
        sequence=sequence,
        event_type="model_call",
        occurred_at="2026-09-07T00:00:00Z",
        payload={"model": "scripted-v1"},
        budget_delta=BudgetDelta(steps=steps, tokens=tokens, time_ms=10, cost_micros=20),
    )


def test_budget_ledger_is_idempotent_for_duplicate_event_id():
    from budgettrace.contracts import BudgetLimits
    from budgettrace.events import BudgetLedger

    ledger = BudgetLedger(BudgetLimits(steps=3, tokens=1000, time_ms=100, cost_micros=1000))
    first = _event("evt-1", 1, steps=1, tokens=100)
    second = _event("evt-2", 2, steps=1, tokens=200)

    ledger.apply(first)
    ledger.apply(second)
    snapshot = ledger.apply(first)

    assert snapshot.steps_used == 2
    assert snapshot.tokens_used == 300
    assert snapshot.remaining_steps == 1


def test_event_journal_round_trip_and_duplicate_rejection():
    from budgettrace.events import EventJournal

    journal = EventJournal()
    first = _event("evt-1", 1, steps=1, tokens=100)
    second = _event("evt-2", 2, steps=1, tokens=200)
    journal.append(first)
    journal.append(second)

    with pytest.raises(ValueError, match="duplicate event_id"):
        journal.append(first)

    restored = EventJournal.from_jsonl(journal.to_jsonl())
    assert restored.events() == [first, second]


def test_event_journal_rejects_sequence_gap():
    from budgettrace.events import EventJournal

    journal = EventJournal()
    journal.append(_event("evt-1", 1, steps=1, tokens=100))

    with pytest.raises(ValueError, match="sequence"):
        journal.append(_event("evt-3", 3, steps=1, tokens=100))


def test_budget_ledger_rejects_conflicting_duplicate_event_id():
    from budgettrace.contracts import BudgetLimits
    from budgettrace.events import BudgetLedger

    ledger = BudgetLedger(BudgetLimits(steps=3, tokens=1000, time_ms=100, cost_micros=1000))
    first = _event("evt-1", 1, steps=1, tokens=100)
    conflicting = _event("evt-1", 1, steps=2, tokens=200)
    ledger.apply(first)

    with pytest.raises(ValueError, match="conflicting duplicate"):
        ledger.apply(conflicting)


def test_budget_ledger_validates_run_id_before_duplicate_event_return():
    from budgettrace.contracts import BudgetLimits
    from budgettrace.events import BudgetLedger

    ledger = BudgetLedger(BudgetLimits(steps=3, tokens=1000, time_ms=100, cost_micros=1000))
    first = _event("evt-1", 1, steps=1, tokens=100)
    ledger.apply(first)

    other_run_reuse = first.model_copy(update={"run_id": "run-2"})

    with pytest.raises(ValueError, match="run_id"):
        ledger.apply(other_run_reuse)


def test_budget_ledger_validates_duplicate_event_sequence_against_original():
    from budgettrace.contracts import BudgetLimits
    from budgettrace.events import BudgetLedger

    ledger = BudgetLedger(BudgetLimits(steps=3, tokens=1000, time_ms=100, cost_micros=1000))
    first = _event("evt-1", 1, steps=1, tokens=100)
    ledger.apply(first)

    wrong_sequence_reuse = first.model_copy(update={"sequence": 2})

    with pytest.raises(ValueError, match="sequence"):
        ledger.apply(wrong_sequence_reuse)


def test_budget_ledger_rejects_events_from_another_run():
    from budgettrace.contracts import BudgetLimits
    from budgettrace.events import BudgetLedger

    ledger = BudgetLedger(BudgetLimits(steps=3, tokens=1000, time_ms=100, cost_micros=1000))
    ledger.apply(_event("evt-1", 1, steps=1, tokens=100))
    other = _event("evt-2", 2, steps=1, tokens=100).model_copy(update={"run_id": "run-2"})

    with pytest.raises(ValueError, match="run_id"):
        ledger.apply(other)


def test_budget_ledger_rejects_sequence_gap():
    from budgettrace.contracts import BudgetLimits
    from budgettrace.events import BudgetLedger

    ledger = BudgetLedger(BudgetLimits(steps=3, tokens=1000, time_ms=100, cost_micros=1000))

    with pytest.raises(ValueError, match="sequence"):
        ledger.apply(_event("evt-2", 2, steps=1, tokens=100))


def test_budget_ledger_admission_does_not_mutate_and_reports_dimensions():
    from budgettrace.contracts import BudgetDelta, BudgetLimits
    from budgettrace.events import BudgetExceededError, BudgetLedger

    ledger = BudgetLedger(BudgetLimits(steps=2, tokens=100, time_ms=100, cost_micros=1000))

    with pytest.raises(BudgetExceededError) as excinfo:
        ledger.admit(BudgetDelta(steps=1, tokens=101, time_ms=1, cost_micros=1))

    assert excinfo.value.dimensions == ["tokens"]
    assert ledger.exceeded_dimensions(BudgetDelta(steps=1, tokens=101, time_ms=1, cost_micros=1)) == [
        "tokens"
    ]
    assert ledger.snapshot.steps_used == 0
    assert ledger.snapshot.tokens_used == 0


def test_budget_ledger_reconcile_records_over_limit_observations():
    from budgettrace.contracts import BudgetLimits
    from budgettrace.events import BudgetLedger

    ledger = BudgetLedger(BudgetLimits(steps=2, tokens=100, time_ms=100, cost_micros=1000))
    snapshot = ledger.reconcile(_event("evt-1", 1, steps=1, tokens=101))

    assert snapshot.steps_used == 1
    assert snapshot.tokens_used == 101
    assert ledger.snapshot.remaining_tokens == 0
    assert ledger.exceeded_dimensions(_event("evt-2", 2, steps=1, tokens=0).budget_delta) == [
        "tokens"
    ]
