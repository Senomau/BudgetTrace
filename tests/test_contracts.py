import pytest


def _valid_event_payload():
    return {
        "event_id": "evt-1",
        "run_id": "run-1",
        "sequence": 1,
        "event_type": "task_started",
        "occurred_at": "2026-09-07T00:00:00Z",
        "payload": {},
    }


def test_valid_task_contract_keeps_budget_and_tool_boundary():
    from budgettrace.contracts import BudgetLimits, TaskContract

    contract = TaskContract(
        task_id="code-001",
        workspace_root="workspace",
        allowed_tools=["read_file", "run_tests"],
        budgets=BudgetLimits(steps=8, tokens=4000, time_ms=30_000, cost_micros=100_000),
        success_condition="tests_pass",
        scorer_version="pytest-v1",
    )

    assert contract.budgets.steps == 8
    assert contract.allowed_tools == ["read_file", "run_tests"]


def test_task_contract_rejects_negative_budget_and_empty_allowlist():
    from budgettrace.contracts import BudgetLimits, TaskContract

    with pytest.raises(ValueError):
        BudgetLimits(steps=-1, tokens=4000, time_ms=30_000, cost_micros=100_000)

    with pytest.raises(ValueError):
        TaskContract(
            task_id="code-001",
            workspace_root="workspace",
            allowed_tools=[],
            budgets=BudgetLimits(steps=8, tokens=4000, time_ms=30_000, cost_micros=100_000),
            success_condition="tests_pass",
            scorer_version="pytest-v1",
        )


def test_event_rejects_non_positive_sequence():
    from budgettrace.contracts import Event

    with pytest.raises(ValueError):
        Event(**(_valid_event_payload() | {"sequence": 0}))


def test_legacy_event_defaults_to_schema_version_one():
    from budgettrace.contracts import Event

    payload = _valid_event_payload()
    payload.pop("schema_version", None)
    assert Event.model_validate(payload).schema_version == 1


def test_new_event_can_declare_schema_version_two():
    from budgettrace.contracts import Event

    payload = _valid_event_payload() | {"schema_version": 2}
    assert Event.model_validate(payload).schema_version == 2


def test_billing_status_accepts_billed_for_future_account_reconciliation():
    from budgettrace.contracts import BudgetLimits, BudgetSnapshot, RunCard

    card = RunCard(
        run_id="run-1",
        task_id="task-1",
        policy_version="v1",
        terminal_reason="success",
        success=True,
        event_count=0,
        budget=BudgetSnapshot(limits=BudgetLimits(steps=1, tokens=1, time_ms=1, cost_micros=1)),
        billing_status="billed",
    )

    assert card.billing_status == "billed"


def test_budget_contract_rejects_string_and_boolean_numbers():
    from budgettrace.contracts import BudgetLimits

    with pytest.raises(ValueError):
        BudgetLimits(steps="8", tokens=4000, time_ms=30_000, cost_micros=100_000)

    with pytest.raises(ValueError):
        BudgetLimits(steps=True, tokens=4000, time_ms=30_000, cost_micros=100_000)


def test_runtime_and_event_numeric_fields_are_strict():
    from budgettrace.contracts import BudgetLimits, Event, RunConfig

    with pytest.raises(ValueError):
        RunConfig(run_id="run", task_id="task", policy_version="v1", budget_window_steps="2")

    with pytest.raises(ValueError):
        Event(
            event_id="evt-1",
            run_id="run-1",
            sequence="1",
            event_type="task_started",
            occurred_at="2026-09-07T00:00:00Z",
            payload={"budgets": BudgetLimits(steps=1, tokens=1, time_ms=1, cost_micros=1).model_dump()},
        )


def test_tool_levels_must_reference_allowlisted_tools():
    from budgettrace.contracts import BudgetLimits, TaskContract

    with pytest.raises(ValueError, match="tool_levels"):
        TaskContract(
            task_id="task-1",
            workspace_root="workspace",
            allowed_tools=["read_file"],
            tool_levels={"shell": "L3"},
            budgets=BudgetLimits(steps=1, tokens=1, time_ms=1, cost_micros=1),
            success_condition="done",
            scorer_version="v1",
        )
