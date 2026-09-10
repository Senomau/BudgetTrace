from budgettrace.evaluation import evaluate_suite, summarize_records


def test_evaluation_returns_four_policies_and_marks_naive_false_stop():
    tasks = [
        {
            "task_id": "late-001",
            "late_breakthrough": True,
            "budget_window_steps": 2,
            "budgets": {"steps": 6, "tokens": 1000, "time_ms": 1000, "cost_micros": 1000},
            "script": [
                {"kind": "feedback", "feedback_id": "plateau-1", "score": 0.0},
                {"kind": "feedback", "feedback_id": "plateau-2", "score": 0.0},
                {"kind": "feedback", "feedback_id": "breakthrough", "score": 1.0},
                {"kind": "feedback", "feedback_id": "verified", "score": 1.0},
            ],
        }
    ]

    records = evaluate_suite(tasks)

    assert {record["policy"] for record in records} == {
        "fixed_steps",
        "fixed_budget",
        "naive_stop",
        "budgettrace_rules_v2",
    }
    assert all("task_count" in record and record["task_count"] == 1 for record in records)
    naive = next(record for record in records if record["policy"] == "naive_stop")
    assert naive["false_stop"] == 1


def test_summary_aggregates_denominators_and_false_stop_rate():
    tasks = [
        {
            "task_id": "late-001",
            "late_breakthrough": True,
            "budget_window_steps": 2,
            "budgets": {"steps": 6, "tokens": 1000, "time_ms": 1000, "cost_micros": 1000},
            "script": [
                {"kind": "feedback", "feedback_id": "plateau-1", "score": 0.0},
                {"kind": "feedback", "feedback_id": "plateau-2", "score": 0.0},
                {"kind": "feedback", "feedback_id": "breakthrough", "score": 1.0},
                {"kind": "feedback", "feedback_id": "verified", "score": 1.0},
            ],
        }
    ]

    summary = summarize_records(evaluate_suite(tasks))

    naive = next(record for record in summary if record["policy"] == "naive_stop")
    assert naive["task_count"] == 1
    assert naive["success_count"] == 0
    assert naive["success_rate"] == 0.0
    assert naive["false_stop_count"] == 1
    assert naive["false_stop_rate"] == 1.0
    assert naive["provenance"] == "offline"


def test_budgettrace_stops_success_before_fixed_policy_redundant_steps():
    tasks = [
        {
            "task_id": "redundant-001",
            "budget_window_steps": 2,
            "budgets": {"steps": 8, "tokens": 1000, "time_ms": 1000, "cost_micros": 1000},
            "script": [
                {"kind": "feedback", "feedback_id": "done", "score": 1.0},
                {"kind": "feedback", "feedback_id": "post-check-1", "score": 1.0},
                {"kind": "feedback", "feedback_id": "post-check-2", "score": 1.0},
            ],
        }
    ]

    summary = summarize_records(evaluate_suite(tasks))
    fixed = next(row for row in summary if row["policy"] == "fixed_steps")
    budgettrace = next(row for row in summary if row["policy"] == "budgettrace_rules_v2")

    assert budgettrace["total_cost_micros"] < fixed["total_cost_micros"]


def test_evaluation_records_carry_manifest_revision():
    tasks = [
        {
            "task_id": "manifest-001",
            "budget_window_steps": 1,
            "budgets": {"steps": 2, "tokens": 100, "time_ms": 100, "cost_micros": 100},
            "script": [{"kind": "feedback", "feedback_id": "ok", "score": 1.0}],
        }
    ]

    records = evaluate_suite(tasks, policies=["budgettrace_rules_v2"], manifest_revision="sha256:test")

    assert records[0]["manifest_revision"] == "sha256:test"


def test_fixed_steps_and_fixed_budget_diverge_on_heterogeneous_costs():
    tasks = [
        {
            "task_id": "hetero-cost-001",
            "budget_window_steps": 1,
            "budgets": {"steps": 3, "tokens": 1000, "time_ms": 1000, "cost_micros": 2},
            "script": [
                {"kind": "feedback", "feedback_id": "expensive-start", "score": 0.0, "cost_micros": 3},
                {"kind": "feedback", "feedback_id": "done", "score": 1.0, "cost_micros": 3},
            ],
        }
    ]

    records = evaluate_suite(tasks, policies=["fixed_steps", "fixed_budget"])
    fixed_steps = next(record for record in records if record["policy"] == "fixed_steps")
    fixed_budget = next(record for record in records if record["policy"] == "fixed_budget")

    assert fixed_steps["success"] is True
    assert fixed_budget["success"] is False
    assert fixed_steps["terminal_reason"] != fixed_budget["terminal_reason"]
    assert fixed_steps["active_limit"] == {"dimension": "steps", "value": 3}
    assert fixed_budget["active_limit"] == {"dimension": "cost_micros", "value": 2}
    assert fixed_steps["safety_ceilings"]["cost_micros"] == 2


def test_evaluation_records_include_active_limit_and_safety_ceilings():
    tasks = [
        {
            "task_id": "limits-001",
            "budget_window_steps": 1,
            "budgets": {"steps": 2, "tokens": 100, "time_ms": 100, "cost_micros": 100},
            "script": [{"kind": "feedback", "feedback_id": "ok", "score": 1.0}],
        }
    ]

    record = evaluate_suite(tasks, policies=["fixed_steps"])[0]

    assert record["active_limit"] == {"dimension": "steps", "value": 2}
    assert record["safety_ceilings"] == {
        "steps": 2,
        "tokens": 100,
        "time_ms": 100,
        "cost_micros": 100,
    }


def test_business_contract_scorer_requires_tool_and_feedback_evidence():
    task = {
        "task_id": "business-001",
        "allowed_tools": ["validate"],
        "budget_window_steps": 2,
        "budgets": {"steps": 4, "tokens": 1000, "time_ms": 1000, "cost_micros": 1000},
        "success_contract": {
            "required_tools": ["validate"],
            "required_feedback_ids": ["validated"],
        },
        "script": [
            {"kind": "tool_call", "tool": "validate", "args": {}},
            {"kind": "feedback", "feedback_id": "validated", "score": 1.0},
        ],
    }

    record = evaluate_suite([task], policies=["budgettrace_rules_v2"])[0]

    assert record["success"] is True
    assert record["final_score"] == 1.0
    assert record["scorer_version"] == "business-contract-v1"


def test_business_contract_scorer_rejects_score_without_required_tool():
    task = {
        "task_id": "business-002",
        "budget_window_steps": 1,
        "budgets": {"steps": 2, "tokens": 1000, "time_ms": 1000, "cost_micros": 1000},
        "success_contract": {
            "required_tools": ["validate"],
            "required_feedback_ids": ["validated"],
        },
        "script": [{"kind": "feedback", "feedback_id": "validated", "score": 1.0}],
    }

    record = evaluate_suite([task], policies=["budgettrace_rules_v2"])[0]

    assert record["success"] is False
    assert record["final_score"] == 0.5


def test_evaluation_matrix_tracks_trials_and_model_routes():
    task = {
        "task_id": "matrix-001",
        "budget_window_steps": 1,
        "budgets": {"steps": 2, "tokens": 100, "time_ms": 100, "cost_micros": 100},
        "script": [{"kind": "feedback", "feedback_id": "ok", "score": 1.0}],
    }

    records = evaluate_suite(
        [task],
        policies=["budgettrace_rules_v2"],
        seed=10,
        trials=2,
        model_routes=["route-a", "route-b"],
    )

    assert len(records) == 4
    assert {(record["trial"], record["model_route"]) for record in records} == {
        (1, "route-a"),
        (2, "route-a"),
        (1, "route-b"),
        (2, "route-b"),
    }
    assert len({record["run_id"] for record in records}) == 4

    from budgettrace.evaluation import summarize_records

    summary = summarize_records(records)[0]
    assert summary["unit_success_cost_micros"] == (
        summary["total_cost_micros"] / summary["success_count"]
    )
