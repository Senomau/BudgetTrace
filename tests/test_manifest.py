import json


def test_manifest_revision_is_deterministic_and_lists_tasks(tmp_path):
    from budgettrace.manifest import build_manifest

    tasks = [
        {"task_id": "task-a", "script": [{"kind": "feedback", "score": 1.0}]},
        {"task_id": "task-b", "script": [{"kind": "feedback", "score": 0.0}]},
    ]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")

    manifest = build_manifest(tasks, dataset_id="mini-test-v1", seed=7)

    assert manifest["dataset_id"] == "mini-test-v1"
    assert manifest["dataset_revision"].startswith("sha256:")
    assert manifest["task_ids"] == ["task-a", "task-b"]
    assert manifest["seed"] == 7
    assert "budgettrace_rules_v2" in manifest["policy_versions"]
    assert manifest["policy_active_limits"]["fixed_steps"]["dimension"] == "steps"
    assert manifest["policy_active_limits"]["fixed_budget"]["dimension"] == "cost_micros"


def test_manifest_records_matrix_dimensions_and_scorer_versions():
    from budgettrace.manifest import build_manifest

    tasks = [
        {
            "task_id": "task-a",
            "success_contract": {"required_feedback_ids": ["done"]},
        }
    ]

    manifest = build_manifest(tasks, trials=3, model_routes=["route-a", "route-b"])

    assert manifest["trials"] == 3
    assert manifest["model_routes"] == ["route-a", "route-b"]
    assert manifest["scorer_versions"] == {"task-a": "business-contract-v1"}
