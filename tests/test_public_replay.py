import json

import pytest

COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"

TASK_33 = {
    "id": "33",
    "evaluation_criteria": {
        "actions": [
            {"action_id": "33_0", "name": "find_user_id_by_name_zip", "arguments": {"zip": "10108"}},
            {"action_id": "33_1", "name": "get_user_details", "arguments": {"user_id": "noah_patel_6952"}},
        ],
        "reward_basis": ["DB"],
        "nl_assertions": [],
        "communicate_info": [],
    },
}

TASK_38 = {
    "id": "38",
    "evaluation_criteria": {
        "actions": [
            {"action_id": "38_0", "name": "find_user_id_by_email", "arguments": {"email": "a@b.c"}},
        ],
        "reward_basis": ["DB", "NL_ASSERTION", "COMMUNICATE"],
        "nl_assertions": ["agent confirms the exchange"],
        "communicate_info": ["order_id"],
    },
}


def _write_public_source(tmp_path, tasks, split_test_ids):
    source = tmp_path / "tau2-bench"
    data_dir = source / "data" / "tau2" / "domains" / "retail"
    data_dir.mkdir(parents=True)
    (data_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    (data_dir / "split_tasks.json").write_text(
        json.dumps({"test": list(split_test_ids)}),
        encoding="utf-8",
    )
    return source


def _write_manifest(tmp_path, task_ids):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "tau3-retail-audited-v1",
                "upstream": {
                    "name": "sierra-research/tau2-bench",
                    "tag": "v1.0.1",
                    "commit": COMMIT,
                    "data_path": "data/tau2/domains/retail",
                },
                "domain": "retail",
                "split": "test",
                "task_ids": task_ids,
                "excluded_known_issue_ids": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _happy_source_and_manifest(tmp_path):
    tasks = [TASK_33, TASK_38]
    source = _write_public_source(tmp_path, tasks, ["33", "38"])
    manifest = _write_manifest(tmp_path, ["33", "38"])
    return source, manifest


def test_adapter_rejects_revision_mismatch(tmp_path):
    from budgettrace.public_replay import AdapterError, load_tau3_reference_tasks

    source = _write_public_source(tmp_path, [TASK_33], ["33"])
    manifest = _write_manifest(tmp_path, ["33"])

    with pytest.raises(AdapterError) as excinfo:
        load_tau3_reference_tasks(source, manifest, observed_commit="deadbeef")

    assert excinfo.value.code == "source_revision_mismatch"


def test_adapter_rejects_missing_task(tmp_path):
    from budgettrace.public_replay import AdapterError, load_tau3_reference_tasks

    source = _write_public_source(tmp_path, [TASK_33], ["33"])
    manifest = _write_manifest(tmp_path, ["33", "99"])

    with pytest.raises(AdapterError) as excinfo:
        load_tau3_reference_tasks(source, manifest, observed_commit=COMMIT)

    assert excinfo.value.code == "selected_task_not_found"
    assert excinfo.value.task_id == "99"


def test_adapter_rejects_split_mismatch(tmp_path):
    from budgettrace.public_replay import AdapterError, load_tau3_reference_tasks

    source = _write_public_source(tmp_path, [TASK_33], ["33"])
    tasks = json.loads((source / "data/tau2/domains/retail/tasks.json").read_text(encoding="utf-8"))
    tasks.append(TASK_38)
    (source / "data/tau2/domains/retail/tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    manifest = _write_manifest(tmp_path, ["33", "38"])

    with pytest.raises(AdapterError) as excinfo:
        load_tau3_reference_tasks(source, manifest, observed_commit=COMMIT)

    assert excinfo.value.code == "selected_task_not_in_split"
    assert excinfo.value.task_id == "38"


def test_adapter_rejects_empty_declared_reward_criteria(tmp_path):
    from budgettrace.public_replay import AdapterError, load_tau3_reference_tasks

    broken = json.loads(json.dumps(TASK_38))
    broken["evaluation_criteria"]["nl_assertions"] = []
    source = _write_public_source(tmp_path, [TASK_33, broken], ["33", "38"])
    manifest = _write_manifest(tmp_path, ["33", "38"])

    with pytest.raises(AdapterError) as excinfo:
        load_tau3_reference_tasks(source, manifest, observed_commit=COMMIT)

    assert excinfo.value.code == "reward_criteria_empty"
    assert excinfo.value.task_id == "38"


def test_adapter_rejects_task_without_reference_actions(tmp_path):
    from budgettrace.public_replay import AdapterError, load_tau3_reference_tasks

    broken = json.loads(json.dumps(TASK_33))
    broken["evaluation_criteria"]["actions"] = []
    source = _write_public_source(tmp_path, [broken], ["33"])
    manifest = _write_manifest(tmp_path, ["33"])

    with pytest.raises(AdapterError) as excinfo:
        load_tau3_reference_tasks(source, manifest, observed_commit=COMMIT)

    assert excinfo.value.code == "reference_actions_missing"
    assert excinfo.value.task_id == "33"


def test_adapter_rejects_excluded_known_issue_task(tmp_path):
    from budgettrace.public_replay import AdapterError, load_tau3_reference_tasks

    source = _write_public_source(tmp_path, [TASK_33], ["33"])
    manifest_path = _write_manifest(tmp_path, ["33"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["excluded_known_issue_ids"] = ["33"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AdapterError) as excinfo:
        load_tau3_reference_tasks(source, manifest_path, observed_commit=COMMIT)

    assert excinfo.value.code == "selected_task_known_issue"
    assert excinfo.value.task_id == "33"


def test_adapter_preserves_provenance_fields(tmp_path):
    from budgettrace.public_replay import load_tau3_reference_tasks

    source, manifest = _happy_source_and_manifest(tmp_path)

    tasks = load_tau3_reference_tasks(source, manifest, observed_commit=COMMIT)

    assert [task.task_id for task in tasks] == ["33", "38"]
    first = tasks[0]
    assert first.upstream_commit == COMMIT
    assert first.split == "test"
    assert first.domain == "retail"
    assert first.source_path == "data/tau2/domains/retail/tasks.json"
    assert first.reward_basis == ["DB"]
    assert first.provenance == "public-upstream"


def test_gold_path_replay_replays_official_reference_actions(tmp_path):
    from budgettrace.public_replay import load_tau3_reference_tasks, replay_tau3_reference

    source, manifest = _happy_source_and_manifest(tmp_path)
    tasks = load_tau3_reference_tasks(source, manifest, observed_commit=COMMIT)

    result = replay_tau3_reference(tasks)

    assert result.provenance == "public-reference-replay"
    assert result.dataset_id == "tau3-retail-audited-v1"
    assert result.upstream_commit == COMMIT
    assert result.replayed_count == 2
    assert result.failed_count == 0
    assert result.skipped_count == 0
    assert result.reference_action_count == 3

    record_33 = result.records[0]
    assert record_33["task_id"] == "33"
    assert record_33["status"] == "replayed"
    assert record_33["protocol_success"] is True
    assert record_33["reference_actions"] == 2
    assert record_33["replayed_actions"] == 2
    assert record_33["budget_violation"] is False
    assert record_33["provenance"] == "public-reference-replay"
    scoring = record_33["scoring_basis"]
    assert scoring == {"DB": "reference-action-replay"}

    record_38 = result.records[1]
    assert record_38["scoring_basis"]["COMMUNICATE"] == "declared-not-executed-offline"
    assert record_38["scoring_basis"]["NL_ASSERTION"] == "declared-not-executed-offline"


def test_gold_path_journal_round_trips_through_existing_replay(tmp_path):
    from budgettrace.public_replay import load_tau3_reference_tasks, replay_tau3_reference

    source, manifest = _happy_source_and_manifest(tmp_path)
    tasks = load_tau3_reference_tasks(source, manifest, observed_commit=COMMIT)
    result = replay_tau3_reference(tasks)

    journal_jsonl = result.records[0]["journal_jsonl"]
    from budgettrace.replay import replay_run

    card = replay_run(journal_jsonl)

    assert card.task_id == "33"
    assert card.terminal_reason == "reference_replay_complete"
    assert card.success is True
    assert card.event_count == 4  # task_started + 2 actions + terminal


def test_gold_path_replay_is_deterministic(tmp_path):
    from budgettrace.public_replay import load_tau3_reference_tasks, replay_tau3_reference

    source, manifest = _happy_source_and_manifest(tmp_path)
    tasks = load_tau3_reference_tasks(source, manifest, observed_commit=COMMIT)

    first = replay_tau3_reference(tasks).to_dict()
    second = replay_tau3_reference(tasks).to_dict()

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_cli_public_replay_writes_machine_readable_reports(tmp_path, capsys):
    from budgettrace.cli import main

    source, manifest = _happy_source_and_manifest(tmp_path)
    output = tmp_path / "replay-out"

    exit_code = main(
        [
            "public-replay",
            "--source",
            str(source),
            "--manifest",
            str(manifest),
            "--observed-commit",
            COMMIT,
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"] == "public-reference-replay"
    assert summary["replayed_count"] == 2
    assert summary["disclaimer"] != ""
    records = json.loads((output / "records.json").read_text(encoding="utf-8"))
    assert len(records) == 2
    assert (output / "report.md").exists()
    assert (output / "journals" / "task-33.jsonl").exists()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "not an official leaderboard" in report.lower()


def test_cli_public_replay_fails_on_adapter_error(tmp_path, capsys):
    from budgettrace.cli import main

    source = _write_public_source(tmp_path, [TASK_33], ["33"])
    manifest = _write_manifest(tmp_path, ["33", "99"])
    output = tmp_path / "replay-out"

    exit_code = main(
        [
            "public-replay",
            "--source",
            str(source),
            "--manifest",
            str(manifest),
            "--observed-commit",
            COMMIT,
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "selected_task_not_found"
    assert not output.exists()
