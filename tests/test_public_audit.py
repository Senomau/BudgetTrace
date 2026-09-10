import json


COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"


def _write_public_source(tmp_path, *, valid=True):
    source = tmp_path / "tau2-bench"
    data_dir = source / "data" / "tau2" / "domains" / "retail"
    data_dir.mkdir(parents=True)
    task = {
        "id": "33",
        "evaluation_criteria": {
            "actions": [{"name": "lookup", "arguments": {}}],
            "reward_basis": ["DB"],
            "nl_assertions": [],
            "communicate_info": [],
        },
    }
    if not valid:
        task["id"] = "10"
        task["evaluation_criteria"]["reward_basis"] = ["DB", "NL_ASSERTION"]
        task["evaluation_criteria"]["nl_assertions"] = []
    (data_dir / "tasks.json").write_text(json.dumps([task]), encoding="utf-8")
    (data_dir / "split_tasks.json").write_text(
        json.dumps({"test": [task["id"]], "base": [task["id"]]}),
        encoding="utf-8",
    )
    return source


def _write_manifest(tmp_path):
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
                "task_ids": ["33"],
                "excluded_known_issue_ids": ["10"],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_public_audit_accepts_pinned_selected_task(tmp_path):
    from budgettrace.public_audit import audit_tau3_retail

    source = _write_public_source(tmp_path)
    manifest = _write_manifest(tmp_path)

    result = audit_tau3_retail(source, manifest, observed_commit=COMMIT)

    assert result.passed is True
    assert result.selected_task_count == 1
    assert result.source_task_count == 1
    assert result.findings == []


def test_public_audit_rejects_selected_known_issue_missing_from_source(tmp_path):
    from budgettrace.public_audit import audit_tau3_retail

    source = _write_public_source(tmp_path, valid=False)
    manifest = _write_manifest(tmp_path)

    result = audit_tau3_retail(source, manifest, observed_commit=COMMIT)

    assert result.passed is False
    codes = {finding.code for finding in result.findings}
    assert "selected_task_not_found" in codes


def test_public_audit_rejects_empty_declared_reward_criteria(tmp_path):
    from budgettrace.public_audit import audit_tau3_retail

    source = _write_public_source(tmp_path)
    task_path = source / "data" / "tau2" / "domains" / "retail" / "tasks.json"
    tasks = json.loads(task_path.read_text(encoding="utf-8"))
    tasks[0]["evaluation_criteria"]["reward_basis"] = ["DB", "NL_ASSERTION"]
    task_path.write_text(json.dumps(tasks), encoding="utf-8")
    manifest = _write_manifest(tmp_path)

    result = audit_tau3_retail(source, manifest, observed_commit=COMMIT)

    assert result.passed is False
    assert any(finding.code == "reward_criteria_empty" for finding in result.findings)


def test_public_audit_cli_writes_machine_readable_result(tmp_path, capsys):
    from budgettrace.cli import main

    source = _write_public_source(tmp_path)
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "audit.json"

    exit_code = main(
        [
            "public-audit",
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
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
