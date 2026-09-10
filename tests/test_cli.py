import json


class GbkOnlyStdout:
    encoding = "gbk"

    def __init__(self):
        self.parts = []

    def write(self, text):
        text.encode(self.encoding)
        self.parts.append(text)

    def flush(self):
        pass


def test_cli_evaluate_prints_machine_readable_records(tmp_path, capsys):
    from budgettrace.cli import main

    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "tiny-001",
                    "budget_window_steps": 1,
                    "budgets": {"steps": 2, "tokens": 100, "time_ms": 100, "cost_micros": 100},
                    "script": [{"kind": "feedback", "feedback_id": "ok", "score": 1.0}],
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["evaluate", str(tasks_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output[0]["provenance"] == "offline"


def test_cli_json_print_survives_non_utf8_stdout(monkeypatch):
    from budgettrace import cli

    stream = GbkOnlyStdout()
    monkeypatch.setattr(cli.sys, "stdout", stream)

    cli._print_json({"label": "tau3", "text": "τ³-bench"})

    output = "".join(stream.parts)
    assert json.loads(output)["text"] == "τ³-bench"
    assert "\\u03c4" in output


def test_cli_report_writes_markdown_from_json_records(tmp_path):
    from budgettrace.cli import main

    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "records.json").write_text(
        json.dumps([{"policy": "fixed_steps", "task_count": 1, "final_score": 1.0}]),
        encoding="utf-8",
    )

    assert main(["report", str(run_dir)]) == 0
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "fixed_steps" in report


def test_cli_evaluate_output_dir_writes_records_summary_manifest_and_report(tmp_path, capsys):
    from budgettrace.cli import main

    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "tiny-001",
                    "budget_window_steps": 1,
                    "budgets": {"steps": 2, "tokens": 100, "time_ms": 100, "cost_micros": 100},
                    "script": [{"kind": "feedback", "feedback_id": "ok", "score": 1.0}],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    exit_code = main(["evaluate", str(tasks_path), "--output", str(output_dir)])
    capsys.readouterr()

    assert exit_code == 0
    assert (output_dir / "records.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "report.md").exists()


def test_cli_evaluate_accepts_trials_and_model_routes(tmp_path, capsys):
    from budgettrace import cli

    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "matrix-001",
                    "budget_window_steps": 1,
                    "budgets": {"steps": 2, "tokens": 100, "time_ms": 100, "cost_micros": 100},
                    "script": [{"kind": "feedback", "feedback_id": "ok", "score": 1.0}],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "matrix"

    assert cli.main(
        [
            "evaluate",
            str(tasks_path),
            "--trials",
            "2",
            "--model-routes",
            "route-a,route-b",
            "--output",
            str(output_dir),
        ]
    ) == 0
    capsys.readouterr()

    records = json.loads((output_dir / "records.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(records) == 16
    assert manifest["trials"] == 2
    assert manifest["model_routes"] == ["route-a", "route-b"]
