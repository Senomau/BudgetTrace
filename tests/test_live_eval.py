"""TDD tests for the live eval matrix (N tasks x K trials, transport injected)."""

import json

import pytest

from test_live_smoke import (
    FAKE_KEY,
    VALID_ENV,
    _config,
    _fake_response,
    _fake_task,
    _fake_transport,
    _limits,
)


def _sequential_transport(responses):
    calls = []

    def transport(config, payload):
        calls.append({"payload": payload, "config": config})
        return responses[len(calls) - 1]

    transport.calls = calls
    return transport


def test_eval_runs_full_task_trial_matrix():
    from budgettrace.live_eval import run_live_eval

    transport = _fake_transport()
    tasks = [_fake_task("70"), _fake_task("33")]
    result = run_live_eval(_config(), tasks, trials=2, limits=_limits(), transport=transport)

    assert len(transport.calls) == 4
    records = result["records"]
    assert [record["status"] for record in records] == ["completed"] * 4
    assert [record["trial"] for record in records] == [1, 2, 1, 2]
    assert len({record["run_id"] for record in records}) == 4
    for record in records:
        assert record["provenance"] == "live-model-run"
        assert record["scored"] is False

    summary = result["summary"]
    assert summary["calls_planned"] == 4
    assert summary["calls_completed"] == 4
    assert summary["calls_failed"] == 0
    assert summary["calls_skipped"] == 0
    assert summary["tokens"] == {"prompt": 400, "completion": 200, "total": 600}
    assert summary["cost_micros_total"] == 4 * 330
    assert summary["cost_usd_estimate"] == round(4 * 330 / 1_000_000, 6)
    assert summary["truncated_count"] == 0
    assert summary["truncation_rate"] == 0.0
    assert summary["provenance"] == "live-model-run"
    assert summary["scored"] is False
    assert FAKE_KEY not in json.dumps(result)


def test_eval_counts_truncation_rate():
    from budgettrace.live_eval import run_live_eval

    first = _fake_response()
    second = _fake_response()
    second["choices"][0]["finish_reason"] = "length"
    transport = _sequential_transport([first, second])

    result = run_live_eval(_config(), [_fake_task("70")], trials=2, limits=_limits(), transport=transport)
    summary = result["summary"]
    assert summary["calls_completed"] == 2
    assert summary["truncated_count"] == 1
    assert summary["truncation_rate"] == 0.5
    assert result["records"][1]["truncated"] is True
    assert result["records"][1]["finish_reason"] == "length"


def test_eval_stops_issuing_calls_after_budget_cap():
    from budgettrace.live_eval import run_live_eval

    transport = _fake_transport()
    tasks = [_fake_task("70"), _fake_task("33")]
    # cap of 1 USD-micro: the first completed call (330 micros) exhausts it
    result = run_live_eval(
        _config(), tasks, trials=2, limits=_limits(), transport=transport, budget_usd=0.000001
    )

    assert len(transport.calls) == 1
    records = result["records"]
    assert [record["status"] for record in records] == [
        "completed",
        "skipped_budget_cap",
        "skipped_budget_cap",
        "skipped_budget_cap",
    ]
    summary = result["summary"]
    assert summary["calls_completed"] == 1
    assert summary["calls_skipped"] == 3
    assert summary["budget_usd_cap"] == 0.000001
    for record in records:
        assert record["provenance"] == "live-model-run"
        assert record["scored"] is False


def test_eval_continues_after_call_error():
    from budgettrace.live_eval import run_live_eval
    from budgettrace.live_smoke import SmokeError

    class _FlakyTransport:
        def __init__(self):
            self.calls = []

        def __call__(self, config, payload):
            self.calls.append({"payload": payload, "config": config})
            if len(self.calls) == 2:
                raise SmokeError("http_error", "HTTP 500: boom")
            return _fake_response()

    transport = _FlakyTransport()
    result = run_live_eval(_config(), [_fake_task("70")], trials=3, limits=_limits(), transport=transport)

    records = result["records"]
    assert len(transport.calls) == 3  # a failed call does not abort the matrix
    assert records[0]["status"] == "completed"
    assert records[1]["status"] == "error"
    assert records[1]["error"]["code"] == "http_error"
    assert records[2]["status"] == "completed"
    summary = result["summary"]
    assert summary["calls_completed"] == 2
    assert summary["calls_failed"] == 1
    assert summary["cost_micros_total"] == 660  # only completed calls bill


def test_eval_writes_reports(tmp_path):
    from budgettrace.live_eval import run_live_eval, write_eval_reports

    result = run_live_eval(_config(), [_fake_task("70")], trials=2, limits=_limits(), transport=_fake_transport())
    write_eval_reports(result, tmp_path)

    records = json.loads((tmp_path / "records.json").read_text(encoding="utf-8"))
    assert len(records) == 2
    assert all("journal_jsonl" not in record for record in records)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["calls_completed"] == 2
    assert FAKE_KEY not in json.dumps(summary)
    journal_text = (tmp_path / "journals" / "task-70-trial1.jsonl").read_text(encoding="utf-8")
    assert journal_text == result["records"][0]["journal_jsonl"]
    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "not a benchmark" in report_md
    assert "live-eval-v1" in report_md
    assert FAKE_KEY not in report_md


def test_public_writer_redacts_content_preview_and_journal(tmp_path):
    from budgettrace.live_eval import write_eval_reports

    secret = "Bearer" + " " + "abcdefghijklmnopqrstuvwxyz123456"
    result = {
        "records": [
            {
                "task_id": "70",
                "run_id": "run-secret",
                "domain": "retail",
                "split": "test",
                "model": "model",
                "trial": 1,
                "status": "completed",
                "tokens": {"prompt": 1, "completion": 1, "total": 2},
                "cost_micros": 1,
                "cost_usd_estimate": 0.000001,
                "finish_reason": "stop",
                "truncated": False,
                "content_preview": secret,
                "journal_jsonl": "{\"payload\":{\"message\":\"" + secret + "\"}}",
                "scored": False,
                "provenance": "live-model-run",
            }
        ],
        "summary": {
            "policy_version": "live-eval-v1",
            "provenance": "live-model-run",
            "scored": False,
            "model": "model",
            "endpoint": "https://example.com/chat/completions",
            "task_count": 1,
            "trials": 1,
            "calls_planned": 1,
            "calls_completed": 1,
            "calls_failed": 0,
            "calls_skipped": 0,
            "tokens": {"prompt": 1, "completion": 1, "total": 2},
            "cost_micros_total": 1,
            "cost_usd_estimate": 0.000001,
            "truncated_count": 0,
            "truncation_rate": 0.0,
            "budget_usd_cap": None,
            "disclaimer": "not a benchmark",
        },
    }

    write_eval_reports(result, tmp_path)

    for path in [
        tmp_path / "records.json",
        tmp_path / "summary.json",
        tmp_path / "report.md",
        tmp_path / "journals" / "task-70-trial1.jsonl",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "Bearer" + " " not in text
        assert secret not in text


def test_cli_live_eval_matrix(monkeypatch, capsys, tmp_path):
    from budgettrace import cli
    from budgettrace.live_smoke import DEFAULT_MAX_OUTPUT_TOKENS

    transport = _fake_transport()
    monkeypatch.setattr(
        cli, "load_tau3_reference_tasks", lambda source, manifest: [_fake_task("70"), _fake_task("33")]
    )
    monkeypatch.setattr("budgettrace.live_smoke._http_transport", transport)
    for name, value in VALID_ENV.items():
        monkeypatch.setenv(name, value)
    output = tmp_path / "eval-out"
    exit_code = cli.main(
        [
            "live-eval",
            "--source", "unused",
            "--manifest", "unused",
            "--output", str(output),
            "--trials", "2",
        ]
    )
    assert exit_code == 0
    assert len(transport.calls) == 4
    assert transport.calls[0]["payload"]["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
    captured = capsys.readouterr().out
    assert FAKE_KEY not in captured
    assert (output / "records.json").exists()
    assert (output / "summary.json").exists()
    assert (output / "journals" / "task-33-trial2.jsonl").exists()


def test_cli_live_eval_selects_explicit_tasks_with_default_trials(monkeypatch, capsys):
    from budgettrace import cli

    transport = _fake_transport()
    monkeypatch.setattr(
        cli, "load_tau3_reference_tasks", lambda source, manifest: [_fake_task("70"), _fake_task("33")]
    )
    monkeypatch.setattr("budgettrace.live_smoke._http_transport", transport)
    for name, value in VALID_ENV.items():
        monkeypatch.setenv(name, value)
    exit_code = cli.main(
        ["live-eval", "--source", "unused", "--manifest", "unused", "--tasks", "33"]
    )
    assert exit_code == 0
    assert len(transport.calls) == 3  # default --trials is 3


def test_cli_live_eval_missing_env_fails(monkeypatch, capsys, tmp_path):
    from budgettrace import cli

    for name in VALID_ENV:
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "eval-out"
    exit_code = cli.main(
        ["live-eval", "--source", "unused", "--manifest", "unused", "--output", str(output)]
    )
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["error"]["code"] == "missing_env_var"
    assert not output.exists()
