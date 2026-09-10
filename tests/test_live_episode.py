"""TDD tests for the live multi-turn episode runner.

Contract: run_live_episode drives Runtime's agent loop with LiveLLM against a
real (injected) transport, so a real T-step agent episode runs under the
standard BudgetLedger gate with provider-usage-based cost estimation. Episodes are informational:
``scored`` is always False and provenance stays ``live-model-run``.
"""

import json

from test_live_agent import _json_response
from test_live_eval import _sequential_transport
from test_live_smoke import FAKE_KEY, _config, _fake_response, _fake_task

EPISODE_TOOL_ACTION = {"kind": "tool_call", "tool": "noop", "args": {}}
EPISODE_FINISH_ACTION = {"kind": "finish", "score": 1.0}


def _limits(tokens=100000, steps=10, time_ms=10**9, cost_micros=10**9):
    from budgettrace.contracts import BudgetLimits

    return BudgetLimits(steps=steps, tokens=tokens, time_ms=time_ms, cost_micros=cost_micros)


def _run_episode(responses, limits=None, run_id=None, task=None):
    from budgettrace.live_episode import run_live_episode

    transport = _sequential_transport(responses)
    record = run_live_episode(
        _config(),
        task or _fake_task(),
        limits=limits or _limits(),
        transport=transport,
        run_id=run_id,
    )
    return record, transport


def test_finish_action_terminates_episode():
    record, transport = _run_episode(
        [_json_response(EPISODE_TOOL_ACTION), _json_response(EPISODE_FINISH_ACTION)],
        run_id="tau3-retail-test-live-episode-70-t1",
    )

    assert record["status"] == "completed"
    assert record["run_id"] == "tau3-retail-test-live-episode-70-t1"
    assert record["terminal_reason"] == "success"
    assert record["model_finished"] is True
    assert record["scored"] is False
    assert record["provenance"] == "live-model-run"
    assert record["steps_used"] == 2
    assert record["tokens_used"] == 300
    assert record["cost_micros"] == 660
    assert record["cost_usd_estimate"] == 0.00066
    assert record["billing_status"] == "estimated_from_provider_usage"
    # two live calls happened against the injected transport
    assert len(transport.calls) == 2
    assert FAKE_KEY not in json.dumps(record) + json.dumps(record["journal_jsonl"])


def test_reference_action_scorer_requires_business_action_and_finish():
    from dataclasses import replace
    from budgettrace.live_episode import run_live_episode

    task = replace(_fake_task(), actions=[{"action_id": "1", "name": "noop", "arguments": {}}])
    transport = _sequential_transport(
        [
            _json_response({"kind": "tool_call", "tool": "noop", "args": {}}),
            _json_response(EPISODE_FINISH_ACTION),
        ]
    )
    record = run_live_episode(
        _config(),
        task,
        limits=_limits(),
        transport=transport,
        scorer_mode="reference-actions",
    )

    assert record["scored"] is True
    assert record["scorer_version"] == "reference-action-contract-v1"
    assert record["business_success"] is True
    assert record["score"] == 1.0


def test_reference_action_scorer_marks_finish_without_action_as_false_stop():
    from dataclasses import replace
    from budgettrace.live_episode import run_live_episode

    record, _ = _run_episode([_json_response(EPISODE_FINISH_ACTION)])
    # default mode remains informational; opt into the contract scorer explicitly
    task = replace(_fake_task(), actions=[{"action_id": "1", "name": "noop", "arguments": {}}])
    exhausted = _fake_response()
    exhausted["choices"][0]["message"]["content"] = "not-json"
    scored = run_live_episode(
        _config(),
        task,
        limits=_limits(),
        transport=_sequential_transport([_json_response(EPISODE_FINISH_ACTION), exhausted]),
        scorer_mode="reference-actions",
    )

    assert record["scored"] is False
    assert scored["business_success"] is False
    assert scored["false_stop"] is True


def test_episode_budget_reservation_prevents_transport_when_estimate_unaffordable():
    from budgettrace.contracts import BudgetLimits

    limits = BudgetLimits(steps=10, tokens=100000, time_ms=10**9, cost_micros=500)
    record, transport = _run_episode(
        [_json_response(EPISODE_TOOL_ACTION), _json_response(EPISODE_TOOL_ACTION)],
        limits=limits,
    )

    assert len(transport.calls) == 0
    assert record["terminal_reason"] == "budget_rejected"
    assert record["cost_micros"] == 0
    assert record["over_budget"] is False
    assert record["over_budget_dimensions"] == ["cost_micros"]
    assert record["model_finished"] is False


def test_unparseable_content_terminates_script_exhausted_but_still_records_provider_usage():
    response = _fake_response()
    response["choices"][0]["message"]["content"] = "sorry, I cannot comply"
    record, _ = _run_episode([response])

    assert record["terminal_reason"] == "script_exhausted"
    assert record["model_finished"] is False
    assert record["status"] == "completed"
    assert record["cost_micros"] == 330
    assert record["billing_status"] == "estimated_from_provider_usage"


def test_episode_journal_records_provenance_and_costs():
    from budgettrace.events import EventJournal

    record, _ = _run_episode(
        [_json_response(EPISODE_TOOL_ACTION), _json_response(EPISODE_FINISH_ACTION)]
    )

    events = EventJournal.from_jsonl(record["journal_jsonl"]).events()
    assert [event.event_type for event in events] == [
        "task_started",
        "model_call",
        "tool_call",
        "tool_result",
        "model_call",
        "feedback",
        "terminal",
    ]
    started = events[0].payload
    assert started["provenance"] == "live-model-run"
    assert started["policy_version"] == "live-episode-v1"
    assert events[1].budget_delta.cost_micros == 330
    assert events[1].budget_delta.steps == 1
    assert events[2].payload["tool"] == "noop"
    assert events[2].payload["operation_id"] == events[3].payload["operation_id"]
    assert len(events[2].payload["args_hash"]) == 64
    assert events[5].payload["feedback_id"] == "finish"
    assert events[6].payload["scored"] is False
    assert events[6].payload["reason"] == "success"


def test_episode_contract_is_runtime_compliant(tmp_path):
    from budgettrace.live_episode import build_task_contract

    task = _fake_task()
    contract = build_task_contract(task, _limits(), str(tmp_path))

    assert contract.task_id == "70"
    assert contract.workspace_root == str(tmp_path)
    assert contract.allowed_tools == ["noop"]
    assert contract.tool_levels == {"noop": "L0"}
    assert contract.success_condition == "model_emitted_finish"
    assert contract.scorer_version == "unscored-v1"
    # tool_levels subset constraint holds
    assert set(contract.tool_levels) <= set(contract.allowed_tools)


def test_write_episode_reports(tmp_path):
    from budgettrace.live_episode import run_live_episode, write_episode_reports

    transport = _sequential_transport([_json_response(EPISODE_FINISH_ACTION)])
    record = run_live_episode(
        _config(), _fake_task(), limits=_limits(), transport=transport
    )
    write_episode_reports(record, tmp_path)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_id"] == "70"
    assert summary["scored"] is False
    assert summary["provenance"] == "live-model-run"
    assert "journal_jsonl" not in summary
    journal_text = (tmp_path / "journals" / "task-70.jsonl").read_text(encoding="utf-8")
    assert journal_text == record["journal_jsonl"]
    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "not a benchmark" in report_md
    assert FAKE_KEY not in report_md


def test_cli_live_episode_success(monkeypatch, capsys, tmp_path):
    from budgettrace import cli

    transport = _sequential_transport([_json_response(EPISODE_FINISH_ACTION)])
    monkeypatch.setattr(cli, "load_tau3_reference_tasks", lambda source, manifest: [_fake_task()])
    monkeypatch.setattr("budgettrace.live_episode._http_transport", transport)
    for name, value in _config_env().items():
        monkeypatch.setenv(name, value)
    output = tmp_path / "episode-out"
    exit_code = cli.main(
        [
            "live-episode",
            "--source", "unused",
            "--manifest", "unused",
            "--task", "70",
            "--output", str(output),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert FAKE_KEY not in captured
    assert (output / "summary.json").exists()
    assert (output / "journals" / "task-70.jsonl").exists()
    assert (output / "report.md").exists()


def test_cli_episode_time_budget_scales_with_timeout_and_retries(monkeypatch, capsys, tmp_path):
    # every step may legitimately consume up to two read-timeout windows
    # (one automatic retry), so the ledger time budget must scale with
    # steps and the configured timeout; the smoke-scale default tripped
    # the gate on a successful first call of a long-thinking episode
    from budgettrace import cli
    from budgettrace.events import EventJournal

    transport = _sequential_transport([_json_response(EPISODE_FINISH_ACTION)])
    monkeypatch.setattr(cli, "load_tau3_reference_tasks", lambda source, manifest: [_fake_task()])
    monkeypatch.setattr("budgettrace.live_episode._http_transport", transport)
    for name, value in _config_env().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("BUDGETTRACE_LIVE_TIMEOUT", "180")
    output = tmp_path / "episode-out"
    exit_code = cli.main(
        [
            "live-episode",
            "--source", "unused",
            "--manifest", "unused",
            "--task", "70",
            "--output", str(output),
        ]
    )
    assert exit_code == 0
    journal_text = (output / "journals" / "task-70.jsonl").read_text(encoding="utf-8")
    started = EventJournal.from_jsonl(journal_text).events()[0]
    budgets = started.payload["budgets"]
    # 8 steps * 2 attempts * 180000 ms read-timeout window each
    assert budgets["time_ms"] >= budgets["steps"] * 2 * 180 * 1000


def test_cli_live_episode_budget_window_steps_are_independent(monkeypatch, tmp_path):
    from budgettrace import cli
    from budgettrace.events import EventJournal

    transport = _sequential_transport(
        [
            _json_response(EPISODE_TOOL_ACTION),
            _json_response(EPISODE_TOOL_ACTION),
            _json_response(EPISODE_TOOL_ACTION),
            _json_response(EPISODE_FINISH_ACTION),
        ]
    )
    monkeypatch.setattr(cli, "load_tau3_reference_tasks", lambda source, manifest: [_fake_task()])
    monkeypatch.setattr("budgettrace.live_episode._http_transport", transport)
    for name, value in _config_env().items():
        monkeypatch.setenv(name, value)
    output = tmp_path / "episode-out"

    exit_code = cli.main(
        [
            "live-episode",
            "--source", "unused",
            "--manifest", "unused",
            "--task", "70",
            "--output", str(output),
            "--max-steps", "6",
            "--budget-window-steps", "2",
        ]
    )

    assert exit_code == 0
    events = EventJournal.from_jsonl((output / "journals" / "task-70.jsonl").read_text(encoding="utf-8")).events()
    decision = next(event for event in events if event.event_type == "decision")
    model_calls_before_decision = [
        event for event in events if event.event_type == "model_call" and event.sequence < decision.sequence
    ]
    assert len(model_calls_before_decision) == 2
    assert len(model_calls_before_decision) < 6


def test_cli_live_episode_rejects_invalid_budget_window_steps(monkeypatch, capsys):
    from budgettrace import cli

    for name, value in _config_env().items():
        monkeypatch.setenv(name, value)

    exit_code = cli.main(
        [
            "live-episode",
            "--source", "unused",
            "--manifest", "unused",
            "--task", "70",
            "--max-steps", "2",
            "--budget-window-steps", "2",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "invalid_budget_window_steps"


def test_cli_live_episode_missing_config_fails_fast(monkeypatch, capsys, tmp_path):
    from budgettrace import cli

    for name in _config_env():
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "episode-out"
    exit_code = cli.main(
        [
            "live-episode",
            "--source", "unused",
            "--manifest", "unused",
            "--task", "70",
            "--output", str(output),
        ]
    )
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["error"]["code"] == "missing_env_var"
    assert not output.exists()


def _config_env():
    from test_live_smoke import VALID_ENV

    return VALID_ENV
