"""TDD tests for the live single-call model smoke (transport injected, no real network)."""

import io
import http.client
import json
import ssl
import urllib.error

import pytest

FAKE_KEY = "unit-test-live-key-1234567890"
VALID_ENV = {
    "BUDGETTRACE_LIVE_BASE_URL": "https://api.deepseek.com",
    "BUDGETTRACE_LIVE_MODEL": "deepseek-v4-pro",
    "BUDGETTRACE_LIVE_API_KEY": FAKE_KEY,
    "BUDGETTRACE_LIVE_PRICE_INPUT_PER_MTOK": "1.32",
    "BUDGETTRACE_LIVE_PRICE_OUTPUT_PER_MTOK": "3.96",
}


def _config(env=None):
    from budgettrace.live_config import load_live_config

    return load_live_config(env if env is not None else VALID_ENV)


def _fake_task(task_id="70"):
    from budgettrace.public_replay import Tau3ReferenceTask

    return Tau3ReferenceTask(
        task_id=task_id,
        dataset_id="tau3-retail-audited-v1",
        domain="retail",
        split="test",
        upstream_commit="fc0055dc4e0a316c3f83133267fbd6faaa770992",
        source_path="data/tau2/domains/retail/tasks.json",
        reward_basis=["DB"],
        actions=[{"action_id": "1", "name": "get_order", "arguments": {"order_id": "1"}}],
        communicate_info=[],
        nl_assertions=[],
        raw={"id": task_id, "instructions": ["Reply with the single word OK."]},
    )


def _fake_task_nested(task_id="70", raw=None):
    """τ³-bench shape: instructions nested under user_scenario.instructions."""
    from budgettrace.public_replay import Tau3ReferenceTask

    if raw is None:
        raw = {
            "id": task_id,
            "description": {"purpose": None, "relevant_policies": None, "notes": None},
            "user_scenario": {
                "instructions": {
                    "task_instructions": "You are impatient, confident, direct, messy.",
                    "reason_for_call": "You received a helmet but are not happy and want an exchange.",
                    "known_info": "You are Test User in ZIP code 00000.",
                    "unknown_info": "You do not remember your email address.",
                }
            },
        }

    return Tau3ReferenceTask(
        task_id=task_id,
        dataset_id="tau3-retail-audited-v1",
        domain="retail",
        split="test",
        upstream_commit="fc0055dc4e0a316c3f83133267fbd6faaa770992",
        source_path="data/tau2/domains/retail/tasks.json",
        reward_basis=["DB"],
        actions=[{"action_id": "1", "name": "get_order", "arguments": {"order_id": "1"}}],
        communicate_info=[],
        nl_assertions=[],
        raw=raw,
    )


def _limits(tokens=100000, steps=10, time_ms=60000, cost_micros=10**9):
    from budgettrace.contracts import BudgetLimits

    return BudgetLimits(steps=steps, tokens=tokens, time_ms=time_ms, cost_micros=cost_micros)


def _fake_response():
    return {
        "id": "cmpl-test-1",
        "model": "deepseek-v4-pro",
        "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def _fake_transport(response=None, error=None):
    calls = []

    def transport(config, payload):
        calls.append({"payload": payload, "config": config})
        if error is not None:
            raise error
        return response if response is not None else _fake_response()

    transport.calls = calls
    return transport


def test_chat_endpoint_join():
    from budgettrace.live_smoke import chat_endpoint

    assert chat_endpoint("https://api.deepseek.com") == "https://api.deepseek.com/chat/completions"
    assert chat_endpoint("https://api.deepseek.com/") == "https://api.deepseek.com/chat/completions"
    assert (
        chat_endpoint("https://api.xiaomimimo.com/v1")
        == "https://api.xiaomimimo.com/v1/chat/completions"
    )


def test_cost_computation_exact():
    from budgettrace.live_smoke import estimate_cost_micros

    # (100 * 1.32 + 50 * 3.96) = 330 USD-micros
    assert estimate_cost_micros(100, 50, 1.32, 3.96) == 330
    assert estimate_cost_micros(0, 0, 1.32, 3.96) == 0


def test_prompt_source_fallback():
    from budgettrace.live_smoke import prompt_from_task

    task = _fake_task()
    text, source = prompt_from_task(task)
    assert source == "instructions"
    assert "OK" in text

    raw_only = _fake_task()
    object.__setattr__(raw_only, "raw", {"id": "70"})
    text, source = prompt_from_task(raw_only)
    assert source == "raw-json"


def test_smoke_success_with_fake_transport():
    from budgettrace.live_smoke import run_live_smoke

    transport = _fake_transport()
    config = _config()
    task = _fake_task()
    record = run_live_smoke(config, task, limits=_limits(), transport=transport)

    assert record["status"] == "completed"
    assert record["provenance"] == "live-model-run"
    assert record["scored"] is False
    assert record["tokens"]["prompt"] == 100
    assert record["tokens"]["completion"] == 50
    assert record["tokens"]["total"] == 150
    assert record["cost_micros"] == 330
    assert record["event_count"] == 3

    # transport received the key at call time, and the payload never carries it
    assert len(transport.calls) == 1
    assert transport.calls[0]["config"].api_key == FAKE_KEY
    payload = transport.calls[0]["payload"]
    assert FAKE_KEY not in json.dumps(payload)
    assert payload["model"] == "deepseek-v4-pro"

    dumped = json.dumps(record) + repr(config)
    assert FAKE_KEY not in dumped

    journal_jsonl = record["journal_jsonl"]
    from budgettrace.events import EventJournal

    journal = EventJournal.from_jsonl(journal_jsonl)
    events = journal.events()
    assert [event.event_type for event in events] == ["task_started", "model_call", "terminal"]
    model_call = events[1]
    assert model_call.budget_delta.tokens == 150
    assert model_call.budget_delta.cost_micros == 330
    assert model_call.budget_delta.steps == 1
    assert model_call.payload["prompt_tokens"] == 100
    assert events[0].payload["provenance"] == "live-model-run"
    assert events[0].payload["prompt_source"] == "instructions"
    assert events[2].payload["scored"] is False
    assert record["truncated"] is False
    assert record["reasoning_tokens"] is None


def test_budget_exceeded_on_tiny_limit():
    from budgettrace.live_smoke import SmokeError, run_live_smoke

    transport = _fake_transport()
    with pytest.raises(SmokeError) as excinfo:
        run_live_smoke(_config(), _fake_task(), limits=_limits(tokens=10), transport=transport)
    assert excinfo.value.to_dict()["code"] == "budget_exceeded"
    assert excinfo.value.to_dict()["task_id"] == "70"


def test_http_error_maps_to_machine_readable():
    from budgettrace.live_smoke import SmokeError, run_live_smoke

    error = urllib.error.HTTPError(
        "https://api.deepseek.com/chat/completions",
        401,
        "Unauthorized",
        {},
        io.BytesIO(b'{"error": {"message": "bad key"}}'),
    )
    with pytest.raises(SmokeError) as excinfo:
        run_live_smoke(_config(), _fake_task(), limits=_limits(), transport=_fake_transport(error=error))
    payload = excinfo.value.to_dict()
    assert payload["code"] == "http_error"
    assert "401" in payload["message"]
    assert FAKE_KEY not in json.dumps(payload)


def test_missing_usage_fails_fast():
    from budgettrace.live_smoke import SmokeError, run_live_smoke

    response = _fake_response()
    response.pop("usage")
    with pytest.raises(SmokeError) as excinfo:
        run_live_smoke(_config(), _fake_task(), limits=_limits(), transport=_fake_transport(response=response))
    assert excinfo.value.to_dict()["code"] == "response_missing_usage"


def test_select_task_filters_audited_subset():
    from budgettrace.live_smoke import SmokeError, select_task

    tasks = [_fake_task("70"), _fake_task("33")]
    assert select_task(tasks, "33").task_id == "33"
    with pytest.raises(SmokeError) as excinfo:
        select_task(tasks, "999")
    assert excinfo.value.to_dict()["code"] == "task_not_selected"


def test_write_smoke_reports(tmp_path):
    from budgettrace.live_smoke import run_live_smoke, write_smoke_reports

    record = run_live_smoke(_config(), _fake_task(), limits=_limits(), transport=_fake_transport())
    write_smoke_reports(record, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_id"] == "70"
    assert summary["provenance"] == "live-model-run"
    assert "journal_jsonl" not in summary
    journal_text = (tmp_path / "journals" / "task-70.jsonl").read_text(encoding="utf-8")
    assert journal_text == record["journal_jsonl"]
    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "not a benchmark" in report_md
    assert FAKE_KEY not in report_md


def test_cli_live_smoke_success(monkeypatch, capsys, tmp_path):
    from budgettrace import cli

    fake_transport = _fake_transport()
    monkeypatch.setattr(cli, "load_tau3_reference_tasks", lambda source, manifest: [_fake_task()])
    monkeypatch.setattr("budgettrace.live_smoke._http_transport", fake_transport)
    for name, value in VALID_ENV.items():
        monkeypatch.setenv(name, value)
    output = tmp_path / "smoke-out"
    exit_code = cli.main(
        [
            "live-smoke",
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


def test_cli_defaults_max_output_tokens_to_live_smoke_default(monkeypatch, capsys):
    from budgettrace import cli
    from budgettrace.live_smoke import DEFAULT_MAX_OUTPUT_TOKENS

    fake_transport = _fake_transport()
    monkeypatch.setattr(cli, "load_tau3_reference_tasks", lambda source, manifest: [_fake_task()])
    monkeypatch.setattr("budgettrace.live_smoke._http_transport", fake_transport)
    for name, value in VALID_ENV.items():
        monkeypatch.setenv(name, value)
    exit_code = cli.main(
        ["live-smoke", "--source", "unused", "--manifest", "unused", "--task", "70"]
    )
    assert exit_code == 0
    assert fake_transport.calls[0]["payload"]["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


def test_prompt_from_nested_user_scenario_instructions():
    from budgettrace.live_smoke import prompt_from_task

    text, source = prompt_from_task(_fake_task_nested())
    assert source == "instructions"
    assert "impatient, confident, direct" in text
    assert "helmet" in text
    assert "Known information: You are Test User in ZIP code 00000." in text
    assert "Unknown information: You do not remember your email address." in text
    assert "Do not invent unknown identifiers" in text

    # list-form task_instructions is also extracted from the nested path
    list_task = _fake_task_nested(
        raw={"id": "70", "user_scenario": {"instructions": {"task_instructions": ["Line one.", "Line two."]}}}
    )
    text, source = prompt_from_task(list_task)
    assert source == "instructions"
    assert "Line one." in text
    assert "Line two." in text


def test_default_max_output_tokens_supports_reasoning_models():
    from budgettrace.live_smoke import DEFAULT_MAX_OUTPUT_TOKENS

    assert DEFAULT_MAX_OUTPUT_TOKENS >= 1024


def test_truncated_reasoning_response_recorded():
    from budgettrace.events import EventJournal
    from budgettrace.live_smoke import run_live_smoke

    response = {
        "id": "cmpl-reasoning-1",
        "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": 420,
            "completion_tokens": 256,
            "total_tokens": 676,
            "completion_tokens_details": {"reasoning_tokens": 230},
        },
    }
    record = run_live_smoke(
        _config(), _fake_task_nested(), limits=_limits(), transport=_fake_transport(response=response)
    )
    assert record["finish_reason"] == "length"
    assert record["truncated"] is True
    assert record["content_preview"] == ""
    assert record["reasoning_tokens"] == 230

    events = EventJournal.from_jsonl(record["journal_jsonl"]).events()
    assert events[0].payload["prompt_source"] == "instructions"
    assert events[1].payload["finish_reason"] == "length"
    assert events[1].payload["truncated"] is True
    assert events[1].payload["reasoning_tokens"] == 230


def test_report_warns_on_truncation(tmp_path):
    from budgettrace.live_smoke import run_live_smoke, write_smoke_reports

    response = _fake_response()
    response["choices"][0]["finish_reason"] = "length"
    record = run_live_smoke(_config(), _fake_task(), limits=_limits(), transport=_fake_transport(response=response))
    write_smoke_reports(record, tmp_path)
    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "truncated" in report_md


def test_http_transport_uses_configured_timeout(monkeypatch):
    from budgettrace import live_smoke

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["timeout"] = timeout
        return io.BytesIO(b"{}")

    monkeypatch.setattr(live_smoke.urllib.request, "urlopen", fake_urlopen)
    config = _config({**VALID_ENV, "BUDGETTRACE_LIVE_TIMEOUT": "45"})
    live_smoke._http_transport(config, {"model": "m"})
    assert captured["timeout"] == 45


def test_http_transport_retries_once_on_read_timeout(monkeypatch):
    from budgettrace import live_smoke

    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("The read operation timed out")
        return io.BytesIO(b'{"id": "cmpl-retry"}')

    monkeypatch.setattr(live_smoke.urllib.request, "urlopen", fake_urlopen)
    response = live_smoke._http_transport(_config(), {"model": "m"})
    assert len(calls) == 2
    assert response["id"] == "cmpl-retry"


def test_http_transport_retries_once_on_ssl_eof(monkeypatch):
    from budgettrace import live_smoke

    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise urllib.error.URLError(ssl.SSLEOFError("EOF occurred in violation of protocol"))
        return io.BytesIO(b'{"id": "cmpl-ssl-retry"}')

    monkeypatch.setattr(live_smoke.urllib.request, "urlopen", fake_urlopen)
    response = live_smoke._http_transport(_config(), {"model": "m"})
    assert len(calls) == 2
    assert response["id"] == "cmpl-ssl-retry"


def test_http_transport_retries_once_on_remote_disconnect(monkeypatch):
    from budgettrace import live_smoke

    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return io.BytesIO(b'{"id": "cmpl-remote-retry"}')

    monkeypatch.setattr(live_smoke.urllib.request, "urlopen", fake_urlopen)
    response = live_smoke._http_transport(_config(), {"model": "m"})
    assert len(calls) == 2
    assert response["id"] == "cmpl-remote-retry"


def test_http_transport_two_timeouts_raise_smoke_error(monkeypatch):
    from budgettrace import live_smoke
    from budgettrace.live_smoke import SmokeError

    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(timeout)
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(live_smoke.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SmokeError) as excinfo:
        live_smoke._http_transport(_config(), {"model": "m"})
    assert excinfo.value.to_dict()["code"] == "http_error"
    assert len(calls) == 2


def test_http_transport_non_transient_oserror_not_retried(monkeypatch):
    from budgettrace import live_smoke
    from budgettrace.live_smoke import SmokeError

    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(timeout)
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(live_smoke.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SmokeError) as excinfo:
        live_smoke._http_transport(_config(), {"model": "m"})
    assert excinfo.value.to_dict()["code"] == "http_error"
    assert len(calls) == 1


def test_cli_live_smoke_failure_no_output_written(monkeypatch, capsys, tmp_path):
    from budgettrace import cli

    for name in VALID_ENV:
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "smoke-out"
    exit_code = cli.main(
        [
            "live-smoke",
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
