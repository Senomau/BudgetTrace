"""TDD tests for live-eval configuration loading (secrets stay out of chat/code)."""

import json
from pathlib import Path

import pytest

FAKE_KEY = "unit-test-live-key-1234567890"
VALID_ENV = {
    "BUDGETTRACE_LIVE_BASE_URL": "https://api.example.com/v1",
    "BUDGETTRACE_LIVE_MODEL": "gpt-4o-mini",
    "BUDGETTRACE_LIVE_API_KEY": FAKE_KEY,
    "BUDGETTRACE_LIVE_PRICE_INPUT_PER_MTOK": "0.15",
    "BUDGETTRACE_LIVE_PRICE_OUTPUT_PER_MTOK": "0.60",
}


def _load(env):
    from budgettrace.live_config import load_live_config

    return load_live_config(env)


def _cheap_env():
    return {
        "BUDGETTRACE_CHEAP_BASE_URL": "https://cheap.example.com/v1",
        "BUDGETTRACE_CHEAP_MODEL": "cheap-model",
        "BUDGETTRACE_CHEAP_API_KEY": "unit-test-cheap-key-1234567890",
        "BUDGETTRACE_CHEAP_PRICE_INPUT_PER_MTOK": "0.01",
        "BUDGETTRACE_CHEAP_PRICE_OUTPUT_PER_MTOK": "0.02",
        "BUDGETTRACE_CHEAP_TIMEOUT": "30",
    }


def test_missing_env_vars_reported():
    from budgettrace.live_config import LiveConfigError

    with pytest.raises(LiveConfigError) as excinfo:
        _load({})
    payload = excinfo.value.to_dict()
    assert payload["code"] == "missing_env_var"
    for name in VALID_ENV:
        assert name in payload["variables"]


def test_invalid_api_key_rejected():
    from budgettrace.live_config import LiveConfigError

    env = dict(VALID_ENV, BUDGETTRACE_LIVE_API_KEY="short")
    with pytest.raises(LiveConfigError) as excinfo:
        _load(env)
    assert excinfo.value.to_dict()["code"] == "invalid_api_key"

    env = dict(VALID_ENV, BUDGETTRACE_LIVE_API_KEY="has whitespace inside")
    with pytest.raises(LiveConfigError):
        _load(env)


def test_api_key_never_leaks_via_repr_or_dict():
    config = _load(VALID_ENV)
    dumped = repr(config) + str(config) + json.dumps(config.to_dict())
    assert FAKE_KEY not in dumped
    assert config.api_key == FAKE_KEY  # only in memory, in the field itself


def test_invalid_base_url_rejected():
    from budgettrace.live_config import LiveConfigError

    env = dict(VALID_ENV, BUDGETTRACE_LIVE_BASE_URL="ftp://api.example.com/v1")
    with pytest.raises(LiveConfigError) as excinfo:
        _load(env)
    assert excinfo.value.to_dict()["code"] == "invalid_base_url"


def test_base_url_markdown_quoting_stripped():
    from budgettrace.live_smoke import chat_endpoint

    env = dict(VALID_ENV, BUDGETTRACE_LIVE_BASE_URL="`https://api.deepseek.com`")
    config = _load(env)
    assert config.base_url == "https://api.deepseek.com"
    assert chat_endpoint(config.base_url) == "https://api.deepseek.com/chat/completions"


def test_base_url_boundary_quotes_and_whitespace_stripped():
    env = dict(VALID_ENV, BUDGETTRACE_LIVE_BASE_URL=" \"https://api.example.com/v1\" \r\n")
    config = _load(env)
    assert config.base_url == "https://api.example.com/v1"


def test_base_url_interior_backtick_still_rejected():
    from budgettrace.live_config import LiveConfigError

    env = dict(VALID_ENV, BUDGETTRACE_LIVE_BASE_URL="https://api.deep`seek.com")
    with pytest.raises(LiveConfigError) as excinfo:
        _load(env)
    assert excinfo.value.to_dict()["code"] == "invalid_base_url"


def test_invalid_price_rejected():
    from budgettrace.live_config import LiveConfigError

    for bad in ("0", "-1", "abc"):
        env = dict(VALID_ENV, BUDGETTRACE_LIVE_PRICE_INPUT_PER_MTOK=bad)
        with pytest.raises(LiveConfigError) as excinfo:
            _load(env)
        assert excinfo.value.to_dict()["code"] == "invalid_price"


def test_timeout_env_var_loaded_with_default():
    config = _load(VALID_ENV)
    assert config.timeout_seconds == 120
    config = _load(dict(VALID_ENV, BUDGETTRACE_LIVE_TIMEOUT="30"))
    assert config.timeout_seconds == 30


def test_invalid_timeout_rejected():
    from budgettrace.live_config import LiveConfigError

    for bad in ("abc", "0", "-5", "1.5"):
        env = dict(VALID_ENV, BUDGETTRACE_LIVE_TIMEOUT=bad)
        with pytest.raises(LiveConfigError) as excinfo:
            _load(env)
        assert excinfo.value.to_dict()["code"] == "invalid_timeout"


def test_valid_config_loads_and_masks():
    config = _load(VALID_ENV)
    assert config.base_url == "https://api.example.com/v1"
    assert config.model == "gpt-4o-mini"
    assert config.price_input_per_mtok == 0.15
    assert config.price_output_per_mtok == 0.60
    masked = config.to_dict()["api_key_masked"]
    assert FAKE_KEY not in masked
    assert masked.startswith("unit")
    assert masked.endswith("7890")
    assert "***" in masked


def test_gitignore_covers_env_files():
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    text = gitignore.read_text(encoding="utf-8")
    assert any(line.strip().startswith(".env") for line in text.splitlines())
    assert any(line.strip() == "!.env.example" for line in text.splitlines())


def test_cli_live_config_check_success(monkeypatch, capsys):
    from budgettrace.cli import main

    for name, value in VALID_ENV.items():
        monkeypatch.setenv(name, value)
    exit_code = main(["live-config-check"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert FAKE_KEY not in captured
    assert "gpt-4o-mini" in captured


def test_optional_cheap_config_is_all_or_none():
    from budgettrace.live_config import LiveConfigError, load_optional_cheap_config

    assert load_optional_cheap_config(VALID_ENV) is None
    with pytest.raises(LiveConfigError) as excinfo:
        load_optional_cheap_config(dict(VALID_ENV, BUDGETTRACE_CHEAP_MODEL="cheap-model"))

    payload = excinfo.value.to_dict()
    assert payload["code"] == "incomplete_cheap_config"
    assert "BUDGETTRACE_CHEAP_BASE_URL" in payload["variables"]


def test_optional_cheap_config_loads_and_masks_in_diagnostics(monkeypatch):
    from budgettrace.live_config import config_check_payload, load_optional_cheap_config

    env = dict(VALID_ENV, **_cheap_env())
    cheap = load_optional_cheap_config(env)
    assert cheap.model == "cheap-model"
    assert cheap.timeout_seconds == 30

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    payload = config_check_payload()
    dumped = json.dumps(payload)
    assert "cheap-model" in dumped
    assert _cheap_env()["BUDGETTRACE_CHEAP_API_KEY"] not in dumped
    assert payload["cheap_config"]["api_key_masked"].startswith("unit")


def test_cli_live_config_check_failure(monkeypatch, capsys):
    from budgettrace.cli import main

    for name in VALID_ENV:
        monkeypatch.delenv(name, raising=False)
    exit_code = main(["live-config-check"])
    captured = capsys.readouterr().out
    assert exit_code == 2
    payload = json.loads(captured)
    assert payload["passed"] is False
    assert payload["error"]["code"] == "missing_env_var"
