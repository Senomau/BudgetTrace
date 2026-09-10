"""Live-eval configuration loaded from environment variables only.

Secrets (API keys) never enter the codebase, reports, or CLI output:
values are read from the environment at runtime and always masked in
any derived payload. Missing or invalid configuration fails fast with
a machine-readable error, before any network call is attempted.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

ENV_BASE_URL = "BUDGETTRACE_LIVE_BASE_URL"
ENV_MODEL = "BUDGETTRACE_LIVE_MODEL"
ENV_API_KEY = "BUDGETTRACE_LIVE_API_KEY"
ENV_PRICE_INPUT = "BUDGETTRACE_LIVE_PRICE_INPUT_PER_MTOK"
ENV_PRICE_OUTPUT = "BUDGETTRACE_LIVE_PRICE_OUTPUT_PER_MTOK"
ENV_TIMEOUT = "BUDGETTRACE_LIVE_TIMEOUT"
CHEAP_ENV_BASE_URL = "BUDGETTRACE_CHEAP_BASE_URL"
CHEAP_ENV_MODEL = "BUDGETTRACE_CHEAP_MODEL"
CHEAP_ENV_API_KEY = "BUDGETTRACE_CHEAP_API_KEY"
CHEAP_ENV_PRICE_INPUT = "BUDGETTRACE_CHEAP_PRICE_INPUT_PER_MTOK"
CHEAP_ENV_PRICE_OUTPUT = "BUDGETTRACE_CHEAP_PRICE_OUTPUT_PER_MTOK"
CHEAP_ENV_TIMEOUT = "BUDGETTRACE_CHEAP_TIMEOUT"

ENV_VARS = (ENV_BASE_URL, ENV_MODEL, ENV_API_KEY, ENV_PRICE_INPUT, ENV_PRICE_OUTPUT)
CHEAP_ENV_VARS = (
    CHEAP_ENV_BASE_URL,
    CHEAP_ENV_MODEL,
    CHEAP_ENV_API_KEY,
    CHEAP_ENV_PRICE_INPUT,
    CHEAP_ENV_PRICE_OUTPUT,
)

_MIN_KEY_LENGTH = 8
DEFAULT_TIMEOUT_SECONDS = 120


class LiveConfigError(Exception):
    """Raised when live configuration is missing or invalid. Never carries secret values."""

    def __init__(self, code: str, message: str, variables: Optional[List[str]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.variables = variables or []

    def to_dict(self) -> Dict:
        return {"code": self.code, "message": self.message, "variables": list(self.variables)}


def mask_secret(value: str) -> str:
    """Return a fixed-width mask that never reconstructs the original value."""
    if len(value) <= _MIN_KEY_LENGTH:
        return "***"
    return f"{value[:4]}***{value[-4:]}"


@dataclass(frozen=True)
class LiveConfig:
    base_url: str
    model: str
    api_key: str
    price_input_per_mtok: float
    price_output_per_mtok: float
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    def __repr__(self) -> str:  # keep the key out of logs/tracebacks
        return (
            f"LiveConfig(base_url={self.base_url!r}, model={self.model!r}, "
            f"api_key={mask_secret(self.api_key)!r}, "
            f"price_input_per_mtok={self.price_input_per_mtok!r}, "
            f"price_output_per_mtok={self.price_output_per_mtok!r})"
        )

    __str__ = __repr__

    def to_dict(self) -> Dict:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key_masked": mask_secret(self.api_key),
            "price_input_per_mtok": self.price_input_per_mtok,
            "price_output_per_mtok": self.price_output_per_mtok,
            "timeout_seconds": self.timeout_seconds,
        }


def _require(name: str, env: Mapping[str, str], missing: List[str]) -> str:
    value = env.get(name)
    if value is None or value == "":
        missing.append(name)
        return ""
    return value


def _load_config(
    source: Mapping[str, str],
    *,
    base_url_name: str,
    model_name: str,
    api_key_name: str,
    price_input_name: str,
    price_output_name: str,
    timeout_name: str,
) -> LiveConfig:
    missing: List[str] = []
    base_url = _require(base_url_name, source, missing)
    model = _require(model_name, source, missing)
    api_key = _require(api_key_name, source, missing)
    price_input_raw = _require(price_input_name, source, missing)
    price_output_raw = _require(price_output_name, source, missing)
    if missing:
        raise LiveConfigError(
            "missing_env_var",
            "missing required environment variables: " + ", ".join(missing),
            missing,
        )

    # Copying a URL out of rendered markdown inline code often picks up
    # boundary backticks/quotes; strip those (and stray boundary whitespace)
    # before validation. Interior bad characters are still rejected below.
    base_url = base_url.strip(" \t\r\n`\"'")

    if len(api_key) < _MIN_KEY_LENGTH or any(ch.isspace() for ch in api_key):
        raise LiveConfigError("invalid_api_key", "api key must be at least 8 chars without whitespace", [api_key_name])
    bad_url_chars = ('`', '"', "'")
    if (
        not base_url.startswith(("http://", "https://"))
        or any(ch.isspace() for ch in base_url)
        or any(ch in base_url for ch in bad_url_chars)
    ):
        raise LiveConfigError(
            "invalid_base_url",
            "base url must start with http(s):// without whitespace or markdown quoting",
            [base_url_name],
        )
    if not model or any(ch.isspace() for ch in model):
        raise LiveConfigError("invalid_model", "model name must be non-empty without whitespace", [model_name])
    try:
        price_input = float(price_input_raw)
        price_output = float(price_output_raw)
    except ValueError:
        raise LiveConfigError(
            "invalid_price",
            "prices must be numbers (USD per 1M tokens)",
            [price_input_name, price_output_name],
        )
    if price_input <= 0 or price_output <= 0:
        raise LiveConfigError(
            "invalid_price",
            "prices must be greater than zero",
            [price_input_name, price_output_name],
        )

    timeout_raw = source.get(timeout_name, "")
    timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    if timeout_raw != "":
        try:
            timeout_seconds = int(timeout_raw)
        except ValueError:
            raise LiveConfigError(
                "invalid_timeout",
                "timeout must be an integer number of seconds",
                [timeout_name],
            )
        if timeout_seconds <= 0:
            raise LiveConfigError(
                "invalid_timeout",
                "timeout must be a positive number of seconds",
                [timeout_name],
            )

    return LiveConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        price_input_per_mtok=price_input,
        price_output_per_mtok=price_output,
        timeout_seconds=timeout_seconds,
    )


def load_live_config(env: Optional[Mapping[str, str]] = None) -> LiveConfig:
    source = os.environ if env is None else env
    return _load_config(
        source,
        base_url_name=ENV_BASE_URL,
        model_name=ENV_MODEL,
        api_key_name=ENV_API_KEY,
        price_input_name=ENV_PRICE_INPUT,
        price_output_name=ENV_PRICE_OUTPUT,
        timeout_name=ENV_TIMEOUT,
    )


def load_optional_cheap_config(env: Optional[Mapping[str, str]] = None) -> Optional[LiveConfig]:
    source = os.environ if env is None else env
    present = [name for name in CHEAP_ENV_VARS if source.get(name)]
    if not present:
        return None
    missing = [name for name in CHEAP_ENV_VARS if not source.get(name)]
    if missing:
        raise LiveConfigError(
            "incomplete_cheap_config",
            "cheap route configuration must be supplied all-or-none",
            missing,
        )
    return _load_config(
        source,
        base_url_name=CHEAP_ENV_BASE_URL,
        model_name=CHEAP_ENV_MODEL,
        api_key_name=CHEAP_ENV_API_KEY,
        price_input_name=CHEAP_ENV_PRICE_INPUT,
        price_output_name=CHEAP_ENV_PRICE_OUTPUT,
        timeout_name=CHEAP_ENV_TIMEOUT,
    )


def config_check_payload() -> Dict:
    """Machine-readable, masked summary of the live configuration."""
    config = load_live_config()
    cheap_config = load_optional_cheap_config()
    return {
        "passed": True,
        "config": config.to_dict(),
        "cheap_config": cheap_config.to_dict() if cheap_config is not None else None,
    }
