"""Secret-free route manifest loading for live model matrices."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .live_config import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_PRICE_INPUT,
    ENV_PRICE_OUTPUT,
    ENV_TIMEOUT,
    LiveConfig,
    LiveConfigError,
    _load_config,
)


@dataclass(frozen=True)
class LiveRoute:
    route_id: str
    config: LiveConfig


def _route_config(spec: Mapping[str, Any], env: Mapping[str, str]) -> LiveConfig:
    api_key_env = str(spec.get("api_key_env", "")).strip()
    if not api_key_env or "api_key" in spec:
        raise LiveConfigError(
            "invalid_route_manifest",
            "route must declare api_key_env and must not contain an api_key value",
            ["api_key_env"],
        )
    source = {
        ENV_BASE_URL: str(spec.get("base_url", "")),
        ENV_MODEL: str(spec.get("model", "")),
        ENV_API_KEY: str(env.get(api_key_env, "")),
        ENV_PRICE_INPUT: str(spec.get("price_input_per_mtok", "")),
        ENV_PRICE_OUTPUT: str(spec.get("price_output_per_mtok", "")),
        ENV_TIMEOUT: str(spec.get("timeout_seconds", "")),
    }
    try:
        return _load_config(
            source,
            base_url_name=ENV_BASE_URL,
            model_name=ENV_MODEL,
            api_key_name=ENV_API_KEY,
            price_input_name=ENV_PRICE_INPUT,
            price_output_name=ENV_PRICE_OUTPUT,
            timeout_name=ENV_TIMEOUT,
        )
    except LiveConfigError as exc:
        variables = [api_key_env if item == ENV_API_KEY else item for item in exc.variables]
        raise LiveConfigError(exc.code, exc.message, variables) from exc


def load_route_manifest(
    path: Path,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> List[LiveRoute]:
    """Load route metadata while reading each API key from a named env var."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs = raw.get("routes") if isinstance(raw, Mapping) else raw
    if not isinstance(specs, list) or not specs:
        raise LiveConfigError("invalid_route_manifest", "routes must be a non-empty array", [str(path)])
    source = os.environ if env is None else env
    routes: List[LiveRoute] = []
    seen = set()
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise LiveConfigError("invalid_route_manifest", "each route must be an object", [str(path)])
        route_id = str(spec.get("route_id", "")).strip()
        if not route_id or route_id in seen:
            raise LiveConfigError("invalid_route_manifest", "route_id must be unique and non-empty", [str(path)])
        seen.add(route_id)
        routes.append(LiveRoute(route_id=route_id, config=_route_config(spec, source)))
    return routes
