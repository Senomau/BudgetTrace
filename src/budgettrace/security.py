"""Fail-closed tool registry used by the offline runtime."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from .contracts import SideEffectLevel, TaskContract


class ToolDeniedError(PermissionError):
    """Raised when a tool call crosses an explicit safety boundary."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    level: SideEffectLevel
    handler: Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    side_effect_level: SideEffectLevel
    output: Any
    redacted_args: Dict[str, Any]


_SECRET_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|authorization)($|[_-])",
    re.I,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/\-=]{16,}\b"),
    re.compile(r"https?://[^/\s:@]+:[^@\s/]+@"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)


def redact_secrets(value: Any, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): redact_secrets(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str) and any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
        return "[REDACTED]"
    return value


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        level: SideEffectLevel,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> None:
        if name in self._specs:
            raise ValueError(f"tool already registered: {name}")
        self._specs[name] = ToolSpec(name=name, level=level, handler=handler)

    def invoke(
        self,
        name: str,
        args: Mapping[str, Any],
        contract: TaskContract,
        *,
        approved: bool = False,
    ) -> ToolResult:
        if name not in contract.allowed_tools:
            raise ToolDeniedError(f"tool not allowlisted: {name}")
        spec = self._specs.get(name)
        if spec is None:
            raise ToolDeniedError(f"tool not registered: {name}")
        effective_level = contract.tool_levels.get(name, spec.level)
        if effective_level == "L3" and not approved:
            raise ToolDeniedError(f"tool requires approval: {name}")
        self._validate_paths(args, contract.workspace_root)
        output = spec.handler(dict(args))
        return ToolResult(
            tool_name=name,
            side_effect_level=effective_level,
            output=output,
            redacted_args=redact_secrets(dict(args)),
        )

    def side_effect_level(self, name: str, contract: TaskContract) -> str:
        if name in contract.tool_levels:
            return contract.tool_levels[name]
        spec = self._specs.get(name)
        if spec is not None:
            return spec.level
        return "unknown"

    @staticmethod
    def _validate_paths(args: Mapping[str, Any], workspace_root: str) -> None:
        root = os.path.realpath(workspace_root)
        for key, value in args.items():
            if key.lower() not in {"path", "file", "directory", "paths"}:
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            for raw_path in values:
                if not isinstance(raw_path, str):
                    continue
                candidate = os.path.realpath(os.path.join(root, raw_path))
                try:
                    inside = os.path.commonpath([root, candidate]) == root
                except ValueError:
                    inside = False
                if not inside:
                    raise ToolDeniedError(f"path outside workspace: {raw_path}")
