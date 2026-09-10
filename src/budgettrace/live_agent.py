"""Runtime-compatible LLM adapter backed by a live chat transport.

LiveLLM wraps the audited live transport (the same OpenAI-compatible call
used by the smoke and eval paths) and emits action dicts in the exact shape
ScriptedLLM produces, so the standard Runtime agent loop and BudgetLedger
can drive a real model. Every action carries the live usage (tokens,
time_ms, cost_micros), which makes the offline-style budget gate apply to
real costs.

Budget enforcement deliberately stays in Runtime's ledger: LiveLLM reports
usage, it never gates calls itself.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from .contracts import (
    AgentCallError,
    AgentCapabilities,
    AgentObservation,
    BudgetDelta,
    BudgetLimits,
    ControlOutcome,
    Decision,
)
from .live_config import LiveConfig
from .live_smoke import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    SmokeError,
    Transport,
    _http_transport,
    estimate_cost_micros,
    prompt_from_task,
)
from .public_replay import Tau3ReferenceTask

PROTOCOL_VERSION = "live-agent-v1"

SYSTEM_PROMPT = (
    "You are a budget-aware customer-service agent. "
    "Reply with exactly one JSON object and nothing else. "
    'Actions: {"kind": "tool_call", "tool": "<name>", "args": {...}}, '
    '{"kind": "finish", "score": <number>}.'
)

NATIVE_TOOL_SYSTEM_PROMPT = (
    "You are a budget-aware customer-service agent. "
    "Use the provided native function tools for every lookup or state-changing "
    "operation; do not write tool-call JSON in text and do not invent tool names. "
    "An identity lookup is only a preparation step. For tasks requiring order "
    "details, an exchange, a return, a refund, or another state change, do not "
    "finish after the identity lookup; continue with every required tool operation "
    "and finish only after those operations succeed. "
    "After the tool results establish that the task is complete, reply with "
    'exactly one JSON object: {"kind":"finish","score":<number>}. '
    "Do not claim completion before the required tool operations have succeeded."
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_ACTION_KINDS = {"tool_call", "finish", "feedback", "script_exhausted"}


def parse_action(content: str) -> Optional[Dict[str, Any]]:
    """Parse exactly one action object, tolerating small amounts of prose around it.

    Multiple action objects and JSON arrays are rejected so a verbose model
    response cannot silently turn into an arbitrary first action.
    """

    def is_action(value: Any) -> bool:
        return isinstance(value, dict) and value.get("kind") in _ACTION_KINDS

    stripped = content.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if is_action(parsed):
        return parsed
    if isinstance(parsed, list):
        return None

    fenced = _JSON_FENCE_RE.findall(content)
    fenced_actions: List[Dict[str, Any]] = []
    for block in fenced:
        try:
            candidate = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        if is_action(candidate):
            fenced_actions.append(candidate)
    if len(fenced_actions) == 1 and len(fenced) == 1:
        return fenced_actions[0]
    if len(fenced_actions) > 1:
        return None

    decoder = json.JSONDecoder()
    embedded: List[Dict[str, Any]] = []
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if is_action(candidate):
            embedded.append(candidate)
    return embedded[0] if len(embedded) == 1 else None


class LiveLLM:
    """ScriptedLLM-shaped adapter over a live chat transport."""

    def __init__(
        self,
        config: LiveConfig,
        task: Tau3ReferenceTask,
        *,
        limits: BudgetLimits,
        transport: Optional[Transport] = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        cheap_config: Optional[LiveConfig] = None,
        tool_instructions: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self._cheap_config = cheap_config
        self.capabilities = AgentCapabilities(
            supports_replan=True,
            supports_route_cheaper=cheap_config is not None,
        )
        self.task = task
        self.limits = limits
        self._transport = transport if transport is not None else _http_transport
        self._max_output_tokens = max_output_tokens
        self._tool_schemas = [dict(schema) for schema in (tool_schemas or [])]
        self._pending_tool_call_id: Optional[str] = None
        self._native_tool_turns = 0
        self.calls: List[Dict[str, Any]] = []
        self.total_tokens = 0
        self.total_cost_micros = 0
        prompt, _ = prompt_from_task(task)
        system_prompt = NATIVE_TOOL_SYSTEM_PROMPT if self._tool_schemas else SYSTEM_PROMPT
        if tool_instructions:
            system_prompt = SYSTEM_PROMPT + "\n\n" + tool_instructions
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    def respond(
        self,
        task_id: str,
        observation: Optional[AgentObservation] = None,
    ) -> Dict[str, Any]:
        if observation is not None:
            observation_content = json.dumps(
                observation.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            if self._tool_schemas and self._pending_tool_call_id:
                tool_result_content = json.dumps(
                    observation.payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": self._pending_tool_call_id,
                        "content": tool_result_content,
                    }
                )
                self._pending_tool_call_id = None
            else:
                self.messages.append(
                    {"role": "user", "content": observation_content}
                )
        payload = {
            "model": self.config.model,
            "messages": [dict(message) for message in self.messages],
            "max_tokens": self._max_output_tokens,
        }
        if self._tool_schemas:
            payload["tools"] = [dict(schema) for schema in self._tool_schemas]
            # Require one first action when native tools are enabled; later
            # turns can finish after receiving the tool result.
            is_deepseek = self.config.model.lower().startswith("deepseek")
            is_glm = self.config.model.lower().startswith("glm-")
            if is_deepseek or is_glm:
                # The runtime needs a deterministic tool-call message and does
                # not carry provider-specific reasoning fields.
                payload["thinking"] = {"type": "disabled"}
            payload["tool_choice"] = (
                "required" if is_deepseek and self._native_tool_turns == 0 else "auto"
            )
                # Runtime executes one action per step and returns one matching
                # tool result, so parallel tool calls are disabled.
            payload["parallel_tool_calls"] = False
            self._native_tool_turns += 1
        call_start = time.perf_counter()
        try:
            response = self._transport(self.config, payload)
        except SmokeError as exc:
            raise AgentCallError(
                code=exc.code,
                task_id=str(exc.task_id or task_id),
                billing_status="unknown",
                message=self._safe_error_message(str(exc)),
            ) from exc
        except OSError as exc:  # e.g. socket read timeout on long reasoning
            raise AgentCallError(
                code="http_error",
                task_id=str(task_id),
                billing_status="unknown",
                message=self._safe_error_message(f"transport failed: {exc}"),
            ) from exc
        time_ms = int((time.perf_counter() - call_start) * 1000)

        usage = response.get("usage")
        if not isinstance(usage, dict) or "prompt_tokens" not in usage or "completion_tokens" not in usage:
            raise AgentCallError(
                code="response_missing_usage",
                task_id=str(task_id),
                billing_status="unknown",
                message="provider response has no token usage",
            )
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        tokens = prompt_tokens + completion_tokens
        cost_micros = estimate_cost_micros(
            prompt_tokens,
            completion_tokens,
            self.config.price_input_per_mtok,
            self.config.price_output_per_mtok,
        )

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        raw_content = message.get("content", "") if isinstance(message, dict) else ""
        content = raw_content if isinstance(raw_content, str) else ""
        action = self._parse_native_tool_call(message) if self._tool_schemas else None
        if action is None:
            action = parse_action(content)
        if action is None:
            action = {"kind": "script_exhausted"}

        # Keep the conversation going: in native mode retain the complete
        # assistant tool-call message so the provider can correlate its next
        # tool result. Runtime supplies the real observation on the next call.
        if self._tool_schemas and isinstance(message, dict) and message.get("tool_calls"):
            self.messages.append(dict(message))
            if action.get("kind") == "tool_call":
                self._pending_tool_call_id = str(action["tool_call_id"])
        else:
            self.messages.append({"role": "assistant", "content": content})

        # Provider usage always overwrites anything the model itself claimed.
        # This is an estimate from provider-reported token usage, not account-level billing.
        action["tokens"] = tokens
        action["time_ms"] = time_ms
        action["cost_micros"] = cost_micros
        action["billing_status"] = "estimated_from_provider_usage"

        self.calls.append(
            {
                "task_id": str(task_id),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens": tokens,
                "cost_micros": cost_micros,
                "time_ms": time_ms,
                "finish_reason": choice.get("finish_reason"),
                "action_kind": action.get("kind", "unknown"),
                "billing_status": action["billing_status"],
                "tool_protocol": "openai-compatible-native" if self._tool_schemas else "text-json",
            }
        )
        self.total_tokens += tokens
        self.total_cost_micros += cost_micros
        return action

    @staticmethod
    def _parse_native_tool_call(message: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(message, dict):
            return None
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            return None
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            return None
        call_id = tool_call.get("id")
        function = tool_call.get("function")
        if not isinstance(call_id, str) or not call_id.strip() or not isinstance(function, dict):
            return None
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, str):
            return None
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed_arguments, dict):
            return None
        return {
            "kind": "tool_call",
            "tool": name,
            "args": parsed_arguments,
            "tool_call_id": call_id,
        }

    def estimate_next_call(self, task_id: str) -> BudgetDelta:
        prompt_tokens = self._estimate_prompt_tokens()
        tokens = prompt_tokens + self._max_output_tokens
        max_price = max(self.config.price_input_per_mtok, self.config.price_output_per_mtok)
        return BudgetDelta(
            steps=1,
            tokens=tokens,
            time_ms=self.config.timeout_seconds * 1000,
            cost_micros=int(round(tokens * max_price)) + 1,
        )

    def apply_control(self, decision: Decision) -> ControlOutcome:
        if decision.action == "replan":
            self.messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "schema_version": 2,
                            "kind": "control",
                            "action": "replan",
                            "reason_code": decision.reason_code,
                            "budget_window": decision.budget_window,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
            return ControlOutcome(
                applied=True,
                action=decision.action,
                reason_code=decision.reason_code,
                metadata={"message_appended": True},
            )
        if decision.action == "route_cheaper" and self._cheap_config is not None:
            previous_model = self.config.model
            self.config = self._cheap_config
            return ControlOutcome(
                applied=True,
                action=decision.action,
                reason_code=decision.reason_code,
                metadata={"previous_model": previous_model, "model": self.config.model},
            )
        return ControlOutcome(
            applied=False,
            action=decision.action,
            reason_code=decision.reason_code,
            metadata={"unsupported": True},
        )

    def _safe_error_message(self, message: str) -> str:
        api_key = getattr(self.config, "api_key", "")
        if api_key:
            message = message.replace(api_key, "[redacted]")
        return message

    def _estimate_prompt_tokens(self) -> int:
        # Conservative tokenizer-free estimate suitable for admission control.
        characters = sum(len(message.get("content", "")) for message in self.messages)
        return max(1, (characters + 3) // 4)
