"""TDD tests for the LiveLLM adapter: a runtime-compatible LLM backed by a live transport.

Contract: LiveLLM wraps the live chat transport and produces action dicts in the
same shape ScriptedLLM emits, so Runtime's agent loop and BudgetLedger can be
driven by a real model. Each action carries the live usage (tokens, time_ms,
cost_micros) so the offline-style budget gate applies to real costs.
"""
import json

import pytest

from budgettrace.contracts import AgentObservation, BudgetLimits, RunConfig, TaskContract
from budgettrace.fixtures import FixtureScorer
from budgettrace.runtime import Runtime
from budgettrace.security import ToolRegistry
from test_live_eval import _sequential_transport
from test_live_smoke import (
    FAKE_KEY,
    _config,
    _fake_response,
    _fake_task,
    _limits,
)

TOOL_ACTION = {"kind": "tool_call", "tool": "read_file", "args": {"path": "input.py"}}
FINISH_ACTION = {"kind": "finish", "score": 1.0}


def _live_response(content):
    response = _fake_response()
    response["choices"][0]["message"]["content"] = content
    return response


def _json_response(action):
    return _live_response(json.dumps(action))


def _native_tool_response(name="find_user_id_by_name_zip", arguments=None, call_id="call-1"):
    response = _fake_response()
    response["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments
                        or {"first_name": "Test", "last_name": "User", "zip": "00000"}
                    ),
                },
            }
        ],
    }
    response["choices"][0]["finish_reason"] = "tool_calls"
    return response


def _live_llm(responses):
    from budgettrace.live_agent import LiveLLM

    transport = _sequential_transport(responses)
    llm = LiveLLM(_config(), _fake_task(), limits=_limits(), transport=transport)
    return llm, transport


def _cheap_config():
    from budgettrace.live_config import LiveConfig

    return LiveConfig(
        base_url="https://cheap.example.com/v1",
        model="cheap-model",
        api_key="unit-test-cheap-key-1234567890",
        price_input_per_mtok=0.01,
        price_output_per_mtok=0.02,
        timeout_seconds=30,
    )


def _decision(action, reason_code):
    from budgettrace.contracts import Decision

    return Decision(
        action=action,
        reason_code=reason_code,
        budget_window=1,
        policy_version="rules-v1",
    )


def test_respond_returns_runtime_action_with_live_usage():
    llm, _ = _live_llm([_json_response(TOOL_ACTION)])

    action = llm.respond("70")

    assert action["kind"] == "tool_call"
    assert action["tool"] == "read_file"
    assert action["args"] == {"path": "input.py"}
    assert action["tokens"] == 150
    assert action["cost_micros"] == 330
    assert isinstance(action["time_ms"], int)
    assert action["billing_status"] == "estimated_from_provider_usage"


def test_non_json_content_falls_back_to_script_exhausted_with_usage():
    llm, _ = _live_llm([_live_response("plain text, no json here")])

    action = llm.respond("70")

    assert action["kind"] == "script_exhausted"
    assert action["tokens"] == 150
    assert action["cost_micros"] == 330
    assert action["billing_status"] == "estimated_from_provider_usage"


def test_json_fence_content_is_parsed():
    fenced = "```json\n" + json.dumps(FINISH_ACTION) + "\n```"
    llm, _ = _live_llm([_live_response(fenced)])

    action = llm.respond("70")

    assert action["kind"] == "finish"
    assert action["score"] == 1.0


def test_usage_accumulates_across_calls():
    llm, _ = _live_llm([_json_response(TOOL_ACTION), _json_response(TOOL_ACTION)])

    llm.respond("70")
    llm.respond("70")

    assert len(llm.calls) == 2
    assert llm.total_tokens == 300
    assert llm.total_cost_micros == 660


def test_transport_smoke_error_without_task_id_becomes_agent_call_error():
    from budgettrace.contracts import AgentCallError
    from budgettrace.live_agent import LiveLLM
    from budgettrace.live_smoke import SmokeError

    def transport(config, payload):
        raise SmokeError("http_error", "transport failed: The read operation timed out")

    llm = LiveLLM(_config(), _fake_task(), limits=_limits(), transport=transport)

    with pytest.raises(AgentCallError) as excinfo:
        llm.respond("70")

    error = excinfo.value
    assert error.code == "http_error"
    assert error.task_id == "70"
    assert error.billing_status == "unknown"
    assert "read operation timed out" in error.message
    assert FAKE_KEY not in error.message


def test_payload_sends_protocol_system_prompt_and_task_instructions():
    llm, transport = _live_llm([_json_response(TOOL_ACTION)])

    llm.respond("70")

    payload = transport.calls[0]["payload"]
    roles = [message["role"] for message in payload["messages"]]
    assert roles == ["system", "user"]
    assert "JSON" in payload["messages"][0]["content"]
    assert "Reply with the single word OK." in payload["messages"][1]["content"]
    assert FAKE_KEY not in json.dumps(payload)


def test_native_tools_send_schema_and_round_trip_tool_result_message():
    from budgettrace.live_agent import LiveLLM

    schema = {
        "type": "function",
        "function": {
            "name": "find_user_id_by_name_zip",
            "description": "Find a retail user by name and ZIP code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["first_name", "last_name", "zip"],
                "additionalProperties": False,
            },
        },
    }
    transport = _sequential_transport(
        [_native_tool_response(), _json_response(FINISH_ACTION)]
    )
    llm = LiveLLM(
        _config(),
        _fake_task(),
        limits=_limits(),
        transport=transport,
        tool_schemas=[schema],
    )

    action = llm.respond("70")
    assert action["kind"] == "tool_call"
    assert action["tool"] == "find_user_id_by_name_zip"
    assert action["args"]["zip"] == "00000"
    assert action["tool_call_id"] == "call-1"
    assert transport.calls[0]["payload"]["tool_choice"] == "required"
    assert transport.calls[0]["payload"]["thinking"] == {"type": "disabled"}

    llm.respond(
        "70",
        AgentObservation(
            kind="tool_result",
            operation_id="op-1",
            payload={"tool": "find_user_id_by_name_zip", "output": "user-1", "ok": True},
        ),
    )
    second = transport.calls[1]["payload"]
    assert second["tools"] == [schema]
    assert second["tool_choice"] == "auto"
    assert second["parallel_tool_calls"] is False
    assert [message["role"] for message in second["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second["messages"][2]["tool_calls"][0]["id"] == "call-1"
    assert second["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": json.dumps(
            {"tool": "find_user_id_by_name_zip", "ok": True, "output": "user-1"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    }


def test_native_tools_use_native_instruction_prompt():
    from budgettrace.live_agent import LiveLLM

    schema = {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Get order details.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    }
    transport = _sequential_transport([_native_tool_response(name="get_order_details")])
    llm = LiveLLM(
        _config(),
        _fake_task(),
        limits=_limits(),
        transport=transport,
        tool_schemas=[schema],
    )

    llm.respond("70")

    prompt = transport.calls[0]["payload"]["messages"][0]["content"]
    assert "native function tools" in prompt
    assert "Reply with exactly one JSON object and nothing else" not in prompt
    assert "An identity lookup is only a preparation step." in prompt
    assert "do not finish after the identity lookup" in prompt
    assert "order details, an exchange, a return, a refund, or another state change" in prompt


def test_native_tool_call_with_invalid_arguments_is_exhausted_safely():
    from budgettrace.live_agent import LiveLLM

    response = _native_tool_response()
    response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "not-json"
    llm = LiveLLM(
        _config(),
        _fake_task(),
        limits=_limits(),
        transport=_sequential_transport([response]),
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "find_user_id_by_name_zip",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    action = llm.respond("70")
    assert action["kind"] == "script_exhausted"


def test_runtime_integration_live_costs_reach_ledger(tmp_path):
    from budgettrace.live_agent import LiveLLM

    contract = TaskContract(
        task_id="70",
        workspace_root=str(tmp_path),
        allowed_tools=["read_file"],
        tool_levels={"read_file": "L0"},
        budgets=BudgetLimits(steps=8, tokens=4000, time_ms=300_000, cost_micros=10**9),
        success_condition="score_at_least_one",
        scorer_version="fixture-v1",
    )
    config = RunConfig(
        run_id="live-70",
        task_id="70",
        policy_version="live-v1",
        budget_window_steps=2,
        offline=False,
    )
    transport = _sequential_transport([_json_response(TOOL_ACTION), _json_response(FINISH_ACTION)])
    llm = LiveLLM(_config(), _fake_task(), limits=_limits(), transport=transport)
    tools = ToolRegistry()
    tools.register("read_file", "L0", lambda args: {"content": "print('ok')"})

    card = Runtime().run(contract, config, llm, tools, FixtureScorer())

    # stop_on_success halts the loop after the finish action: exactly 2 live calls
    assert card.provenance == "live"
    assert len(transport.calls) == 2
    assert card.budget.cost_micros_used == 660
    assert card.budget.tokens_used == 300
    assert card.terminal_reason == "success"
    assert card.billing_status == "estimated_from_provider_usage"
    assert llm.total_cost_micros == 660


def test_conversation_history_accumulates_across_responds():
    llm, transport = _live_llm([_json_response(TOOL_ACTION), _json_response(FINISH_ACTION)])

    llm.respond("70")
    observation = AgentObservation(
        kind="tool_result",
        operation_id="op-1",
        payload={"output": {"ok": True}},
    )
    llm.respond("70", observation)

    # second call must let the model see its previous turn and the real runtime observation
    second = transport.calls[1]["payload"]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert second["messages"][2]["content"] == json.dumps(TOOL_ACTION)
    tool_message = json.loads(second["messages"][3]["content"])
    assert tool_message["kind"] == "tool_result"
    assert tool_message["operation_id"] == "op-1"
    assert tool_message["payload"]["output"] == {"ok": True}
    assert "Tool call accepted" not in second["messages"][3]["content"]


def test_finish_action_appends_no_synthetic_followup():
    llm, transport = _live_llm([_json_response(FINISH_ACTION), _json_response(TOOL_ACTION)])

    llm.respond("70")
    llm.respond("70")

    # finish ends the episode: no loop-continuation message is synthesized
    second = transport.calls[1]["payload"]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_runtime_budget_reservation_rejects_unaffordable_live_call(tmp_path):
    from budgettrace.live_agent import LiveLLM

    contract = TaskContract(
        task_id="70",
        workspace_root=str(tmp_path),
        allowed_tools=["read_file"],
        tool_levels={"read_file": "L0"},
        budgets=BudgetLimits(steps=8, tokens=4000, time_ms=300_000, cost_micros=500),
        success_condition="score_at_least_one",
        scorer_version="fixture-v1",
    )
    config = RunConfig(
        run_id="live-70-capped",
        task_id="70",
        policy_version="live-v1",
        budget_window_steps=2,
        offline=False,
    )
    transport = _sequential_transport([_json_response(TOOL_ACTION), _json_response(TOOL_ACTION)])
    llm = LiveLLM(_config(), _fake_task(), limits=_limits(), transport=transport)
    tools = ToolRegistry()
    tools.register("read_file", "L0", lambda args: {"content": "print('ok')"})

    card = Runtime().run(contract, config, llm, tools, FixtureScorer())

    assert len(transport.calls) == 0
    assert card.terminal_reason == "budget_rejected"
    assert card.budget.cost_micros_used == 0
    assert card.over_budget_dimensions == ["cost_micros"]


def test_replan_changes_the_next_model_request():
    llm, transport = _live_llm([_json_response(FINISH_ACTION)])

    outcome = llm.apply_control(_decision("replan", "repeated_no_progress"))
    llm.respond("70")

    assert outcome.applied is True
    payload = transport.calls[-1]["payload"]
    control = json.loads(payload["messages"][-1]["content"])
    assert control["kind"] == "control"
    assert control["action"] == "replan"
    assert control["reason_code"] == "repeated_no_progress"


def test_route_cheaper_changes_model_and_price_source():
    from budgettrace.live_agent import LiveLLM

    cheap = _cheap_config()
    transport = _sequential_transport([_json_response(FINISH_ACTION)])
    llm = LiveLLM(_config(), _fake_task(), limits=_limits(), transport=transport, cheap_config=cheap)

    outcome = llm.apply_control(_decision("route_cheaper", "low_complexity_window"))
    estimate = llm.estimate_next_call("70")
    llm.respond("70")

    assert outcome.applied is True
    assert llm.config.model == cheap.model
    assert estimate.cost_micros < 1000
    assert transport.calls[-1]["payload"]["model"] == cheap.model
    assert llm.config.price_output_per_mtok == cheap.price_output_per_mtok
