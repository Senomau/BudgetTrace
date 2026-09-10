import pytest
from pydantic import ValidationError

from budgettrace.contracts import AgentObservation, Decision
from budgettrace.fixtures import ScriptedLLM


def test_agent_observation_is_frozen_and_versioned():
    observation = AgentObservation(kind="tool_result", operation_id="op-1", payload={"ok": True})

    assert observation.schema_version == 2
    with pytest.raises(ValidationError):
        observation.operation_id = "op-2"


def test_agent_observation_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        AgentObservation(
            kind="tool_result",
            operation_id="op-1",
            payload={"ok": True},
            unexpected=True,
        )


def test_scripted_adapter_accepts_an_observation_and_estimates_next_call():
    adapter = ScriptedLLM({"task": [{"kind": "finish", "score": 1.0}]})

    result = adapter.respond(
        "task",
        AgentObservation(kind="tool_result", operation_id="op-1", payload={"ok": True}),
    )

    assert result["kind"] == "finish"
    assert adapter.observations[0].operation_id == "op-1"
    assert adapter.estimate_next_call("task").steps == 1


def test_scripted_adapter_reports_control_outcomes():
    adapter = ScriptedLLM({"task": [{"kind": "finish", "score": 1.0}]})

    outcome = adapter.apply_control(
        Decision(action="replan", reason_code="stall", budget_window=0, policy_version="v1")
    )

    assert outcome.applied is True
    assert outcome.action == "replan"
    assert outcome.reason_code == "stall"
