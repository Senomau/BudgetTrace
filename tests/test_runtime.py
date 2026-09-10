from typing import Iterable, Mapping, Optional

from budgettrace.contracts import AgentCapabilities, AgentObservation, BudgetLimits, ControlOutcome, RunConfig, TaskContract
from budgettrace.events import EventJournal
from budgettrace.fixtures import FixtureScorer, ScriptedLLM
from budgettrace.runtime import Runtime
from budgettrace.security import ToolRegistry


class CapturingAdapter:
    def __init__(self, actions: Iterable[Mapping[str, object]]) -> None:
        self._actions = list(actions)
        self._position = 0
        self.observations = []

    def respond(
        self,
        task_id: str,
        observation: Optional[AgentObservation] = None,
    ) -> Mapping[str, object]:
        self.observations.append(observation)
        if self._position >= len(self._actions):
            return {"kind": "script_exhausted"}
        action = self._actions[self._position]
        self._position += 1
        return action

    def estimate_next_call(self, task_id: str):
        from budgettrace.contracts import BudgetDelta

        return BudgetDelta(steps=1, tokens=10, time_ms=1, cost_micros=1)


def _contract(tmp_path):
    return TaskContract(
        task_id="tool-chain",
        workspace_root=str(tmp_path),
        allowed_tools=["read_file"],
        tool_levels={"read_file": "L0"},
        budgets=BudgetLimits(steps=8, tokens=4000, time_ms=30_000, cost_micros=100_000),
        success_condition="score_at_least_one",
        scorer_version="fixture-v1",
    )


def _config():
    return RunConfig(
        run_id="run-tool-chain",
        task_id="tool-chain",
        policy_version="rules-v1",
        budget_window_steps=4,
        offline=True,
    )


def _tools():
    tools = ToolRegistry()
    tools.register("read_file", "L0", lambda args: {"ok": True})
    return tools


def test_runtime_records_tool_call_and_returns_result_as_next_observation(tmp_path):
    llm = CapturingAdapter(
        [
            {
                "kind": "tool_call",
                "tool": "read_file",
                "args": {"path": "input.py", "api_key": "secret-value"},
            },
            {"kind": "finish", "score": 1.0},
        ]
    )

    card = Runtime().run(_contract(tmp_path), _config(), llm, _tools(), FixtureScorer())

    events = EventJournal.from_jsonl(card.journal_jsonl).events()
    call = next(event for event in events if event.event_type == "tool_call")
    result = next(event for event in events if event.event_type == "tool_result")
    assert call.schema_version == 2
    assert result.schema_version == 2
    assert len(call.payload["args_hash"]) == 64
    assert call.payload["operation_id"] == result.payload["operation_id"]
    assert call.payload["args"]["api_key"] == "[REDACTED]"
    assert llm.observations[0] is None
    assert llm.observations[1].payload["output"] == {"ok": True}
    assert llm.observations[1].operation_id == call.payload["operation_id"]


def test_tool_handler_error_is_returned_to_model_without_crashing_episode(tmp_path):
    llm = CapturingAdapter(
        [
            {"kind": "tool_call", "tool": "read_file", "args": {"path": "input.py"}},
            {"kind": "finish", "score": 0.0},
        ]
    )
    tools = ToolRegistry()

    def failing_handler(args):
        raise ValueError("invalid task state")

    tools.register("read_file", "L0", failing_handler)

    card = Runtime().run(_contract(tmp_path), _config(), llm, tools, FixtureScorer())

    events = EventJournal.from_jsonl(card.journal_jsonl).events()
    error = next(event for event in events if event.event_type == "error")
    assert error.payload["error_code"] == "tool_execution_error"
    assert error.payload["ok"] is False
    assert error.payload["message"] == "invalid task state"
    assert llm.observations[1].kind == "tool_error"
    assert llm.observations[1].payload == {
        "error_code": "tool_execution_error",
        "message": "invalid task state",
        "ok": False,
    }
    assert card.terminal_reason == "script_exhausted"


def test_scripted_runtime_produces_successful_replayable_runcard(tmp_path):
    contract = TaskContract(
        task_id="code-001",
        workspace_root=str(tmp_path),
        allowed_tools=["read_file", "run_tests"],
        tool_levels={"read_file": "L0", "run_tests": "L1"},
        budgets=BudgetLimits(steps=8, tokens=4000, time_ms=30_000, cost_micros=100_000),
        success_condition="tests_pass",
        scorer_version="fixture-v1",
    )
    config = RunConfig(
        run_id="run-code-001",
        task_id="code-001",
        policy_version="rules-v1",
        budget_window_steps=2,
        offline=True,
    )
    llm = ScriptedLLM(
        {
            "code-001": [
                {"kind": "tool_call", "tool": "read_file", "args": {"path": "input.py"}},
                {"kind": "feedback", "feedback_id": "tests-failed", "score": 0.2},
                {"kind": "tool_call", "tool": "run_tests", "args": {}},
                {"kind": "feedback", "feedback_id": "tests-pass", "score": 1.0},
            ]
        }
    )
    tools = ToolRegistry()
    tools.register("read_file", "L0", lambda args: {"content": "print('ok')"})
    tools.register("run_tests", "L1", lambda args: {"passed": True})

    card = Runtime().run(contract, config, llm, tools, FixtureScorer())

    assert card.success is True
    assert card.terminal_reason == "success"
    assert card.budget.steps_used == 4
    assert card.event_count >= 7
    assert EventJournal.from_jsonl(card.journal_jsonl).events()


def test_runtime_scores_terminal_feedback_when_script_ends_mid_window(tmp_path):
    contract = TaskContract(
        task_id="tiny-001",
        workspace_root=str(tmp_path),
        allowed_tools=["read_file"],
        tool_levels={"read_file": "L0"},
        budgets=BudgetLimits(steps=4, tokens=1000, time_ms=1000, cost_micros=1000),
        success_condition="score_at_least_one",
        scorer_version="fixture-v1",
    )
    config = RunConfig(
        run_id="run-tiny-001",
        task_id="tiny-001",
        policy_version="rules-v1",
        budget_window_steps=2,
    )
    llm = ScriptedLLM({"tiny-001": [{"kind": "feedback", "feedback_id": "done", "score": 1.0}]})
    tools = ToolRegistry()
    tools.register("read_file", "L0", lambda args: {"ok": True})

    card = Runtime().run(contract, config, llm, tools, FixtureScorer())

    assert card.success is True
    assert card.terminal_reason == "success"


def test_runtime_stops_immediately_after_successful_feedback(tmp_path):
    contract = TaskContract(
        task_id="early-success",
        workspace_root=str(tmp_path),
        allowed_tools=["read_file"],
        tool_levels={"read_file": "L0"},
        budgets=BudgetLimits(steps=6, tokens=1000, time_ms=1000, cost_micros=1000),
        success_condition="score_at_least_one",
        scorer_version="fixture-v1",
    )
    config = RunConfig(
        run_id="run-early-success",
        task_id="early-success",
        policy_version="rules-v1",
        budget_window_steps=4,
    )
    llm = ScriptedLLM(
        {
            "early-success": [
                {"kind": "feedback", "feedback_id": "done", "score": 1.0},
                {"kind": "feedback", "feedback_id": "should-not-run", "score": 1.0},
            ]
        }
    )
    tools = ToolRegistry()
    tools.register("read_file", "L0", lambda args: {"ok": True})

    card = Runtime().run(contract, config, llm, tools, FixtureScorer())

    assert card.success is True
    assert card.budget.steps_used == 1


def test_unaffordable_reservation_prevents_transport_and_records_rejection(tmp_path):
    class EstimatingAdapter(CapturingAdapter):
        def estimate_next_call(self, task_id: str):
            from budgettrace.contracts import BudgetDelta

            return BudgetDelta(steps=1, tokens=999, time_ms=1, cost_micros=1)

    contract = _contract(tmp_path).model_copy(
        update={"budgets": BudgetLimits(steps=2, tokens=20, time_ms=1000, cost_micros=1000)}
    )
    llm = EstimatingAdapter([{"kind": "finish", "score": 1.0, "tokens": 1}])

    card = Runtime().run(contract, _config(), llm, _tools(), FixtureScorer())
    events = EventJournal.from_jsonl(card.journal_jsonl).events()

    assert llm.observations == []
    assert card.terminal_reason == "budget_rejected"
    assert card.budget.steps_used == 0
    assert card.over_budget is False
    assert card.over_budget_dimensions == ["tokens"]
    assert [event.event_type for event in events][-2:] == ["budget_rejected", "terminal"]
    assert events[-2].payload["reserved_budget"]["tokens"] == 999
    assert events[-2].payload["dimensions"] == ["tokens"]


def test_runtime_reconciles_observed_overrun_after_transport(tmp_path):
    class UnderestimatingAdapter(CapturingAdapter):
        def estimate_next_call(self, task_id: str):
            from budgettrace.contracts import BudgetDelta

            return BudgetDelta(steps=1, tokens=1, time_ms=1, cost_micros=1)

    contract = _contract(tmp_path).model_copy(
        update={"budgets": BudgetLimits(steps=2, tokens=20, time_ms=1000, cost_micros=1000)}
    )
    llm = UnderestimatingAdapter([{"kind": "feedback", "feedback_id": "costly", "score": 0.2, "tokens": 25}])

    card = Runtime().run(contract, _config(), llm, _tools(), FixtureScorer())

    assert len(llm.observations) == 1
    assert card.terminal_reason == "budget_exhausted"
    assert card.over_budget is True
    assert card.over_budget_dimensions == ["tokens"]
    assert card.reserved_budget.tokens == 1
    assert card.observed_budget.tokens == 25
    assert card.budget.tokens_used == 25


def test_failed_provider_call_has_unknown_billing_status(tmp_path):
    from budgettrace.contracts import AgentCallError

    class FailingAdapter(CapturingAdapter):
        def respond(self, task_id: str, observation: Optional[AgentObservation] = None):
            raise AgentCallError(
                code="transport_failed",
                task_id=task_id,
                billing_status="unknown",
                message="timeout",
            )

    llm = FailingAdapter([])
    card = Runtime().run(_contract(tmp_path), _config(), llm, _tools(), FixtureScorer())
    events = EventJournal.from_jsonl(card.journal_jsonl).events()

    assert card.terminal_reason == "agent_call_error"
    assert card.billing_status == "unknown"
    assert card.budget.steps_used == 0
    assert events[-2].event_type == "error"
    assert events[-2].payload["error"]["billing_status"] == "unknown"
    assert events[-2].payload["error"]["code"] == "transport_failed"


def test_unsupported_replan_control_records_failure_and_terminates(tmp_path):
    class UnsupportedControlAdapter(CapturingAdapter):
        capabilities = AgentCapabilities(supports_replan=True, supports_route_cheaper=False)

        def apply_control(self, decision):
            return ControlOutcome(
                applied=False,
                action=decision.action,
                reason_code=decision.reason_code,
                metadata={"unsupported": True},
            )

    llm = UnsupportedControlAdapter(
        [
            {"kind": "tool_call", "tool": "read_file", "args": {"path": "input.py"}},
            {"kind": "tool_call", "tool": "read_file", "args": {"path": "input.py"}},
            {"kind": "finish", "score": 1.0},
        ]
    )
    config = RunConfig(
        run_id="run-control-failed",
        task_id="tool-chain",
        policy_version="rules-v1",
        budget_window_steps=2,
    )

    card = Runtime().run(_contract(tmp_path), config, llm, _tools(), FixtureScorer())
    events = EventJournal.from_jsonl(card.journal_jsonl).events()

    assert card.terminal_reason == "control_unavailable"
    assert [event.event_type for event in events][-3:] == ["decision", "control_failed", "terminal"]
    assert events[-3].payload["action"] == "replan"
    assert events[-2].payload["action"] == "replan"
    assert events[-2].payload["reason_code"] == "repeated_no_progress"
