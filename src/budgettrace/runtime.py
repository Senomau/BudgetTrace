"""Offline event-first runtime for deterministic BudgetTrace experiments."""

from typing import Any, Mapping

from .contracts import (
    AgentCallError,
    AgentCapabilities,
    AgentObservation,
    BillingStatus,
    BudgetDelta,
    Decision,
    Event,
    RunCard,
    RunConfig,
    TaskContract,
)
from .events import BudgetExceededError, BudgetLedger, EventJournal, canonical_args_hash
from .features import extract_features
from .policy import BudgetController
from .security import ToolDeniedError, ToolRegistry, redact_secrets


class Runtime:
    def __init__(self, controller: BudgetController = None) -> None:
        self.controller = controller or BudgetController()

    def run(
        self,
        contract: TaskContract,
        config: RunConfig,
        llm: Any,
        tools: ToolRegistry,
        scorer: Any,
    ) -> RunCard:
        if config.task_id != contract.task_id:
            raise ValueError("RunConfig.task_id must match TaskContract.task_id")
        journal = EventJournal()
        ledger = BudgetLedger(contract.budgets)
        score_history = []
        seen_feedback_ids = set()
        safety_blocked = False
        terminal_reason = "script_exhausted"
        success = False
        final_score = 0.0
        observation = None
        reserved_budget = BudgetDelta()
        observed_budget = BudgetDelta()
        over_budget = False
        over_budget_dimensions = []
        billing_status: BillingStatus = "measured"

        self._append(
            journal,
            ledger,
            config.run_id,
            "task_started",
            {
                "task_id": contract.task_id,
                "policy_version": config.policy_version,
                "budgets": contract.budgets.model_dump(mode="json"),
                "provenance": str(
                    contract.metadata.get("provenance", "offline" if config.offline else "live")
                ),
            },
        )

        while True:
            reserved_budget = self._estimate_next_call(llm, contract.task_id)
            try:
                ledger.admit(reserved_budget)
            except BudgetExceededError as exc:
                over_budget_dimensions = list(exc.dimensions)
                self._append(
                    journal,
                    ledger,
                    config.run_id,
                    "budget_rejected",
                    {
                        "dimensions": over_budget_dimensions,
                        "reserved_budget": reserved_budget.model_dump(mode="json"),
                    },
                )
                terminal_reason = "budget_rejected"
                break
            try:
                action = dict(llm.respond(contract.task_id, observation))
            except AgentCallError as exc:
                billing_status = exc.billing_status
                self._append(
                    journal,
                    ledger,
                    config.run_id,
                    "error",
                    {
                        "error": self._agent_call_error_payload(exc),
                        "ok": False,
                    },
                )
                terminal_reason = "agent_call_error"
                break
            observation = None
            if action.get("billing_status") == "estimated_from_provider_usage":
                billing_status = "estimated_from_provider_usage"
            elif action.get("billing_status") == "unknown":
                billing_status = "unknown"
            kind = str(action.get("kind", "script_exhausted"))
            if kind == "script_exhausted":
                # A live adapter reports the measured usage of the final call it
                # already made, so that call must still be billed. A scripted
                # exhaustion carries no usage keys and stays unbilled.
                if "cost_micros" in action or "tokens" in action:
                    step_number = ledger.snapshot.steps_used + 1
                    observed_budget = self._observed_delta(action)
                    model_event = self._event(
                        journal,
                        config.run_id,
                        "model_call",
                        {
                            "kind": kind,
                            "step": step_number,
                            "model": str(action.get("model", "scripted-v1")),
                        },
                        observed_budget,
                    )
                    ledger.reconcile(model_event)
                    journal.append(model_event)
                    over_budget_dimensions = ledger.exceeded_dimensions(BudgetDelta())
                    if over_budget_dimensions:
                        over_budget = True
                        terminal_reason = "budget_exhausted"
                        break
                final_score, success = scorer.evaluate(journal.events())
                terminal_reason = "success" if success else "script_exhausted"
                break
            step_number = ledger.snapshot.steps_used + 1
            observed_budget = self._observed_delta(action)
            model_event = self._event(
                journal,
                config.run_id,
                "model_call",
                {
                    "kind": kind,
                    "step": step_number,
                    "model": str(action.get("model", "scripted-v1")),
                },
                observed_budget,
            )
            ledger.reconcile(model_event)
            journal.append(model_event)
            over_budget_dimensions = ledger.exceeded_dimensions(BudgetDelta())
            if over_budget_dimensions:
                over_budget = True
                terminal_reason = "budget_exhausted"
                break

            if kind == "tool_call":
                tool_name = str(action.get("tool", ""))
                raw_args = action.get("args", {})
                args = raw_args if isinstance(raw_args, Mapping) else {}
                redacted_args = redact_secrets(dict(args))
                args_hash = canonical_args_hash(args)
                operation_id = f"{config.run_id}-op{len(journal.events()) + 1}"
                side_effect_level = tools.side_effect_level(tool_name, contract)
                self._append(
                    journal,
                    ledger,
                    config.run_id,
                    "tool_call",
                    {
                        "operation_id": operation_id,
                        "tool": tool_name,
                        "args_hash": args_hash,
                        "args": redacted_args,
                        "side_effect_level": side_effect_level,
                    },
                )
                try:
                    result = tools.invoke(tool_name, args, contract, approved=bool(action.get("approved")))
                    redacted_output = redact_secrets(result.output)
                    self._append(
                        journal,
                        ledger,
                        config.run_id,
                        "tool_result",
                        {
                            "operation_id": operation_id,
                            "tool": result.tool_name,
                            "side_effect_level": result.side_effect_level,
                            "args_hash": args_hash,
                            "args": result.redacted_args,
                            "output": redacted_output,
                            "ok": True,
                        },
                    )
                    observation = AgentObservation(
                        kind="tool_result",
                        operation_id=operation_id,
                        payload={"tool": result.tool_name, "output": redacted_output, "ok": True},
                    )
                except ToolDeniedError as exc:
                    safety_blocked = True
                    message = str(redact_secrets({"message": str(exc)})["message"])
                    self._append(
                        journal,
                        ledger,
                        config.run_id,
                        "error",
                        {
                            "operation_id": operation_id,
                            "tool": tool_name,
                            "side_effect_level": side_effect_level,
                            "args_hash": args_hash,
                            "args": redacted_args,
                            "error_code": "safety_denied",
                            "message": message,
                            "ok": False,
                        },
                    )
                    observation = AgentObservation(
                        kind="tool_error",
                        operation_id=operation_id,
                        payload={"error_code": "safety_denied", "message": message, "ok": False},
                    )
                except Exception as exc:
                    message = str(redact_secrets({"message": str(exc)})["message"])
                    self._append(
                        journal,
                        ledger,
                        config.run_id,
                        "error",
                        {
                            "operation_id": operation_id,
                            "tool": tool_name,
                            "side_effect_level": side_effect_level,
                            "args_hash": args_hash,
                            "args": redacted_args,
                            "error_code": "tool_execution_error",
                            "message": message,
                            "ok": False,
                        },
                    )
                    observation = AgentObservation(
                        kind="tool_error",
                        operation_id=operation_id,
                        payload={
                            "error_code": "tool_execution_error",
                            "message": message,
                            "ok": False,
                        },
                    )
            elif kind == "feedback":
                feedback_payload = {
                    "feedback_id": str(action.get("feedback_id", "feedback-missing")),
                    "score": float(action.get("score", 0.0)),
                }
                score_history.append(feedback_payload["score"])
                self._append(journal, ledger, config.run_id, "feedback", feedback_payload)
            elif kind == "finish":
                finish_score = float(action.get("score", 0.0))
                score_history.append(finish_score)
                self._append(
                    journal,
                    ledger,
                    config.run_id,
                    "feedback",
                    {"feedback_id": "finish", "score": finish_score},
                )

            if kind in {"feedback", "finish"}:
                final_score, success = scorer.evaluate(journal.events())
                if success and self.controller.stop_on_success:
                    terminal_reason = "success"
                    break

            at_window_end = ledger.snapshot.steps_used % config.budget_window_steps == 0
            if at_window_end:
                final_score, success = scorer.evaluate(journal.events())
                capabilities = self._capabilities(llm)
                snapshot = extract_features(
                    journal.events(),
                    score_history,
                    contract.budgets,
                    late_breakthrough_guard=bool(contract.metadata.get("late_breakthrough_guard")),
                    success=success,
                    safety_blocked=safety_blocked,
                    can_replan=capabilities.supports_replan,
                    cheap_route_safe=bool(
                        capabilities.supports_route_cheaper
                        and contract.metadata.get("cheap_route_safe") is True
                    ),
                    window_index=ledger.snapshot.steps_used // config.budget_window_steps,
                    previous_feedback_ids=seen_feedback_ids,
                )
                seen_feedback_ids.update(
                    str(event.payload["feedback_id"])
                    for event in journal.events()
                    if event.event_type == "feedback" and event.payload.get("feedback_id")
                )
                decision = self.controller.choose(snapshot)
                self._append(
                    journal,
                    ledger,
                    config.run_id,
                    "decision",
                    decision.model_dump(mode="json"),
                )
                if decision.action == "stop":
                    terminal_reason = decision.reason_code
                    break
                if decision.action in {"replan", "route_cheaper"}:
                    outcome = self._apply_control(llm, decision)
                    self._append(
                        journal,
                        ledger,
                        config.run_id,
                        "control_applied" if outcome.applied else "control_failed",
                        outcome.model_dump(mode="json"),
                    )
                    if not outcome.applied:
                        terminal_reason = "control_unavailable"
                        break

        self._append(
            journal,
            ledger,
            config.run_id,
            "terminal",
            {
                "reason": terminal_reason,
                "success": success,
                "score": final_score,
                "scored": bool(getattr(scorer, "scored", True)),
            },
        )
        return RunCard(
            run_id=config.run_id,
            task_id=contract.task_id,
            policy_version=config.policy_version,
            terminal_reason=terminal_reason,
            success=success,
            final_score=final_score,
            event_count=len(journal.events()),
            budget=ledger.snapshot,
            over_budget=over_budget,
            over_budget_dimensions=over_budget_dimensions,
            billing_status=billing_status,
            reserved_budget=reserved_budget,
            observed_budget=observed_budget,
            provenance="offline" if config.offline else "live",
            journal_jsonl=journal.to_jsonl(),
        )

    @staticmethod
    def _event(
        journal: EventJournal,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        budget_delta: BudgetDelta = None,
    ) -> Event:
        sequence = len(journal.events()) + 1
        return Event(
            schema_version=2,
            event_id=f"{run_id}-e{sequence}",
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            occurred_at="2026-09-07T00:00:00Z",
            payload=dict(payload),
            budget_delta=budget_delta or BudgetDelta(),
        )

    def _append(
        self,
        journal: EventJournal,
        ledger: BudgetLedger,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        budget_delta: BudgetDelta = None,
    ) -> None:
        event = self._event(journal, run_id, event_type, payload, budget_delta)
        ledger.apply(event)
        journal.append(event)

    @staticmethod
    def _estimate_next_call(llm: Any, task_id: str) -> BudgetDelta:
        estimate = getattr(llm, "estimate_next_call", None)
        if estimate is None:
            return BudgetDelta(steps=1, tokens=10, time_ms=1, cost_micros=1)
        return BudgetDelta.model_validate(estimate(task_id))

    @staticmethod
    def _observed_delta(action: Mapping[str, Any]) -> BudgetDelta:
        return BudgetDelta(
            steps=1,
            tokens=int(action.get("tokens", 10)),
            time_ms=int(action.get("time_ms", 1)),
            cost_micros=int(action.get("cost_micros", 1)),
        )

    @staticmethod
    def _agent_call_error_payload(exc: AgentCallError) -> Mapping[str, Any]:
        return {
            "code": exc.code,
            "task_id": exc.task_id,
            "billing_status": exc.billing_status,
            "message": str(redact_secrets({"message": str(exc)})["message"]),
        }

    @staticmethod
    def _capabilities(llm: Any) -> AgentCapabilities:
        capabilities = getattr(llm, "capabilities", None)
        if capabilities is None:
            return AgentCapabilities(supports_replan=False, supports_route_cheaper=False)
        return AgentCapabilities.model_validate(capabilities)

    @staticmethod
    def _apply_control(llm: Any, decision: Decision):
        apply_control = getattr(llm, "apply_control", None)
        if apply_control is None:
            from .contracts import ControlOutcome

            return ControlOutcome(
                applied=False,
                action=decision.action,
                reason_code=decision.reason_code,
                metadata={"unsupported": True},
            )
        return apply_control(decision)
