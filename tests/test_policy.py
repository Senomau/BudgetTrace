from budgettrace.contracts import FeatureSnapshot
from budgettrace.policy import BudgetController


def _snapshot(**changes):
    values = {
        "window_index": 1,
        "success": False,
        "safety_blocked": False,
        "budget_exhausted": False,
        "new_feedback": False,
        "high_repeat_no_novelty": False,
        "cheap_route_safe": False,
        "can_replan": False,
    }
    values.update(changes)
    return FeatureSnapshot(**values)


def test_success_and_safety_have_terminal_stop_decisions():
    controller = BudgetController()

    assert controller.choose(_snapshot(success=True)).reason_code == "success"
    assert controller.choose(_snapshot(safety_blocked=True)).reason_code == "safety_denied"
    assert controller.choose(_snapshot(success=True)).action == "stop"


def test_policy_preserves_late_breakthrough_window():
    decision = BudgetController().choose(
        _snapshot(late_breakthrough_guard=True, new_feedback=True)
    )

    assert decision.action == "continue"
    assert decision.reason_code == "late_breakthrough_window"


def test_policy_replans_repeated_no_progress_before_stopping():
    controller = BudgetController()
    decision = controller.choose(_snapshot(high_repeat_no_novelty=True, can_replan=True))

    assert decision.action == "replan"
    assert decision.reason_code == "repeated_no_progress"


def test_policy_replans_to_preserve_late_breakthrough_option_without_new_feedback():
    decision = BudgetController().choose(
        _snapshot(late_breakthrough_guard=True, new_feedback=False, can_replan=True)
    )

    assert decision.action == "replan"
    assert decision.reason_code == "preserve_breakthrough_option"


def test_policy_can_route_safe_low_complexity_window():
    decision = BudgetController().choose(_snapshot(cheap_route_safe=True))

    assert decision.action == "route_cheaper"
    assert decision.reason_code == "low_complexity_window"


def test_policy_priority_is_safety_then_success_then_budget():
    controller = BudgetController()

    assert controller.choose(_snapshot(safety_blocked=True, success=True, budget_exhausted=True)).reason_code == "safety_denied"
    assert controller.choose(_snapshot(success=True, budget_exhausted=True)).reason_code == "success"
    assert controller.choose(_snapshot(budget_exhausted=True)).reason_code == "budget_exhausted"
