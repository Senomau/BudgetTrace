"""Pure deterministic BudgetTrace controller."""

from .contracts import Decision, FeatureSnapshot


class BudgetController:
    def __init__(self, policy_version: str = "rules-v1") -> None:
        self.policy_version = policy_version
        self.stop_on_success = True

    def choose(self, snapshot: FeatureSnapshot) -> Decision:
        if snapshot.safety_blocked:
            return self._decision("stop", "safety_denied", snapshot)
        if snapshot.success:
            return self._decision("stop", "success", snapshot)
        if snapshot.budget_exhausted:
            return self._decision("stop", "budget_exhausted", snapshot)
        if snapshot.late_breakthrough_guard and snapshot.new_feedback:
            return self._decision("continue", "late_breakthrough_window", snapshot)
        if snapshot.late_breakthrough_guard and snapshot.can_replan:
            return self._decision("replan", "preserve_breakthrough_option", snapshot)
        if snapshot.high_repeat_no_novelty:
            if snapshot.can_replan:
                return self._decision("replan", "repeated_no_progress", snapshot)
            return self._decision("stop", "repeated_no_progress", snapshot)
        if snapshot.cheap_route_safe:
            return self._decision("route_cheaper", "low_complexity_window", snapshot)
        return self._decision("continue", "positive_or_uncertain_progress", snapshot)

    def _decision(self, action: str, reason_code: str, snapshot: FeatureSnapshot) -> Decision:
        return Decision(
            action=action,
            reason_code=reason_code,
            budget_window=snapshot.window_index,
            policy_version=self.policy_version,
        )
