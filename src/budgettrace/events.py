"""Append-only JSONL events and idempotent budget accounting."""

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Set

from .contracts import BudgetDelta, BudgetLimits, BudgetSnapshot, Event


class BudgetExceededError(ValueError):
    """Raised when an event reservation would exceed a contract budget."""

    def __init__(self, dimensions: Iterable[str]) -> None:
        self.dimensions = list(dimensions)
        super().__init__("budget exceeded: " + ", ".join(self.dimensions))


def canonical_args_hash(args: Mapping[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EventJournal:
    def __init__(self, events: Iterable[Event] = ()) -> None:
        self._events: List[Event] = []
        self._event_ids: Set[str] = set()
        self._run_id = None
        for event in events:
            self.append(event)

    def append(self, event: Event) -> None:
        if event.event_id in self._event_ids:
            raise ValueError(f"duplicate event_id: {event.event_id}")
        if self._run_id is None:
            self._run_id = event.run_id
        if event.run_id != self._run_id:
            raise ValueError("all events in a journal must share run_id")
        expected = len(self._events) + 1
        if event.sequence != expected:
            raise ValueError(f"sequence must be {expected}, got {event.sequence}")
        self._events.append(event)
        self._event_ids.add(event.event_id)

    def events(self) -> List[Event]:
        return list(self._events)

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            for event in self._events
        )

    @classmethod
    def from_jsonl(cls, text: str) -> "EventJournal":
        journal = cls()
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = Event.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"invalid event at line {line_number}: {exc}") from exc
            journal.append(event)
        return journal


class BudgetLedger:
    def __init__(self, limits: BudgetLimits) -> None:
        self._limits = limits
        self._steps = 0
        self._tokens = 0
        self._time_ms = 0
        self._cost_micros = 0
        self._applied_event_fingerprints: Dict[str, str] = {}
        self._applied_event_sequences: Dict[str, int] = {}
        self._run_id = None
        self._last_sequence = 0

    @property
    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            limits=self._limits,
            steps_used=self._steps,
            tokens_used=self._tokens,
            time_ms_used=self._time_ms,
            cost_micros_used=self._cost_micros,
        )

    def apply(self, event: Event) -> BudgetSnapshot:
        self.admit(event.budget_delta)
        return self.reconcile(event)

    def exceeded_dimensions(self, delta: BudgetDelta) -> List[str]:
        proposed = (
            self._steps + delta.steps,
            self._tokens + delta.tokens,
            self._time_ms + delta.time_ms,
            self._cost_micros + delta.cost_micros,
        )
        limits = (
            self._limits.steps,
            self._limits.tokens,
            self._limits.time_ms,
            self._limits.cost_micros,
        )
        names = ("steps", "tokens", "time_ms", "cost_micros")
        return [name for name, value, limit in zip(names, proposed, limits) if value > limit]

    def admit(self, delta: BudgetDelta) -> None:
        if delta.steps == 0 and delta.tokens == 0 and delta.time_ms == 0 and delta.cost_micros == 0:
            return
        exceeded = self.exceeded_dimensions(delta)
        if exceeded:
            raise BudgetExceededError(exceeded)

    def reconcile(self, event: Event) -> BudgetSnapshot:
        fingerprint = json.dumps(
            event.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        if self._run_id is None:
            self._run_id = event.run_id
        elif event.run_id != self._run_id:
            raise ValueError("ledger run_id mismatch")
        if event.event_id in self._applied_event_fingerprints:
            if event.sequence != self._applied_event_sequences[event.event_id]:
                raise ValueError(
                    f"conflicting duplicate event sequence for event_id: {event.event_id}"
                )
            if self._applied_event_fingerprints[event.event_id] != fingerprint:
                raise ValueError(f"conflicting duplicate event_id: {event.event_id}")
            return self.snapshot
        expected_sequence = self._last_sequence + 1
        if event.sequence != expected_sequence:
            raise ValueError(f"ledger sequence must be {expected_sequence}, got {event.sequence}")
        delta: BudgetDelta = event.budget_delta
        self._steps += delta.steps
        self._tokens += delta.tokens
        self._time_ms += delta.time_ms
        self._cost_micros += delta.cost_micros
        self._applied_event_fingerprints[event.event_id] = fingerprint
        self._applied_event_sequences[event.event_id] = event.sequence
        self._last_sequence = event.sequence
        return self.snapshot
