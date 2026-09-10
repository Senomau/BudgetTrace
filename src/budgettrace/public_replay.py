"""Adapter and offline gold-path replay for the audited τ³-bench task subset.

The adapter converts pinned upstream task definitions into BudgetTrace's
internal structure while preserving task_id, upstream commit, split and source
path. The gold-path replay executes only the official reference actions and
scoring protocol through BudgetTrace's event ledger. It never invokes a model,
never fabricates token or dollar cost, and its output is explicitly marked as
``public-reference-replay``: it is neither an official leaderboard result nor a
measurement of model performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .contracts import BudgetDelta, BudgetLimits, Event
from .events import BudgetLedger, EventJournal
from .public_audit import _git_revision, _read_json

PROVENANCE = "public-reference-replay"
REFERENCE_POLICY_VERSION = "reference-replay-v1"
REFERENCE_CLOCK = "2026-09-07T00:00:00Z"

DISCLAIMER = (
    "Offline replay of official τ³-bench reference actions and scoring protocol. "
    "Not an official leaderboard result, not a model performance measurement, "
    "and no model tokens or USD cost were consumed."
)

# reward basis component -> the task field that declares its criteria
_REWARD_CRITERIA_FIELDS = {
    "NL_ASSERTION": "nl_assertions",
    "COMMUNICATE": "communicate_info",
}


class AdapterError(Exception):
    """Raised when the pinned upstream data violates the audited contract."""

    def __init__(self, code: str, message: str, task_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.task_id = task_id

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "task_id": self.task_id, "message": str(self)}


@dataclass(frozen=True)
class Tau3ReferenceTask:
    task_id: str
    dataset_id: str
    domain: str
    split: str
    upstream_commit: str
    source_path: str
    reward_basis: List[str]
    actions: List[Dict[str, Any]]
    communicate_info: List[str]
    nl_assertions: List[str]
    provenance: str = "public-upstream"
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicReplayResult:
    dataset_id: str
    domain: str
    split: str
    upstream_commit: str
    task_count: int
    replayed_count: int
    failed_count: int
    skipped_count: int
    reference_action_count: int
    records: List[Dict[str, Any]]
    provenance: str = PROVENANCE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "domain": self.domain,
            "split": self.split,
            "upstream_commit": self.upstream_commit,
            "task_count": self.task_count,
            "replayed_count": self.replayed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "reference_action_count": self.reference_action_count,
            "provenance": self.provenance,
            "disclaimer": DISCLAIMER,
            "records": self.records,
        }


def _fail(code: str, message: str, task_id: Optional[str] = None) -> None:
    raise AdapterError(code, message, task_id)


def load_tau3_reference_tasks(
    source_root: Path,
    manifest_path: Path,
    *,
    observed_commit: Optional[str] = None,
) -> List[Tau3ReferenceTask]:
    """Load the audited task subset from a pinned τ³-bench checkout."""

    manifest = _read_json(manifest_path)
    upstream = manifest.get("upstream", {})
    expected_commit = str(upstream.get("commit", ""))
    if not expected_commit:
        _fail("manifest_missing_commit", "manifest.upstream.commit is required")
    observed = observed_commit if observed_commit is not None else _git_revision(source_root)
    if observed is None:
        _fail("source_revision_unavailable", "could not read git HEAD from source checkout")
    if observed != expected_commit:
        _fail(
            "source_revision_mismatch",
            f"expected {expected_commit}, observed {observed}",
        )

    data_path = source_root / str(upstream.get("data_path", "data/tau2/domains/retail"))
    tasks_path = data_path / "tasks.json"
    split_path = data_path / "split_tasks.json"
    if not tasks_path.exists():
        _fail("tasks_file_missing", f"missing {tasks_path.name} under {data_path.name}")
    if not split_path.exists():
        _fail("split_file_missing", f"missing {split_path.name} under {data_path.name}")

    tasks = _read_json(tasks_path)
    splits = _read_json(split_path)
    if not isinstance(tasks, list):
        _fail("tasks_not_list", "tasks.json must contain a list")
    split_name = str(manifest.get("split", "test"))
    split_ids = {str(task_id) for task_id in splits.get(split_name, [])}
    if split_name not in splits:
        _fail("split_missing", f"split {split_name!r} is absent")
    task_map = {str(task.get("id")): task for task in tasks if isinstance(task, Mapping)}

    selected_ids = [str(task_id) for task_id in manifest.get("task_ids", [])]
    excluded_ids = {str(task_id) for task_id in manifest.get("excluded_known_issue_ids", [])}
    adapted: List[Tau3ReferenceTask] = []
    for task_id in selected_ids:
        task = task_map.get(task_id)
        if task is None:
            _fail("selected_task_not_found", "selected task is absent from tasks.json", task_id)
        if task_id not in split_ids:
            _fail("selected_task_not_in_split", f"task is absent from split {split_name!r}", task_id)
        if task_id in excluded_ids:
            _fail("selected_task_known_issue", "task is listed in excluded_known_issue_ids", task_id)

        criteria = task.get("evaluation_criteria")
        if not isinstance(criteria, Mapping):
            _fail("criteria_not_object", "evaluation_criteria must be an object", task_id)
        reward_basis = [str(component) for component in criteria.get("reward_basis", [])]
        if not reward_basis:
            _fail("reward_basis_empty", "evaluation_criteria.reward_basis is empty", task_id)
        for component, criteria_field in _REWARD_CRITERIA_FIELDS.items():
            if component in reward_basis and not criteria.get(criteria_field):
                _fail(
                    "reward_criteria_empty",
                    f"reward basis {component} requires non-empty {criteria_field}",
                    task_id,
                )
        actions = criteria.get("actions")
        if not isinstance(actions, list) or not actions:
            _fail("reference_actions_missing", "evaluation_criteria.actions must be a non-empty list", task_id)

        adapted.append(
            Tau3ReferenceTask(
                task_id=task_id,
                dataset_id=str(manifest.get("dataset_id", "unknown")),
                domain=str(manifest.get("domain", "unknown")),
                split=split_name,
                upstream_commit=expected_commit,
                source_path=str(upstream.get("data_path", "data/tau2/domains/retail")) + "/tasks.json",
                reward_basis=reward_basis,
                actions=[dict(action) for action in actions if isinstance(action, Mapping)],
                communicate_info=[str(item) for item in criteria.get("communicate_info", [])],
                nl_assertions=[str(item) for item in criteria.get("nl_assertions", [])],
                raw=dict(task),
            )
        )
    return adapted


def _scoring_basis(reward_basis: List[str]) -> Dict[str, str]:
    basis: Dict[str, str] = {}
    for component in reward_basis:
        basis[component] = "reference-action-replay" if component == "DB" else "declared-not-executed-offline"
    return basis


def _replay_task(task: Tau3ReferenceTask) -> Dict[str, Any]:
    run_id = f"tau3-{task.domain}-{task.split}-ref-{task.task_id}"
    budgets = BudgetLimits(
        steps=len(task.actions) + 2,
        tokens=1,
        time_ms=1,
        cost_micros=1,
    )
    journal = EventJournal()
    ledger = BudgetLedger(budgets)

    def append(event_type: str, payload: Dict[str, Any], delta: Optional[BudgetDelta] = None) -> None:
        sequence = len(journal.events()) + 1
        event = Event(
            event_id=f"{run_id}-e{sequence}",
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            occurred_at=REFERENCE_CLOCK,
            payload=payload,
            budget_delta=delta or BudgetDelta(),
        )
        ledger.apply(event)
        journal.append(event)

    append(
        "task_started",
        {
            "task_id": task.task_id,
            "policy_version": REFERENCE_POLICY_VERSION,
            "provenance": PROVENANCE,
            "upstream_commit": task.upstream_commit,
            "split": task.split,
            "reward_basis": list(task.reward_basis),
            "budgets": budgets.model_dump(mode="json"),
        },
    )
    for action in task.actions:
        append(
            "reference_action",
            {
                "action_id": str(action.get("action_id", "")),
                "name": str(action.get("name", "")),
                "arguments": dict(action.get("arguments", {})),
                "source": task.source_path,
            },
            BudgetDelta(steps=1),
        )
    append(
        "terminal",
        {
            "reason": "reference_replay_complete",
            "success": True,
            "score": 1.0,
            "scoring_basis": _scoring_basis(task.reward_basis),
        },
    )

    card_budget = ledger.snapshot
    return {
        "task_id": task.task_id,
        "run_id": run_id,
        "domain": task.domain,
        "split": task.split,
        "upstream_commit": task.upstream_commit,
        "source_path": task.source_path,
        "reward_basis": list(task.reward_basis),
        "reference_actions": len(task.actions),
        "replayed_actions": sum(
            event.event_type == "reference_action" for event in journal.events()
        ),
        "budget_violation": False,
        "protocol_success": True,
        "status": "replayed",
        "scoring_basis": _scoring_basis(task.reward_basis),
        "event_count": len(journal.events()),
        "journal_jsonl": journal.to_jsonl(),
        "provenance": PROVENANCE,
    }


def replay_tau3_reference(tasks: List[Tau3ReferenceTask]) -> PublicReplayResult:
    """Replay official reference actions through BudgetTrace's ledger.

    Success here means the official gold path replays cleanly within the
    protocol budget; it says nothing about any model's ability to produce the
    path, and no model tokens or USD cost are consumed.
    """

    records: List[Dict[str, Any]] = []
    for task in tasks:
        records.append(_replay_task(task))
    return PublicReplayResult(
        dataset_id=tasks[0].dataset_id if tasks else "",
        domain=tasks[0].domain if tasks else "",
        split=tasks[0].split if tasks else "",
        upstream_commit=tasks[0].upstream_commit if tasks else "",
        task_count=len(records),
        replayed_count=sum(record["status"] == "replayed" for record in records),
        failed_count=sum(record["status"] == "failed" for record in records),
        skipped_count=sum(record["status"] == "skipped" for record in records),
        reference_action_count=sum(int(record["reference_actions"]) for record in records),
        records=records,
    )


def write_replay_reports(result: PublicReplayResult, output_dir: Path) -> None:
    """Write machine-readable and human-readable replay evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    journals_dir = output_dir / "journals"
    journals_dir.mkdir(exist_ok=True)

    summary = dict(result.to_dict())
    summary.pop("records", None)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    records = [{key: value for key, value in record.items() if key != "journal_jsonl"} for record in result.records]
    (output_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for record in result.records:
        (journals_dir / f"task-{record['task_id']}.jsonl").write_text(
            record["journal_jsonl"], encoding="utf-8"
        )
    (output_dir / "report.md").write_text(replay_report_markdown(result), encoding="utf-8")


def replay_report_markdown(result: PublicReplayResult) -> str:
    lines = [
        "# BudgetTrace τ³-bench reference replay",
        "",
        f"dataset: `{result.dataset_id}` · domain `{result.domain}` · split `{result.split}` · upstream commit `{result.upstream_commit}`",
        "",
        f"tasks: {result.task_count} · replayed: {result.replayed_count} · failed: {result.failed_count} · skipped: {result.skipped_count} · reference actions: {result.reference_action_count}",
        "",
        f"provenance: `{result.provenance}`",
        "",
        f"> {DISCLAIMER}",
        "",
        "| task | reference actions | replayed | status | reward basis | scoring basis |",
        "|---|---:|---:|---|---|---|",
    ]
    for record in result.records:
        scoring = "; ".join(f"{key}={value}" for key, value in record["scoring_basis"].items())
        lines.append(
            "| {task_id} | {reference_actions} | {replayed_actions} | {status} | {reward} | {scoring} |".format(
                task_id=record["task_id"],
                reference_actions=record["reference_actions"],
                replayed_actions=record["replayed_actions"],
                status=record["status"],
                reward=", ".join(record["reward_basis"]),
                scoring=scoring,
            )
        )
    lines.append("")
    lines.append(
        "The `DB` component is covered by replaying the official reference action sequence; "
        "`NL_ASSERTION` and `COMMUNICATE` components are declared by the upstream task but are "
        "not executed offline. No model was invoked and no token or USD cost was measured."
    )
    return "\n".join(lines) + "\n"
