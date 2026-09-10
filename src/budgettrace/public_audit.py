"""Audit a pinned public τ³-bench task subset before importing it."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class AuditFinding:
    code: str
    severity: str
    message: str
    task_id: Optional[str] = None


@dataclass(frozen=True)
class PublicAuditResult:
    dataset_id: str
    domain: str
    split: str
    expected_commit: str
    observed_commit: Optional[str]
    source_task_count: int
    split_task_count: int
    selected_task_count: int
    selected_task_ids: List[str]
    excluded_known_issue_ids: List[str]
    findings: List[AuditFinding]
    provenance: str = "public-upstream-audit"

    @property
    def passed(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_revision(source_root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _finding(
    findings: List[AuditFinding],
    code: str,
    message: str,
    *,
    task_id: Optional[str] = None,
    severity: str = "error",
) -> None:
    findings.append(AuditFinding(code=code, severity=severity, message=message, task_id=task_id))


def _empty_result(manifest: Mapping[str, Any], findings: Iterable[AuditFinding]) -> PublicAuditResult:
    upstream = manifest.get("upstream", {})
    selected_ids = [str(task_id) for task_id in manifest.get("task_ids", [])]
    return PublicAuditResult(
        dataset_id=str(manifest.get("dataset_id", "unknown")),
        domain=str(manifest.get("domain", "unknown")),
        split=str(manifest.get("split", "unknown")),
        expected_commit=str(upstream.get("commit", "")),
        observed_commit=None,
        source_task_count=0,
        split_task_count=0,
        selected_task_count=0,
        selected_task_ids=selected_ids,
        excluded_known_issue_ids=[str(task_id) for task_id in manifest.get("excluded_known_issue_ids", [])],
        findings=list(findings),
    )


def audit_tau3_retail(
    source_root: Path,
    manifest_path: Path,
    *,
    observed_commit: Optional[str] = None,
) -> PublicAuditResult:
    """Validate the selected τ³-bench Retail subset against a pinned checkout."""

    manifest = _read_json(manifest_path)
    findings: List[AuditFinding] = []
    upstream = manifest.get("upstream", {})
    expected_commit = str(upstream.get("commit", ""))
    observed = observed_commit if observed_commit is not None else _git_revision(source_root)
    if not expected_commit:
        _finding(findings, "manifest_missing_commit", "manifest.upstream.commit is required")
    if observed is None:
        _finding(findings, "source_revision_unavailable", "could not read git HEAD from source checkout")
    elif expected_commit and observed != expected_commit:
        _finding(
            findings,
            "source_revision_mismatch",
            f"expected {expected_commit}, observed {observed}",
        )

    data_path = source_root / str(upstream.get("data_path", "data/tau2/domains/retail"))
    tasks_path = data_path / "tasks.json"
    split_path = data_path / "split_tasks.json"
    selected_ids = [str(task_id) for task_id in manifest.get("task_ids", [])]
    excluded_ids = [str(task_id) for task_id in manifest.get("excluded_known_issue_ids", [])]
    if not tasks_path.exists():
        _finding(findings, "tasks_file_missing", f"missing {tasks_path}")
        return _empty_result(manifest, findings)
    if not split_path.exists():
        _finding(findings, "split_file_missing", f"missing {split_path}")
        return _empty_result(manifest, findings)

    tasks = _read_json(tasks_path)
    splits = _read_json(split_path)
    if not isinstance(tasks, list):
        _finding(findings, "tasks_not_list", "tasks.json must contain a list")
        return _empty_result(manifest, findings)
    task_map = {str(task.get("id")): task for task in tasks if isinstance(task, Mapping)}
    if len(task_map) != len(tasks):
        _finding(findings, "duplicate_or_invalid_task_ids", "tasks.json contains duplicate or invalid task ids")
    split_name = str(manifest.get("split", "test"))
    split_ids = {str(task_id) for task_id in splits.get(split_name, [])}
    if split_name not in splits:
        _finding(findings, "split_missing", f"split {split_name!r} is absent")

    for task_id in selected_ids:
        task = task_map.get(task_id)
        if task is None:
            _finding(findings, "selected_task_not_found", "selected task is absent from tasks.json", task_id=task_id)
            continue
        if task_id not in split_ids:
            _finding(findings, "selected_task_not_in_split", f"task is absent from split {split_name!r}", task_id=task_id)
        if task_id in excluded_ids:
            _finding(findings, "selected_task_known_issue", "task is listed in excluded_known_issue_ids", task_id=task_id)
        criteria = task.get("evaluation_criteria", {})
        if not isinstance(criteria, Mapping):
            _finding(findings, "criteria_not_object", "evaluation_criteria must be an object", task_id=task_id)
            continue
        reward_basis = [str(component) for component in criteria.get("reward_basis", [])]
        if not reward_basis:
            _finding(findings, "reward_basis_empty", "evaluation_criteria.reward_basis is empty", task_id=task_id)
        required_fields = {
            "NL_ASSERTION": "nl_assertions",
            "COMMUNICATE": "communicate_info",
        }
        for component, field in required_fields.items():
            if component in reward_basis and not criteria.get(field):
                _finding(
                    findings,
                    "reward_criteria_empty",
                    f"reward basis {component} requires non-empty {field}",
                    task_id=task_id,
                )

    return PublicAuditResult(
        dataset_id=str(manifest.get("dataset_id", "unknown")),
        domain=str(manifest.get("domain", "unknown")),
        split=split_name,
        expected_commit=expected_commit,
        observed_commit=observed,
        source_task_count=len(tasks),
        split_task_count=len(split_ids),
        selected_task_count=len(selected_ids),
        selected_task_ids=selected_ids,
        excluded_known_issue_ids=excluded_ids,
        findings=findings,
    )
