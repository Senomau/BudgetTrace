# BudgetTrace Architecture

BudgetTrace separates agent execution into contracts, events, budget accounting, feature extraction, policy decisions, and adapter controls.

## Runtime Flow

1. `TaskContract` declares allowed tools, side-effect levels, success condition, and budget limits.
2. `Runtime` writes `task_started`, then asks the adapter for a conservative next-call estimate.
3. `BudgetLedger.admit()` rejects unaffordable reservations before transport.
4. After a model call returns, `BudgetLedger.reconcile()` records observed usage even if it overruns the original limit.
5. Tool calls and results are linked by `operation_id` and argument hash.
6. At each budget window, `extract_features()` builds auditable signals from events only.
7. `BudgetController` chooses `continue`, `replan`, `route_cheaper`, or `stop`.
8. Adapter controls are recorded as `control_applied` or `control_failed`.
9. `RunCard` summarizes terminal state, score, budget, over-budget dimensions, billing status, and journal JSONL.

## Core Modules

| Module | Responsibility |
|---|---|
| `contracts.py` | Pydantic contracts for tasks, events, decisions, observations, and run cards |
| `events.py` | Append-only event journal and budget ledger |
| `runtime.py` | Agent loop, budget admission/reconciliation, tool execution, control execution |
| `features.py` | Deterministic trajectory features |
| `policy.py` | Pure rule controller |
| `fixtures.py` | Scripted offline adapter and deterministic scorer |
| `live_agent.py` | OpenAI-compatible chat adapter with usage accounting and optional cheap route |
| `security.py` | Tool allowlist, path boundary, side-effect gate, recursive redaction |

## Boundary

The current architecture proves the execution protocol and safety boundaries. It does not prove model quality, online savings, or production billing correctness.
