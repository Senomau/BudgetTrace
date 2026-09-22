# BudgetTrace: Runtime Budget Control for Long-Horizon Agents

[中文 README](README_CN.md) · English

BudgetTrace is an Agent Runtime and evaluation harness for tool-using, long-horizon agents. It sits between the agent, model gateway, and tool executor. Every model call, tool result, error, score, and budget change becomes an auditable event. At each budget window, `BudgetController` chooses `continue`, `replan`, `route_cheaper`, or `stop`.

## What it solves

Long-running agents can repeat failed tool calls, spend tokens without new information, or stop just before a late breakthrough. A raw `max_steps` limit caps worst-case spend but does not explain the decision or measure the quality–cost trade-off.

```text
model action -> tool safety gate -> event journal + budget ledger
                                      |
                         trajectory features + controller
                                      |
                 continue / replan / route_cheaper / stop
                                      |
                       RunCard / replay / evaluation
```

## Main components

- `TaskContract` and `RunConfig`: task, tool allowlist, success condition, scorer, and budget limits.
- `Runtime`: drives the model–tool–feedback loop and enforces the ledger before every call.
- `EventJournal` and `BudgetLedger`: append-only JSONL evidence with token, time, and cost accounting.
- `BudgetController`: interpretable decisions from score progress, novelty, repetition, errors, and context pressure.
- `BusinessContractScorer`: deterministic offline scorer requiring declared tool evidence and feedback IDs.
- `Replay` and evaluation matrix: compares fixed steps, fixed budget, naive stop, and BudgetTrace rules.
- Live adapters: OpenAI-compatible smoke, single-call evaluation, multi-turn episodes, and route matrix execution.

## Evidence in this release

| Evidence | Result | Scope |
|---|---:|---|
| Automated tests | 181 passed | Runtime, ledger, safety, replay, live adapters, reports |
| Offline P0 matrix | 120 runs | 5 tasks × 3 trials × 2 fixture routes × 4 policies |
| Public τ³-bench audit | 10/10 tasks | Pinned v1.0.1 retail test subset, 0 audit findings |
| Reference replay | 37 actions | Deterministic protocol replay, no model calls |
| Live pipeline | Multiple providers | Token, latency, cost estimate, tool protocol, and terminal evidence |

The offline matrix reports success rate, late-breakthrough false-stop rate, and unit success cost. The two `scripted-*` routes intentionally share one deterministic fixture; they validate denominators and reporting, not model quality differences.

## Quick start

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q -p no:cacheprovider
python -m ruff check src tests
python -m mypy src/budgettrace
python -m budgettrace replay data/mini-v1/sample_run.jsonl
python -m budgettrace evaluate data/mini-v1/tasks.json --trials 3 --model-routes scripted-v1,scripted-v2 --output output/p0-matrix
```

The offline commands do not call a model provider. Generated files belong under ignored `output/` and should not be committed.

## Live route matrix

Copy `data/public/tau3-retail-v1/model-routes.example.json`, keep only public endpoint/model/price metadata in the file, and set each route's `api_key_env` variable in the current process. Credentials are read from the current process environment; the repository and generated reports contain no credential values.

```powershell
python -m budgettrace live-matrix `
  --source <path-to-tau2-bench-v1.0.1> `
  --manifest data/public/tau3-retail-v1/manifest.json `
  --routes data/public/tau3-retail-v1/model-routes.example.json `
  --tasks 70 --trials 1 --scorer-mode reference-actions `
  --output output/live-matrix
```

The default live episode remains informational and uses `scored: false`; its `success` means that the model emitted `finish`. `reference-actions` additionally checks the declared successful tool-action sequence and `finish`, producing a deterministic protocol proxy. A real business-state or database scorer is not included in this release, so these records must not be presented as official τ²-bench reward, online transaction success, or validated production cost performance.

## Repository layout

```text
src/budgettrace/       runtime, controller, replay, live adapters
tests/                 offline, safety, live transport, and release tests
data/mini-v1/          deterministic synthetic tasks
data/public/           pinned public manifests and route example
docs/                  architecture, runbook, and release checks
.github/workflows/     offline CI, build, and secret scan
```

## Documentation

- [Architecture](docs/architecture.md)
- [Runbook](docs/runbook.md)
- [Release verification](docs/release-verification.md)
- [Security policy](SECURITY.md)
- [Contributing guide](CONTRIBUTING.md)
- [License](LICENSE)
