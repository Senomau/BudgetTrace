# Runbook

All commands below are designed for a clean checkout. Generated files must stay under ignored `output/` or a temporary directory.

## Install and offline checks

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q -p no:cacheprovider
python -m ruff check src tests
python -m mypy src/budgettrace
python -m build
python -m pip install --force-reinstall --no-deps dist/budgettrace-0.1.0-py3-none-any.whl
budgettrace --help
python -m budgettrace replay data/mini-v1/sample_run.jsonl
python -m budgettrace evaluate data/mini-v1/tasks.json --output output/offline-eval
python -m budgettrace evaluate data/mini-v1/tasks.json --trials 3 --model-routes scripted-v1,scripted-v2 --output output/p0-matrix
```

These commands do not call a model provider.

The matrix command runs 5 fixed tasks × 4 policies × 3 trials × 2 route labels and writes per-run records plus aggregate success rate, late-breakthrough false-stop rate, and unit success cost. The two `scripted-*` routes share the same deterministic ScriptedLLM fixture; they validate matrix accounting, not real-model quality differences.

## Public reference replay

The manifest records the expected upstream revision. The upstream checkout is intentionally not bundled.

```powershell
python -m budgettrace public-audit --source <path-to-upstream-checkout> --manifest data/public/tau3-retail-v1/manifest.json --output output/public-audit.json
python -m budgettrace public-replay --source <path-to-upstream-checkout> --manifest data/public/tau3-retail-v1/manifest.json --output output/public-replay
```

Reference replay validates the adapter and scoring protocol. It is not a model evaluation.

## Optional live validation

Credentials are read from the current process environment only. Never save them in `.env` files, reports, notebooks, screenshots, or shell history.

```powershell
$env:BUDGETTRACE_LIVE_BASE_URL = "<provider endpoint>"
$env:BUDGETTRACE_LIVE_MODEL = "<model id>"
$env:BUDGETTRACE_LIVE_API_KEY = "<secret in shell only>"
$env:BUDGETTRACE_LIVE_PRICE_INPUT_PER_MTOK = "<input USD per 1M tokens>"
$env:BUDGETTRACE_LIVE_PRICE_OUTPUT_PER_MTOK = "<output USD per 1M tokens>"
$env:BUDGETTRACE_LIVE_TIMEOUT = "180"
python -m budgettrace live-config-check
python -m budgettrace live-smoke --source <path-to-upstream-checkout> --manifest data/public/tau3-retail-v1/manifest.json --task <task-id> --output output/live-smoke
python -m budgettrace live-eval --source <path-to-upstream-checkout> --manifest data/public/tau3-retail-v1/manifest.json --tasks <task-id> --trials 1 --budget-usd 0.50 --output output/live-eval
python -m budgettrace live-episode --source <path-to-upstream-checkout> --manifest data/public/tau3-retail-v1/manifest.json --task <task-id> --max-steps 8 --budget-window-steps 2 --output output/live-episode
# scored action-contract matrix; route metadata uses data/public/tau3-retail-v1/model-routes.example.json
python -m budgettrace live-matrix --source <path-to-upstream-checkout> --manifest data/public/tau3-retail-v1/manifest.json --routes data/public/tau3-retail-v1/model-routes.example.json --tasks 70 --trials 1 --scorer-mode reference-actions --output output/live-matrix
```

Live results are informational by default. `success` means the agent emitted a finish action; it is not a benchmark score or business-success claim. Costs are estimates from the declared price table. The `reference-actions` matrix mode additionally checks the declared successful tool-action sequence and `finish`; it is a deterministic protocol proxy, not the upstream DB/NL scorer. Route files contain only endpoint/model/price metadata and an `api_key_env` name; keys stay in the process environment.

## Release hygiene

Before publishing, inspect the exact package tree and run a secret scan. Do not include `.git`, caches, build output, live output, raw journals, upstream databases, local absolute paths, or provider credentials. See [release-verification.md](release-verification.md) and [SECURITY.md](../SECURITY.md).
