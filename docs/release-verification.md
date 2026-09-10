# Release verification

This checklist defines the evidence required before publishing a clean GitHub package. It intentionally records commands and acceptance conditions, not local machine output, credentials, provider payloads, or run journals.

## Required checks

Run from the repository root:

```powershell
python -m pytest -q -p no:cacheprovider
python -m ruff check src tests
python -m mypy src/budgettrace
python -m build
python -m pip install --force-reinstall --no-deps dist/budgettrace-0.1.0-py3-none-any.whl
budgettrace --help
python -m budgettrace replay data/mini-v1/sample_run.jsonl
python -m budgettrace evaluate data/mini-v1/tasks.json --output output/release-eval
python -m budgettrace evaluate data/mini-v1/tasks.json --trials 3 --model-routes scripted-v1,scripted-v2 --output output/release-matrix
```

All listed commands must exit with code 0. The offline commands must not make network requests. The matrix command must emit 5 × 4 × 3 × 2 = 120 records and report `success_rate`, `false_stop_rate`, and `unit_success_cost_micros`; route labels are deterministic fixture dimensions, not live model comparisons.

The public package must contain both `README.md` and `README_EN.md`; each README must link to the other, and the English version must preserve the same evidence boundaries as the Chinese version.

## Public-tree checks

The upload tree must not contain:

- `.git/`, caches, `__pycache__/`, `*.pyc`, `dist/`, `build/`, or generated `output/`;
- raw live reports, JSONL journals, provider responses, upstream databases, or screenshots;
- `.env` values, API keys, Authorization headers, private-key blocks, URL credentials, or personal data;
- absolute paths belonging to a developer workstation.

The only task data intended for the repository is the synthetic `data/mini-v1/` fixture and the public manifest metadata under `data/public/`.

## Secret scan

Use a repository secret scanner such as Gitleaks in CI and inspect all findings manually. A local fallback for the most dangerous patterns is:

```powershell
rg -n -i "Bearer\s+[A-Za-z0-9._~+/-]{16,}|sk-[A-Za-z0-9_-]{20,}|BEGIN [A-Z ]*PRIVATE KEY|https://[^/[:space:]]+:[^/@[:space:]]+@|api[_-]?key\s*[:=]\s*[^<\[]" .
```

Environment variable names and redaction tests are expected matches; credential values are not.

## Interpretation boundary

Passing these checks proves package integrity, deterministic runtime behavior, and release hygiene. It does not prove live model quality, online cost savings, benchmark success rate, production SLOs, or real-world transaction safety.
