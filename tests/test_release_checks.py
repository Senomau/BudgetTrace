from pathlib import Path


ROOT = Path(__file__).parents[1]
README_EN = ROOT / "README.md"
README_CN = ROOT / "README_CN.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_metadata_files_and_pyproject_fields_exist():
    for relative_path in [
        "LICENSE",
        "NOTICE",
        "CONTRIBUTING.md",
        "SECURITY.md",
        ".env.example",
    ]:
        assert (ROOT / relative_path).exists(), relative_path

    pyproject = _read(ROOT / "pyproject.toml")

    assert 'readme = "README.md"' in pyproject
    assert 'license = "Apache-2.0"' in pyproject
    assert 'license-files = ["LICENSE", "NOTICE"]' in pyproject
    assert 'license = {text = "Apache-2.0"}' not in pyproject
    assert "License :: OSI Approved :: Apache Software License" not in pyproject
    assert 'budgettrace = "budgettrace.cli:main"' in pyproject
    assert "example.com" not in pyproject


def test_env_example_contains_live_api_key_placeholder_only():
    env_example = _read(ROOT / ".env.example")

    assert "BUDGETTRACE_LIVE_API_KEY=" in env_example
    assert "sk-" not in env_example


def test_release_snapshot_is_offline_only_and_documents_commands():
    readme_en = _read(README_EN)
    readme_cn = _read(README_CN)

    assert "python -m pip install -e" in readme_en
    assert "python -m budgettrace replay" in readme_en
    assert "python -m budgettrace evaluate" in readme_en
    assert "离线" in readme_cn
    assert not list(ROOT.rglob("*.env"))
    assert not list(ROOT.rglob("*api_key*"))


def test_readme_describes_live_code_without_claiming_scored_results():
    readme_en = _read(README_EN)
    readme_cn = _read(README_CN)
    lowered_en = readme_en.lower()

    assert "live" in lowered_en
    assert "scored: false" in lowered_en
    assert "当前实现不调用网络" not in readme_cn
    assert "cost savings" not in lowered_en


def test_readme_links_current_docs():
    readme_en = _read(README_EN)

    for path in [
        "docs/architecture.md",
        "docs/runbook.md",
        "docs/release-verification.md",
        "SECURITY.md",
    ]:
        assert path in readme_en


def test_readmes_offer_bilingual_switch():
    readme_en = _read(README_EN)
    readme_cn = _read(README_CN)

    assert "README_CN.md" in readme_en
    assert "README.md" in readme_cn
    assert "reference-actions" in readme_en
    assert "business-state or database scorer" in readme_en


def test_readme_preserves_live_boundary():
    readme_en = _read(README_EN).lower()
    readme_cn = _read(README_CN)

    assert "scored: false" in readme_en
    assert "生产系统" in readme_cn


def test_tracked_report_paths_are_public_allowlisted():
    report_files = (
        [
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "reports").rglob("*")
            if path.is_file()
        ]
        if (ROOT / "reports").exists()
        else []
    )

    assert not report_files


def test_ci_has_offline_quality_and_build_gates():
    workflow = _read(ROOT / ".github" / "workflows" / "ci.yml")
    lowered = workflow.lower()

    for required in [
        "3.9",
        "3.11",
        "3.12",
        "pytest",
        "ruff",
        "mypy",
        "python -m build",
        "gitleaks",
    ]:
        assert required in lowered
    assert "live-smoke" not in lowered
    assert "live-eval" not in lowered
    assert "live-episode" not in lowered
