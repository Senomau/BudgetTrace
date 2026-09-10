from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_release_metadata_files_and_pyproject_fields_exist():
    for relative_path in [
        "LICENSE",
        "NOTICE",
        "CONTRIBUTING.md",
        "SECURITY.md",
        ".env.example",
    ]:
        assert (ROOT / relative_path).exists(), relative_path

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'readme = "README.md"' in pyproject
    assert 'license = "Apache-2.0"' in pyproject
    assert 'license-files = ["LICENSE", "NOTICE"]' in pyproject
    assert 'license = {text = "Apache-2.0"}' not in pyproject
    assert "License :: OSI Approved :: Apache Software License" not in pyproject
    assert 'budgettrace = "budgettrace.cli:main"' in pyproject
    assert "example.com" not in pyproject


def test_env_example_contains_live_api_key_placeholder_only():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "BUDGETTRACE_LIVE_API_KEY=" in env_example
    assert "sk-" not in env_example


def test_release_snapshot_is_offline_only_and_documents_commands():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m pip install -e" in readme
    assert "python -m budgettrace replay" in readme
    assert "python -m budgettrace evaluate" in readme
    assert "离线" in readme
    assert not list(ROOT.rglob("*.env"))
    assert not list(ROOT.rglob("*api_key*"))


def test_readme_describes_live_code_without_claiming_scored_results():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    assert "live" in readme.lower()
    assert "scored: false" in lowered
    assert "当前实现不调用网络" not in readme
    assert "cost savings" not in lowered
    assert "benchmark" in lowered


def test_readme_links_current_docs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for path in ["docs/architecture.md", "docs/runbook.md", "docs/release-verification.md", "SECURITY.md"]:
        assert path in readme


def test_readmes_offer_bilingual_switch():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_en = (ROOT / "README_EN.md").read_text(encoding="utf-8")

    assert "README_EN.md" in readme
    assert "README.md" in readme_en
    assert "reference-actions" in readme_en
    assert "business-state or database scorer" in readme_en


def test_readme_preserves_live_boundary():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "scored: false" in readme
    assert "benchmark" in readme.lower()
    assert "生产系统" in readme


def test_tracked_report_paths_are_public_allowlisted():
    report_files = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "reports").rglob("*")
        if path.is_file()
    ] if (ROOT / "reports").exists() else []

    assert not report_files


def test_ci_has_offline_quality_and_build_gates():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lowered = workflow.lower()

    for required in ["3.9", "3.11", "3.12", "pytest", "ruff", "mypy", "python -m build", "gitleaks"]:
        assert required in lowered
    assert "live-smoke" not in lowered
    assert "live-eval" not in lowered
    assert "live-episode" not in lowered
