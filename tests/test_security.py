import pytest


def _contract(tmp_path):
    from budgettrace.contracts import BudgetLimits, TaskContract

    return TaskContract(
        task_id="secure-001",
        workspace_root=str(tmp_path),
        allowed_tools=["read_file"],
        tool_levels={"read_file": "L0"},
        budgets=BudgetLimits(steps=4, tokens=1000, time_ms=1000, cost_micros=1000),
        success_condition="read_ok",
        scorer_version="fixture-v1",
    )


def test_unlisted_tool_is_denied_fail_closed(tmp_path):
    from budgettrace.security import ToolDeniedError, ToolRegistry

    registry = ToolRegistry()
    registry.register("read_file", "L0", lambda args: {"ok": True})

    with pytest.raises(ToolDeniedError, match="not allowlisted"):
        registry.invoke("shell", {}, _contract(tmp_path))


def test_path_traversal_is_denied_and_secret_arguments_are_redacted(tmp_path):
    from budgettrace.security import ToolDeniedError, ToolRegistry

    registry = ToolRegistry()
    registry.register("read_file", "L0", lambda args: {"content": "safe"})
    contract = _contract(tmp_path)

    with pytest.raises(ToolDeniedError, match="outside workspace"):
        registry.invoke(
            "read_file",
            {"path": "../secret.txt", "api_key": "do-not-log"},
            contract,
        )

    result = registry.invoke(
        "read_file",
        {"path": "input.txt", "api_key": "do-not-log"},
        contract,
    )
    assert result.redacted_args["api_key"] == "[REDACTED]"


def test_l3_tool_requires_explicit_approval(tmp_path):
    from budgettrace.security import ToolDeniedError, ToolRegistry

    registry = ToolRegistry()
    registry.register("publish", "L3", lambda args: {"published": True})
    contract = _contract(tmp_path).model_copy(
        update={"allowed_tools": ["publish"], "tool_levels": {"publish": "L3"}}
    )

    with pytest.raises(ToolDeniedError, match="approval"):
        registry.invoke("publish", {}, contract)

    result = registry.invoke("publish", {}, contract, approved=True)
    assert result.output == {"published": True}


@pytest.mark.parametrize(
    "secret",
    [
        "Bearer" + " " + "abcdefghijklmnopqrstuvwxyz123456",
        "https://" + "user:password@" + "example.com/path",
        "-----BEGIN " + "PRIVATE KEY-----",
        "sk-" + "abcdefghijklmnopqrstuvwxyz1234567890",
    ],
)
def test_redacts_secret_patterns_in_innocently_named_values(secret):
    from budgettrace.security import redact_secrets

    assert redact_secrets({"message": secret}) == {"message": "[REDACTED]"}
