import json

from budgettrace.contracts import BudgetLimits, TaskContract
from budgettrace.events import EventJournal
from budgettrace.live_episode import run_live_episode
from budgettrace.public_replay import Tau3ReferenceTask
from budgettrace.security import ToolRegistry
from test_live_agent import _json_response
from test_live_eval import _sequential_transport
from test_live_smoke import _config


def _task70() -> Tau3ReferenceTask:
    return Tau3ReferenceTask(
        task_id="70",
        dataset_id="tau3-retail-audited-v1",
        domain="retail",
        split="test",
        upstream_commit="fc0055dc4e0a316c3f83133267fbd6faaa770992",
        source_path="data/tau2/domains/retail/tasks.json",
        reward_basis=["DB", "NL_ASSERTION"],
        actions=[
            {
                "action_id": "70_0",
                "name": "exchange_delivered_order_items",
                "arguments": {
                    "order_id": "ORDER-TEST-001",
                    "item_ids": ["ITEM-OLD-001"],
                    "new_item_ids": ["ITEM-NEW-001"],
                    "payment_method_id": "payment-test-001",
                },
            }
        ],
        communicate_info=["22.55"],
        nl_assertions=[
            "Agent should tell the user they need to pay $22.55 for the price difference."
        ],
        raw={
            "user_scenario": {
                "instructions": {
                    "task_instructions": "You are impatient.",
                    "reason_for_call": "You want to exchange a helmet.",
                }
            }
        },
    )


def _source(tmp_path):
    data_dir = tmp_path / "tau2" / "data" / "tau2" / "domains" / "retail"
    data_dir.mkdir(parents=True)
    (data_dir / "db.json").write_text(
        json.dumps(
            {
                "products": {
                    "PRODUCT-TEST-001": {
                        "product_id": "PRODUCT-TEST-001",
                        "name": "Cycling Helmet",
                        "variants": {
                            "ITEM-OLD-001": {
                                "item_id": "ITEM-OLD-001",
                                "options": {"size": "S", "color": "red", "ventilation": "low"},
                                "available": True,
                                "price": 197.33,
                            },
                            "ITEM-NEW-001": {
                                "item_id": "ITEM-NEW-001",
                                "options": {"size": "M", "color": "blue", "ventilation": "high"},
                                "available": True,
                                "price": 219.88,
                            },
                        },
                    }
                },
                "users": {
                    "test_user_001": {
                        "user_id": "test_user_001",
                        "name": {"first_name": "Test", "last_name": "User"},
                        "address": {"zip": "00000"},
                        "email": "test-user-001@example.invalid",
                        "payment_methods": {
                            "payment-test-001": {
                                "source": "credit_card",
                                "id": "payment-test-001",
                                "brand": "visa",
                                "last_four": "7312",
                            }
                        },
                        "orders": ["ORDER-TEST-001"],
                    }
                },
                "orders": {
                    "ORDER-TEST-001": {
                        "order_id": "ORDER-TEST-001",
                        "user_id": "test_user_001",
                        "address": {"zip": "00000"},
                        "items": [
                            {
                                "name": "Cycling Helmet",
                                "product_id": "PRODUCT-TEST-001",
                                "item_id": "ITEM-OLD-001",
                                "price": 197.33,
                                "options": {"size": "S", "color": "red", "ventilation": "low"},
                            }
                        ],
                        "status": "delivered",
                        "fulfillments": [],
                        "payment_history": [
                            {
                                "transaction_type": "payment",
                                "amount": 197.33,
                                "payment_method_id": "payment-test-001",
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path / "tau2"


def _limits():
    return BudgetLimits(steps=8, tokens=100000, time_ms=10**9, cost_micros=10**9)


def test_tau2_retail_environment_executes_task70_exchange(tmp_path):
    from budgettrace.tau2_retail import RetailToolEnvironment

    environment = RetailToolEnvironment.from_source(_source(tmp_path), _task70())

    result = environment.invoke(
        "exchange_delivered_order_items",
        {
            "order_id": "ORDER-TEST-001",
            "item_ids": ["ITEM-OLD-001"],
            "new_item_ids": ["ITEM-NEW-001"],
            "payment_method_id": "payment-test-001",
        },
    )

    assert result["status"] == "exchange requested"
    assert result["exchange_items"] == ["ITEM-OLD-001"]
    assert result["exchange_new_items"] == ["ITEM-NEW-001"]
    assert result["exchange_price_difference"] == 22.55


def test_register_tau2_retail_tools_allows_official_and_compat_names(tmp_path):
    from budgettrace.tau2_retail import RetailToolEnvironment, register_retail_tools

    environment = RetailToolEnvironment.from_source(_source(tmp_path), _task70())
    registry = ToolRegistry()
    registered = register_retail_tools(registry, environment)
    contract = TaskContract(
        task_id="70",
        workspace_root=str(tmp_path),
        allowed_tools=registered.allowed_tools,
        tool_levels=registered.tool_levels,
        budgets=_limits(),
        success_condition="model_emitted_finish",
        scorer_version="unscored-v1",
    )

    official = registry.invoke(
        "get_order_details", {"order_id": "ORDER-TEST-001"}, contract
    ).output
    compat = registry.invoke("get_order", {"order_id": "ORDER-TEST-001"}, contract).output

    assert official["order_id"] == "ORDER-TEST-001"
    assert compat["order_id"] == "ORDER-TEST-001"
    assert "process_exchange_request" in registered.allowed_tools


def test_retail_tool_schemas_are_official_and_strict():
    from budgettrace.tau2_retail import OFFICIAL_RETAIL_TOOL_LEVELS, retail_tool_schemas

    schemas = retail_tool_schemas()
    names = [schema["function"]["name"] for schema in schemas]
    assert names == sorted(OFFICIAL_RETAIL_TOOL_LEVELS)
    assert "get_order" not in names
    exchange = next(schema for schema in schemas if schema["function"]["name"] == "exchange_delivered_order_items")
    assert exchange["function"]["parameters"]["additionalProperties"] is False
    assert "payment_method_id" in exchange["function"]["parameters"]["required"]


def test_tau2_retail_prompt_requires_official_names_and_rejects_unknown_names(tmp_path):
    from budgettrace.tau2_retail import (
        COMPAT_RETAIL_TOOL_LEVELS,
        OFFICIAL_RETAIL_TOOL_LEVELS,
        RetailToolEnvironment,
    )

    prompt = RetailToolEnvironment.from_source(_source(tmp_path), _task70()).tool_prompt()

    assert "STRICT TOOL PROTOCOL" in prompt
    assert "Tool names are case-sensitive" in prompt
    assert '"tool":"<official name>"' in prompt
    assert "Never emit ask_user, get_orders" in prompt
    for name in OFFICIAL_RETAIL_TOOL_LEVELS:
        assert f"- {name}:" in prompt
    for name in COMPAT_RETAIL_TOOL_LEVELS:
        assert f"- {name} ->" in prompt
        assert f"(do not emit the alias)" in prompt


def test_tau2_retail_native_schemas_are_official_and_task70_ready():
    from budgettrace.tau2_retail import retail_tool_schemas

    schemas = retail_tool_schemas()
    names = [schema["function"]["name"] for schema in schemas]
    assert "find_user_id_by_name_zip" in names
    assert "exchange_delivered_order_items" in names
    assert "process_exchange_request" not in names
    exchange = next(
        schema for schema in schemas
        if schema["function"]["name"] == "exchange_delivered_order_items"
    )
    assert exchange["function"]["parameters"]["required"] == [
        "order_id", "item_ids", "new_item_ids", "payment_method_id"
    ]
    assert exchange["function"]["parameters"]["additionalProperties"] is False


def test_live_episode_uses_tau2_retail_source_instead_of_noop(tmp_path):
    from budgettrace.tau2_retail import RETAIL_TOOL_ENVIRONMENT

    transport = _sequential_transport(
        [
            _json_response(
                {
                    "kind": "tool_call",
                    "tool": "get_order_details",
                    "args": {"order_id": "ORDER-TEST-001"},
                }
            ),
            _json_response(
                {
                    "kind": "tool_call",
                    "tool": "exchange_delivered_order_items",
                    "args": {
                        "order_id": "ORDER-TEST-001",
                        "item_ids": ["ITEM-OLD-001"],
                        "new_item_ids": ["ITEM-NEW-001"],
                        "payment_method_id": "payment-test-001",
                    },
                }
            ),
            _json_response({"kind": "finish", "score": 1.0}),
        ]
    )

    record = run_live_episode(
        _config(),
        _task70(),
        limits=_limits(),
        source_root=_source(tmp_path),
        transport=transport,
    )

    events = EventJournal.from_jsonl(record["journal_jsonl"]).events()
    errors = [event for event in events if event.event_type == "error"]
    tools = [event.payload["tool"] for event in events if event.event_type == "tool_result"]

    assert record["terminal_reason"] == "success"
    assert record["tool_environment"] == RETAIL_TOOL_ENVIRONMENT
    assert tools == ["get_order_details", "exchange_delivered_order_items"]
    assert errors == []
    assert record["scored"] is False
    system_prompt = transport.calls[0]["payload"]["messages"][0]["content"]
    assert "exchange_delivered_order_items" in system_prompt
    assert "process_exchange_request -> exchange_delivered_order_items" in system_prompt
