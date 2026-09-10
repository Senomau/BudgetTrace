"""Lightweight tau2 retail tool execution for unscored live episodes.

The upstream tau2 package is not imported here: its current source requires a
newer interpreter than BudgetTrace's Python 3.9 floor. This module instead
loads the pinned retail ``db.json`` and executes the small, auditable subset of
retail tool semantics needed by BudgetTrace's live episode loop.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .public_replay import Tau3ReferenceTask
from .security import ToolRegistry

RETAIL_TOOL_ENVIRONMENT = "tau2-retail-json-v1"

READ_TOOL_LEVEL = "L0"
WRITE_TOOL_LEVEL = "L1"

OFFICIAL_RETAIL_TOOL_LEVELS: Dict[str, str] = {
    "calculate": READ_TOOL_LEVEL,
    "cancel_pending_order": WRITE_TOOL_LEVEL,
    "exchange_delivered_order_items": WRITE_TOOL_LEVEL,
    "find_user_id_by_email": READ_TOOL_LEVEL,
    "find_user_id_by_name_zip": READ_TOOL_LEVEL,
    "get_item_details": READ_TOOL_LEVEL,
    "get_order_details": READ_TOOL_LEVEL,
    "get_product_details": READ_TOOL_LEVEL,
    "get_user_details": READ_TOOL_LEVEL,
    "list_all_product_types": READ_TOOL_LEVEL,
    "modify_pending_order_address": WRITE_TOOL_LEVEL,
    "modify_pending_order_items": WRITE_TOOL_LEVEL,
    "modify_pending_order_payment": WRITE_TOOL_LEVEL,
    "modify_user_address": WRITE_TOOL_LEVEL,
    "return_delivered_order_items": WRITE_TOOL_LEVEL,
    "transfer_to_human_agents": READ_TOOL_LEVEL,
}

COMPAT_RETAIL_TOOL_LEVELS: Dict[str, str] = {
    "check_inventory": READ_TOOL_LEVEL,
    "get_exchange_options": READ_TOOL_LEVEL,
    "get_order": READ_TOOL_LEVEL,
    "process_exchange_request": WRITE_TOOL_LEVEL,
    "search_products": READ_TOOL_LEVEL,
    "send_message": READ_TOOL_LEVEL,
}

TOOL_HINTS: Dict[str, str] = {
    "find_user_id_by_name_zip": "args: first_name, last_name, zip",
    "find_user_id_by_email": "args: email",
    "get_user_details": "args: user_id",
    "get_order_details": "args: order_id",
    "get_product_details": "args: product_id",
    "get_item_details": "args: item_id",
    "list_all_product_types": "args: {}",
    "exchange_delivered_order_items": (
        "args: order_id, item_ids, new_item_ids, payment_method_id"
    ),
    "return_delivered_order_items": "args: order_id, item_ids, payment_method_id",
    "modify_pending_order_items": (
        "args: order_id, item_ids, new_item_ids, payment_method_id"
    ),
    "modify_pending_order_address": (
        "args: order_id, address1, address2, city, state, country, zip"
    ),
    "modify_pending_order_payment": "args: order_id, payment_method_id",
    "modify_user_address": "args: user_id, address1, address2, city, state, country, zip",
    "cancel_pending_order": "args: order_id, reason",
    "calculate": "args: expression",
    "transfer_to_human_agents": "args: summary",
}


_TOOL_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "calculate": {
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
    "cancel_pending_order": {
        "properties": {
            "order_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["order_id", "reason"],
    },
    "exchange_delivered_order_items": {
        "properties": {
            "order_id": {"type": "string"},
            "item_ids": {"type": "array", "items": {"type": "string"}},
            "new_item_ids": {"type": "array", "items": {"type": "string"}},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    },
    "find_user_id_by_email": {
        "properties": {"email": {"type": "string"}},
        "required": ["email"],
    },
    "find_user_id_by_name_zip": {
        "properties": {
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "zip": {"type": "string"},
        },
        "required": ["first_name", "last_name", "zip"],
    },
    "get_item_details": {
        "properties": {"item_id": {"type": "string"}},
        "required": ["item_id"],
    },
    "get_order_details": {
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
    "get_product_details": {
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
    },
    "get_user_details": {
        "properties": {"user_id": {"type": "string"}},
        "required": ["user_id"],
    },
    "list_all_product_types": {"properties": {}, "required": []},
    "modify_pending_order_address": {
        "properties": {
            "order_id": {"type": "string"},
            "address1": {"type": "string"},
            "address2": {"type": "string"},
            "city": {"type": "string"},
            "state": {"type": "string"},
            "country": {"type": "string"},
            "zip": {"type": "string"},
        },
        "required": ["order_id", "address1", "address2", "city", "state", "country", "zip"],
    },
    "modify_pending_order_items": {
        "properties": {
            "order_id": {"type": "string"},
            "item_ids": {"type": "array", "items": {"type": "string"}},
            "new_item_ids": {"type": "array", "items": {"type": "string"}},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    },
    "modify_pending_order_payment": {
        "properties": {
            "order_id": {"type": "string"},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "payment_method_id"],
    },
    "modify_user_address": {
        "properties": {
            "user_id": {"type": "string"},
            "address1": {"type": "string"},
            "address2": {"type": "string"},
            "city": {"type": "string"},
            "state": {"type": "string"},
            "country": {"type": "string"},
            "zip": {"type": "string"},
        },
        "required": ["user_id", "address1", "address2", "city", "state", "country", "zip"],
    },
    "return_delivered_order_items": {
        "properties": {
            "order_id": {"type": "string"},
            "item_ids": {"type": "array", "items": {"type": "string"}},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "item_ids", "payment_method_id"],
    },
    "transfer_to_human_agents": {
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
}


def retail_tool_schemas() -> List[Dict[str, Any]]:
    """Return OpenAI-compatible schemas for official retail tools only."""
    schemas = []
    for name in sorted(OFFICIAL_RETAIL_TOOL_LEVELS):
        parameters = dict(_TOOL_PARAMETERS.get(name, {"properties": {}, "required": []}))
        parameters["type"] = "object"
        parameters["additionalProperties"] = False
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": TOOL_HINTS.get(name, f"Retail tool: {name}."),
                    "parameters": parameters,
                },
            }
        )
    return schemas


class RetailAdapterError(ValueError):
    """Raised when the local tau2 retail source cannot be used safely."""


@dataclass(frozen=True)
class RegisteredRetailTools:
    allowed_tools: List[str]
    tool_levels: Dict[str, str]


def retail_data_dir(source_root: Path, task: Tau3ReferenceTask) -> Path:
    return source_root / Path(task.source_path).parent


def has_retail_data(source_root: Path, task: Tau3ReferenceTask) -> bool:
    return (retail_data_dir(source_root, task) / "db.json").is_file()


class RetailToolEnvironment:
    def __init__(self, db: Mapping[str, Any]) -> None:
        self.db: Dict[str, Any] = copy.deepcopy(dict(db))

    @classmethod
    def from_source(cls, source_root: Path, task: Tau3ReferenceTask) -> "RetailToolEnvironment":
        db_path = retail_data_dir(source_root, task) / "db.json"
        if not db_path.is_file():
            raise RetailAdapterError(f"missing retail database: {db_path}")
        return cls(json.loads(db_path.read_text(encoding="utf-8")))

    @property
    def allowed_tools(self) -> List[str]:
        return sorted(set(OFFICIAL_RETAIL_TOOL_LEVELS) | set(COMPAT_RETAIL_TOOL_LEVELS))

    @property
    def tool_levels(self) -> Dict[str, str]:
        levels = dict(OFFICIAL_RETAIL_TOOL_LEVELS)
        levels.update(COMPAT_RETAIL_TOOL_LEVELS)
        return {name: levels[name] for name in self.allowed_tools}

    def tool_prompt(self) -> str:
        lines = [
            "STRICT TOOL PROTOCOL: use only the exact official retail tool names listed below.",
            "Tool names are case-sensitive; do not invent, pluralize, or paraphrase a name.",
            'For a tool call emit exactly {"kind":"tool_call","tool":"<official name>","args":{...}}.',
            "Never emit ask_user, get_orders, or any other name that is not listed as an official tool.",
            "The compatibility aliases below are runtime-only replay fallbacks; do not emit them in a model action.",
            "Official retail tools:",
        ]
        for name in sorted(TOOL_HINTS):
            lines.append(f"- {name}: {TOOL_HINTS[name]}")
        lines.extend(
            [
                "Forbidden model-emitted aliases and invented names:",
                "- get_order -> get_order_details (do not emit the alias)",
                "- check_inventory -> get_item_details or get_product_details (do not emit the alias)",
                "- search_products -> product search over local catalog (do not emit the alias)",
                "- get_exchange_options -> available variants for an order item/product (do not emit the alias)",
                "- process_exchange_request -> exchange_delivered_order_items (do not emit the alias)",
                "- send_message -> customer-facing text belongs in the final answer, not a tool call (do not emit the alias)",
            ]
        )
        return "\n".join(lines)

    def invoke(self, name: str, args: Mapping[str, Any]) -> Any:
        method = getattr(self, f"_tool_{name}", None)
        if method is None:
            raise ValueError(f"Tool '{name}' not found.")
        return method(**dict(args))

    def _tool_calculate(self, expression: str) -> str:
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    def _tool_find_user_id_by_name_zip(self, first_name: str, last_name: str, zip: str) -> str:
        for user_id, user in self._users().items():
            name = user.get("name", {})
            address = user.get("address", {})
            if (
                str(name.get("first_name", "")).lower() == first_name.lower()
                and str(name.get("last_name", "")).lower() == last_name.lower()
                and str(address.get("zip", "")) == str(zip)
            ):
                return user_id
        raise ValueError("User not found")

    def _tool_find_user_id_by_email(self, email: str) -> str:
        for user_id, user in self._users().items():
            if str(user.get("email", "")).lower() == email.lower():
                return user_id
        raise ValueError("User not found")

    def _tool_get_order_details(self, order_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self._order(order_id))

    def _tool_get_order(self, order_id: str) -> Dict[str, Any]:
        return self._tool_get_order_details(order_id)

    def _tool_get_product_details(self, product_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self._product(product_id))

    def _tool_get_item_details(self, item_id: str) -> Dict[str, Any]:
        product, variant = self._product_and_variant_for_item(item_id)
        result = copy.deepcopy(variant)
        result["product_id"] = product.get("product_id")
        result["product_name"] = product.get("name")
        return result

    def _tool_get_user_details(self, user_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self._user(user_id))

    def _tool_list_all_product_types(self) -> str:
        products = {
            str(product.get("name")): str(product.get("product_id"))
            for product in self._products().values()
        }
        return json.dumps(products, sort_keys=True)

    def _tool_search_products(self, query: str = "", product_type: str = "", name: str = "") -> List[Dict[str, Any]]:
        needle = (query or product_type or name).strip().lower()
        matches = []
        for product in self._products().values():
            product_name = str(product.get("name", ""))
            if not needle or needle in product_name.lower():
                matches.append(
                    {
                        "product_id": product.get("product_id"),
                        "name": product_name,
                        "variant_count": len(product.get("variants", {})),
                    }
                )
        return matches

    def _tool_check_inventory(
        self,
        item_id: str = "",
        product_id: str = "",
        query: str = "",
    ) -> Any:
        if item_id:
            return self._tool_get_item_details(item_id)
        if product_id:
            return self._tool_get_product_details(product_id)
        return self._tool_search_products(query=query)

    def _tool_get_exchange_options(
        self,
        order_id: str = "",
        item_id: str = "",
        product_id: str = "",
        desired_options: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not product_id and item_id:
            product, _ = self._product_and_variant_for_item(item_id)
            product_id = str(product.get("product_id", ""))
        if not product_id and order_id:
            order = self._order(order_id)
            if order.get("items"):
                product_id = str(order["items"][0].get("product_id", ""))
        product = self._product(product_id)
        wanted = {str(k): str(v).lower() for k, v in (desired_options or {}).items()}
        options = []
        for variant in product.get("variants", {}).values():
            if not variant.get("available", False):
                continue
            variant_options = variant.get("options", {})
            if any(str(variant_options.get(key, "")).lower() != value for key, value in wanted.items()):
                continue
            options.append(copy.deepcopy(variant))
        return options

    def _tool_exchange_delivered_order_items(
        self,
        order_id: str,
        item_ids: List[str],
        new_item_ids: List[str],
        payment_method_id: str,
    ) -> Dict[str, Any]:
        order = self._order(order_id)
        if order.get("status") != "delivered":
            raise ValueError("Non-delivered order cannot be exchanged")
        if len(item_ids) != len(new_item_ids):
            raise ValueError("The number of items to be exchanged should match.")
        order_item_ids = [str(item.get("item_id")) for item in order.get("items", [])]
        for item_id in item_ids:
            if item_ids.count(item_id) > order_item_ids.count(item_id):
                raise ValueError(f"Number of {item_id} not found.")

        diff_price = 0.0
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = self._order_item(order, item_id)
            variant = self._variant(str(item.get("product_id")), new_item_id)
            if not variant.get("available", False):
                raise ValueError(f"New item {new_item_id} not found or available")
            diff_price += float(variant.get("price", 0.0)) - float(item.get("price", 0.0))
        diff_price = round(diff_price, 2)

        payment_method = self._payment_method(str(order.get("user_id")), payment_method_id)
        if payment_method.get("source") == "gift_card" and float(payment_method.get("balance", 0.0)) < diff_price:
            raise ValueError("Insufficient gift card balance to pay for the price difference")

        order["status"] = "exchange requested"
        order["exchange_items"] = sorted(item_ids)
        order["exchange_new_items"] = sorted(new_item_ids)
        order["exchange_payment_method_id"] = payment_method_id
        order["exchange_price_difference"] = diff_price
        return copy.deepcopy(order)

    def _tool_process_exchange_request(self, **kwargs: Any) -> Dict[str, Any]:
        return self._tool_exchange_delivered_order_items(**kwargs)

    def _tool_return_delivered_order_items(
        self,
        order_id: str,
        item_ids: List[str],
        payment_method_id: str,
    ) -> Dict[str, Any]:
        order = self._order(order_id)
        if order.get("status") != "delivered":
            raise ValueError("Non-delivered order cannot be returned")
        order_item_ids = [str(item.get("item_id")) for item in order.get("items", [])]
        for item_id in item_ids:
            if item_ids.count(item_id) > order_item_ids.count(item_id):
                raise ValueError("Some item not found")
        self._payment_method(str(order.get("user_id")), payment_method_id)
        order["status"] = "return requested"
        order["return_items"] = sorted(item_ids)
        order["return_payment_method_id"] = payment_method_id
        return copy.deepcopy(order)

    def _tool_cancel_pending_order(self, order_id: str, reason: str) -> Dict[str, Any]:
        order = self._order(order_id)
        if order.get("status") != "pending":
            raise ValueError("Non-pending order cannot be cancelled")
        if reason not in {"no longer needed", "ordered by mistake"}:
            raise ValueError("Invalid reason")
        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        refunds = []
        for payment in order.get("payment_history", []):
            refunds.append(
                {
                    "transaction_type": "refund",
                    "amount": payment.get("amount"),
                    "payment_method_id": payment.get("payment_method_id"),
                }
            )
        order.setdefault("payment_history", []).extend(refunds)
        return copy.deepcopy(order)

    def _tool_modify_pending_order_address(self, order_id: str, **address: str) -> Dict[str, Any]:
        order = self._order(order_id)
        if "pending" not in str(order.get("status", "")):
            raise ValueError("Non-pending order cannot be modified")
        order["address"] = dict(address)
        return copy.deepcopy(order)

    def _tool_modify_user_address(self, user_id: str, **address: str) -> Dict[str, Any]:
        user = self._user(user_id)
        user["address"] = dict(address)
        return copy.deepcopy(user)

    def _tool_modify_pending_order_items(
        self,
        order_id: str,
        item_ids: List[str],
        new_item_ids: List[str],
        payment_method_id: str,
    ) -> Dict[str, Any]:
        order = self._order(order_id)
        if order.get("status") != "pending":
            raise ValueError("Non-pending order cannot be modified")
        if len(item_ids) != len(new_item_ids):
            raise ValueError("The number of items to be exchanged should match")
        payment_method = self._payment_method(str(order.get("user_id")), payment_method_id)
        diff_price = 0.0
        replacements = []
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = self._order_item(order, item_id)
            if item_id == new_item_id:
                raise ValueError("The new item id should be different from the old item id")
            variant = self._variant(str(item.get("product_id")), new_item_id)
            if not variant.get("available", False):
                raise ValueError(f"New item {new_item_id} not found or available")
            diff_price += float(variant.get("price", 0.0)) - float(item.get("price", 0.0))
            replacements.append((item, variant))
        if payment_method.get("source") == "gift_card" and float(payment_method.get("balance", 0.0)) < diff_price:
            raise ValueError("Insufficient gift card balance to pay for the new item")
        for item, variant in replacements:
            item["item_id"] = variant.get("item_id")
            item["price"] = variant.get("price")
            item["options"] = copy.deepcopy(variant.get("options", {}))
        order["status"] = "pending (item modified)"
        return copy.deepcopy(order)

    def _tool_modify_pending_order_payment(self, order_id: str, payment_method_id: str) -> Dict[str, Any]:
        order = self._order(order_id)
        if "pending" not in str(order.get("status", "")):
            raise ValueError("Non-pending order cannot be modified")
        self._payment_method(str(order.get("user_id")), payment_method_id)
        history = order.setdefault("payment_history", [])
        if len(history) != 1 or history[0].get("transaction_type") != "payment":
            raise ValueError("There should be exactly one payment for a pending order")
        if history[0].get("payment_method_id") == payment_method_id:
            raise ValueError("The new payment method should be different from the current one")
        amount = history[0].get("amount")
        history.extend(
            [
                {"transaction_type": "payment", "amount": amount, "payment_method_id": payment_method_id},
                {
                    "transaction_type": "refund",
                    "amount": amount,
                    "payment_method_id": history[0].get("payment_method_id"),
                },
            ]
        )
        return copy.deepcopy(order)

    def _tool_transfer_to_human_agents(self, summary: str) -> str:
        return "Transfer successful"

    def _tool_send_message(self, message: str = "", content: str = "", **kwargs: Any) -> Dict[str, Any]:
        return {"status": "message_recorded", "message": message or content, "metadata": kwargs}

    def _products(self) -> Dict[str, Any]:
        return self._section("products")

    def _users(self) -> Dict[str, Any]:
        return self._section("users")

    def _orders(self) -> Dict[str, Any]:
        return self._section("orders")

    def _section(self, key: str) -> Dict[str, Any]:
        section = self.db.get(key)
        if not isinstance(section, dict):
            raise RetailAdapterError(f"retail db missing object section: {key}")
        return section

    def _order(self, order_id: str) -> Dict[str, Any]:
        try:
            order = self._orders()[order_id]
        except KeyError as exc:
            raise ValueError("Order not found") from exc
        if not isinstance(order, dict):
            raise RetailAdapterError("order record is not an object")
        return order

    def _user(self, user_id: str) -> Dict[str, Any]:
        try:
            user = self._users()[user_id]
        except KeyError as exc:
            raise ValueError("User not found") from exc
        if not isinstance(user, dict):
            raise RetailAdapterError("user record is not an object")
        return user

    def _product(self, product_id: str) -> Dict[str, Any]:
        try:
            product = self._products()[product_id]
        except KeyError as exc:
            raise ValueError("Product not found") from exc
        if not isinstance(product, dict):
            raise RetailAdapterError("product record is not an object")
        return product

    def _variant(self, product_id: str, item_id: str) -> Dict[str, Any]:
        variants = self._product(product_id).get("variants", {})
        if not isinstance(variants, dict) or item_id not in variants:
            raise ValueError("Variant not found")
        variant = variants[item_id]
        if not isinstance(variant, dict):
            raise RetailAdapterError("variant record is not an object")
        return variant

    def _product_and_variant_for_item(self, item_id: str) -> tuple:
        for product in self._products().values():
            variants = product.get("variants", {}) if isinstance(product, dict) else {}
            if isinstance(variants, dict) and item_id in variants:
                return product, variants[item_id]
        raise ValueError("Item not found")

    @staticmethod
    def _order_item(order: Mapping[str, Any], item_id: str) -> Dict[str, Any]:
        for item in order.get("items", []):
            if isinstance(item, dict) and item.get("item_id") == item_id:
                return item
        raise ValueError(f"Item {item_id} not found")

    def _payment_method(self, user_id: str, payment_method_id: str) -> Dict[str, Any]:
        payment_methods = self._user(user_id).get("payment_methods", {})
        if not isinstance(payment_methods, dict) or payment_method_id not in payment_methods:
            raise ValueError("Payment method not found")
        payment_method = payment_methods[payment_method_id]
        if not isinstance(payment_method, dict):
            raise RetailAdapterError("payment method record is not an object")
        return payment_method


def register_retail_tools(
    registry: ToolRegistry,
    environment: RetailToolEnvironment,
) -> RegisteredRetailTools:
    for name in environment.allowed_tools:
        registry.register(name, environment.tool_levels[name], _handler(environment, name))
    return RegisteredRetailTools(
        allowed_tools=environment.allowed_tools,
        tool_levels=environment.tool_levels,
    )


def _handler(environment: RetailToolEnvironment, name: str) -> Callable[[Mapping[str, Any]], Any]:
    def invoke(args: Mapping[str, Any]) -> Any:
        return environment.invoke(name, args)

    return invoke


def official_tools_used(tasks: Iterable[Tau3ReferenceTask]) -> List[str]:
    names = []
    for task in tasks:
        for action in task.actions:
            name = str(action.get("name", ""))
            if name and name not in names:
                names.append(name)
    return names
