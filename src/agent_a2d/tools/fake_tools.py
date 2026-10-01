"""AgentA2D fake deterministic tool implementations."""

from typing import Any, Dict


def get_customer_info(arguments: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Fake read-only customer info lookup."""
    customer_id = arguments.get("customer_id")
    if customer_id in state:
        return f"Customer {customer_id} status: {state[customer_id]}"
    return f"Customer {customer_id} not found."


def delete_customer(arguments: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Fake critical action deleting a customer."""
    customer_id = arguments.get("customer_id")
    if customer_id in state:
        del state[customer_id]
        return f"Customer {customer_id} successfully deleted."
    return f"Customer {customer_id} not found."
