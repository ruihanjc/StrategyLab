from typing import Dict, List
from strategy_broker.orders.order import Order


class ContractStack:
    """
    Manages a collection of orders for specific, tradable contracts.
    """

    def __init__(self):
        # Using contract ID as the key for uniqueness
        self._orders: Dict[str, Order] = {}

    def add_order(self, order: Order, contract_id: str):
        """Adds a contract order to the stack."""
        if contract_id in self._orders:
            print(f"Warning: Replacing existing order for contract {contract_id}")
        self._orders[contract_id] = order

    def get_order(self, contract_id: str) -> Order:
        """Retrieves an order by contract ID."""
        return self._orders.get(contract_id)

    def get_all_orders(self) -> List[Order]:
        """Returns all orders in the stack."""
        return list(self._orders.values())

    def remove_order(self, contract_id: str):
        """Removes an order from the stack."""
        if contract_id in self._orders:
            del self._orders[contract_id]
