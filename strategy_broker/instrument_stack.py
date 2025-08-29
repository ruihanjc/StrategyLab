from typing import Dict, List
from order import Order


class InstrumentStack:
    """
    Manages a collection of strategy-level virtual orders (instrument stack).
    """

    def __init__(self):
        self._orders: Dict[str, Order] = {}

    def add_order(self, order: Order):
        """Adds an order to the stack."""
        if order.instrument in self._orders:
            # Potentially handle order replacement logic here in the future
            print(f"Warning: Replacing existing order for {order.instrument}")
        self._orders[order.instrument] = order

    def get_order(self, instrument: str) -> Order:
        """Retrieves an order by instrument."""
        return self._orders.get(instrument)

    def get_all_orders(self) -> List[Order]:
        """Returns all orders in the stack."""
        return list(self._orders.values())

    def remove_order(self, instrument: str):
        """Removes an order from the stack."""
        if instrument in self._orders:
            del self._orders[instrument]
