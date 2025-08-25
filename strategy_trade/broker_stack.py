from typing import Dict
from .order import Order, OrderStatus, OrderType
from .ib_connection import IBConnection
from .contract_factory import create_contract
from ib_insync import Trade, Order as IBOrder

class BrokerStack:
    """
    Manages the submission of orders to the broker and tracks their status.
    """
    
    def __init__(self, ib_connection: IBConnection):
        self.ib_conn = ib_connection
        self._live_orders: Dict[int, Order] = {}

    def submit_order(self, order: Order):
        """Submits an order to the broker."""
        ib = self.ib_conn.connect()
        
        # --- Create IB Contract using the factory ---
        contract = create_contract(order.instrument)
        
        # --- Create IB Order ---
        ib_order = IBOrder(
            action="BUY" if order.quantity > 0 else "SELL",
            totalQuantity=abs(order.quantity),
            orderType=order.order_type.value,
            lmtPrice=order.limit_price if order.order_type == OrderType.LIMIT else 0
        )
        
        # --- Place Order ---
        trade = ib.placeOrder(contract, ib_order)
        
        print(f"Placed order for {order.instrument}: {trade.orderStatus.status}")

        order.order_id = trade.order.orderId
        # Safely map status string to enum
        order.status = OrderStatus[trade.orderStatus.status.upper()] if trade.orderStatus.status.upper() in OrderStatus.__members__ else order.status
        self._live_orders[order.order_id] = order
        
        # Set up a callback for order status updates
        trade.orderStatusEvent += self.on_order_status

    def on_order_status(self, trade: Trade):
        """Callback for handling order status updates from the broker."""
        order_id = trade.order.orderId
        if order_id in self._live_orders:
            order = self._live_orders[order_id]
            
            new_status_str = trade.orderStatus.status.upper()
            print(f"Order {order_id} status update: {new_status_str}")
            
            # Update the order status based on the broker's response
            if new_status_str in OrderStatus.__members__:
                order.status = OrderStatus[new_status_str]

    def get_live_orders(self):
        return self._live_orders.values()
