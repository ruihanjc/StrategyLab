from typing import Dict
from .order import Order, OrderStatus
from .ib_connection import IBConnection
from ib_insync import Trade

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
        
        # In a real system, you would create a proper ib_insync Contract object here
        # For now, we'll just simulate the submission
        print(f"Submitting order to broker: {order}")
        
        # Simulate placing an order and getting a trade object
        # In a real scenario, this would be: trade = ib.placeOrder(contract, ib_order)
        trade = Trade()
        trade.orderStatus.status = 'Submitted'
        trade.order.orderId = len(self._live_orders) + 1 # Simulate an order ID

        order.order_id = trade.order.orderId
        order.status = OrderStatus.SUBMITTED
        self._live_orders[order.order_id] = order
        
        # Set up a callback for order status updates
        trade.orderStatusEvent += self.on_order_status

    def on_order_status(self, trade: Trade):
        """Callback for handling order status updates from the broker."""
        order_id = trade.order.orderId
        if order_id in self._live_orders:
            order = self._live_orders[order_id]
            
            new_status = trade.orderStatus.status
            print(f"Order {order_id} status update: {new_status}")
            
            # Update the order status based on the broker's response
            if new_status == 'Filled':
                order.status = OrderStatus.FILLED
            elif new_status in ['Cancelled', 'Inactive']:
                order.status = OrderStatus.CANCELLED
            elif new_status == 'ApiCancelled':
                order.status = OrderStatus.CANCELLED

    def get_live_orders(self):
        return self._live_orders.values()
