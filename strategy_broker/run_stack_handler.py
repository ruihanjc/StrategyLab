from strategy_broker.instrument_stack import InstrumentStack
from strategy_broker.contract_stack import ContractStack
from strategy_broker.broker_stack import BrokerStack
from strategy_broker.order import Order, OrderType
from ib_connection import IBConnection


def run_stack_handler():
    """
    Main handler for the order execution pipeline.
    """

    # 1. Initialize the connection and stacks
    ib_conn = IBConnection()
    instrument_stack = InstrumentStack()
    contract_stack = ContractStack()
    broker_stack = BrokerStack(ib_conn)

    # --- This is where you would get signals from your strategies ---
    # For demonstration, we'll create a sample order
    sample_order = Order(instrument="SPY", quantity=10, order_type=OrderType.MARKET)
    instrument_stack.add_order(sample_order)

    print("--- Instrument Stack ---")
    for order in instrument_stack.get_all_orders():
        print(order)

    # 2. Spawn contract orders from instrument stack
    print("\n--- Contract Stack ---")
    for instrument_order in instrument_stack.get_all_orders():
        contract_id = f"{instrument_order.instrument}_CONTRACT"
        contract_order = Order(
            instrument=instrument_order.instrument,
            quantity=instrument_order.quantity,
            order_type=instrument_order.order_type
        )
        contract_stack.add_order(contract_order, contract_id)

    for order in contract_stack.get_all_orders():
        print(order)

    # 3. Generate force rolls (if any)
    # (To be implemented)
    print("\n--- Force Rolls (TBI) ---")

    # 4. Create broker orders
    print("\n--- Broker Stack ---")
    for contract_order in contract_stack.get_all_orders():
        broker_stack.submit_order(contract_order)

    print("\n--- Live Orders at Broker ---")
    for order in broker_stack.get_live_orders():
        print(order)

    # 5. Process fills
    # (To be implemented)
    print("\n--- Process Fills (TBI) ---")

    # 6. Handle completions
    # (To be implemented)
    print("\n--- Handle Completions (TBI) ---")

    # 7. Stack cleanup
    # (To be implemented)
    print("\n--- Stack Cleanup (TBI) ---")


if __name__ == "__main__":
    run_stack_handler()
