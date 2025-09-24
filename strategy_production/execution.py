import json
import os
import logging
from strategy_broker.stack.instrument_stack import InstrumentStack
from strategy_broker.stack.contract_stack import ContractStack
from strategy_broker.stack.broker_stack import BrokerStack
from strategy_broker.orders.order import Order, OrderType
from strategy_broker.ib_connection import IBConnection

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDERS_FILE_PATH = os.path.join(PROJECT_ROOT, "strategy_production/order_signal/next_orders.json")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def execute_orders():
    """
    Main handler for the order execution pipeline.
    Reads a list of trades from the next_orders.json file and executes them.
    """
    logging.info("Starting order execution pipeline...")

    # --- Load Input Files ---
    try:
        with open(ORDERS_FILE_PATH, 'r') as f:
            trades_to_make = json.load(f)
        logging.info(f"Successfully loaded orders from {ORDERS_FILE_PATH}")

    except FileNotFoundError as e:
        logging.error(f"Error: Input file not found. {e}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error: Could not decode JSON from a file. {e}")
        return

    if not trades_to_make:
        logging.info("No trades to execute.")
        return

    # --- Initialize Connection and Stacks ---
    ib_conn = IBConnection()
    instrument_stack = InstrumentStack()
    contract_stack = ContractStack()
    broker_stack = BrokerStack(ib_conn)

    try:
        ib_conn.connect()
        logging.info("Connected to Interactive Brokers.")

        # --- Create Orders ---
        for trade in trades_to_make:
            action = trade.get("action")
            symbol = trade.get("symbol")
            quantity = trade.get("quantity")

            if not all([action, symbol, quantity]):
                logging.warning(f"Skipping invalid trade object: {trade}")
                continue

            # Determine order direction
            trade_quantity = quantity if action == "BUY" else -quantity

            # For now, we'll use Market orders as a default
            order = Order(instrument=symbol, quantity=trade_quantity, order_type=OrderType.MARKET)
            instrument_stack.add_order(order)
            logging.info(f"Created order for {symbol}: {action} {quantity}")

        # --- Process Orders through Stacks ---
        logging.info("--- Instrument Stack ---")
        for order in instrument_stack.get_all_orders():
            logging.info(order)

        logging.info("\n--- Contract Stack ---")
        for instrument_order in instrument_stack.get_all_orders():
            contract_id = f"{instrument_order.instrument}_CONTRACT"
            contract_order = Order(
                instrument=instrument_order.instrument,
                quantity=instrument_order.quantity,
                order_type=instrument_order.order_type
            )
            contract_stack.add_order(contract_order, contract_id)

        for order in contract_stack.get_all_orders():
            logging.info(order)

        logging.info("\n--- Broker Stack (Submitting Orders) ---")
        for contract_order in contract_stack.get_all_orders():
            broker_stack.submit_order(contract_order)

        logging.info("\n--- Live Orders at Broker ---")
        for order in broker_stack.get_live_orders():
            logging.info(order)

    except Exception as e:
        logging.error(f"An error occurred during order execution: {e}", exc_info=True)

    finally:
        if ib_conn.ib.isConnected():
            ib_conn.disconnect()
            logging.info("Disconnected from Interactive Brokers.")

if __name__ == "__main__":
    execute_orders()
