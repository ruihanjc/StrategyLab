import json
import os
import math

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


    # Define path to target positions file
    project_dir = os.path.abspath(__file__ + "/../../")
    positions_file = os.path.join(project_dir, "strategy_production/order_signal/target_positions.json")

    if not os.path.exists(positions_file):
        print(f"Target positions file not found at {positions_file}. Exiting.")
        return

    # Load target fractional positions
    with open(positions_file, 'r') as f:
        target_positions_fractional = json.load(f)
    
    print("--- Target Fractional Positions ---")
    print(target_positions_fractional)

    # --- MOCK DATA: Replace with live data from your broker ---
    # This data would be fetched live before calculating trades.
    
    # 1. Current Portfolio Capital (marked-to-market)
    current_capital = 900000.0  # Using a value like 900k to get reasonable share counts

    # 2. Current prices for instruments
    current_prices = {
        "SPY": 450.50,
        "AAPL": 175.20,
        "GOOG": 135.80
        # Add other instruments as needed
    }

    # 3. Current share counts held in the portfolio
    current_shares_held = {
        "SPY": 100,
        "AAPL": -50, # A short position of 50 shares
        "GOOG": 0
    }
    
    print("\n--- MOCK Broker Data ---")
    print(f"Current Capital: ${current_capital:,.2f}")
    print(f"Current Prices: {current_prices}")
    print(f"Current Shares Held: {current_shares_held}")
    # --- END MOCK DATA ---

    # --- Calculate and Generate Orders ---
    print("\n--- Generating Orders ---")
    for instrument, target_frac_pos in target_positions_fractional.items():
        if instrument not in current_prices:
            print(f"Warning: No price data for {instrument}, cannot calculate order. Skipping.")
            continue

        # Get current state from (mock) broker data
        price = current_prices[instrument]
        current_shares = current_shares_held.get(instrument, 0)

        # Calculate target number of shares
        target_value = current_capital * target_frac_pos
        target_shares = math.floor(target_value / price) # Use floor to avoid over-leveraging

        # Calculate the difference needed to trade
        delta_shares = target_shares - current_shares

        if delta_shares != 0:
            # Using positive quantity for BUY, negative for SELL
            order = Order(instrument=instrument, quantity=delta_shares, order_type=OrderType.MARKET)
            instrument_stack.add_order(order)
            print(f"Created order for {instrument}: Target {target_shares} shares, Current {current_shares} shares -> TRADE {delta_shares} shares.")

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
