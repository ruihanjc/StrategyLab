import json
import os
import logging
import math
import yaml
from collections import defaultdict
from strategy_core.sysutils.engine_utils import create_instruments_from_config, load_price_data
from datetime import datetime

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCOUNT_DETAILS_PATH = os.path.join(PROJECT_ROOT, "strategy_production/account/account_details.json")
TARGET_POSITIONS_PATH = os.path.join(PROJECT_ROOT, "strategy_production/order_signal/target_positions.json")
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "strategy_production/order_signal/next_orders.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "strategy_backtest/config/backtest_config.yaml")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def check_signals_and_order():
    """
    Compares target position weights with current positions to generate a JSON file
    with a list of trades to be executed, using historical data for prices.
    """
    logging.info("Starting trade calculation using historical data...")

    # --- Load Input Files ---
    try:
        with open(ACCOUNT_DETAILS_PATH, 'r') as f:
            account_details = json.load(f)
        logging.info(f"Successfully loaded account details from {ACCOUNT_DETAILS_PATH}")

        with open(TARGET_POSITIONS_PATH, 'r') as f:
            target_positions = json.load(f)
        logging.info(f"Successfully loaded target positions from {TARGET_POSITIONS_PATH}")
        
        with open(CONFIG_PATH, 'r') as file:
            backtest_config = yaml.safe_load(file)
        logging.info(f"Initializing configuration from {CONFIG_PATH}...")

    except FileNotFoundError as e:
        logging.error(f"Error: Input file not found. {e}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error: Could not decode JSON from a file. {e}")
        return

    # --- Extract Current Positions and Capital ---
    current_positions = defaultdict(int)
    try:
        for position in account_details.get("positions", []):
            instrument = position.get("symbol")
            quantity = int(position.get("position", 0))
            if instrument:
                current_positions[instrument] = quantity
        logging.info(f"Processed {len(current_positions)} current positions.")
        
        net_liquidation = account_details.get("net_liquidation")
        if not net_liquidation:
            logging.error("Net liquidation value not found in account details.")
            return
            
    except (KeyError, TypeError) as e:
        logging.error(f"Error processing account details. Check format. {e}")
        return

    # --- Load Historical Prices and Calculate Trades ---
    trades_to_make = []
    try:
        all_instruments = list(set(current_positions.keys()) | set(target_positions.keys()))
        
        # Create instruments from configuration
        instruments = create_instruments_from_config(backtest_config.get("instruments"))

        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        # Load price data for instruments
        price_data = load_price_data(instruments, start_date, end_date)
        
        prices = {}
        for instrument in all_instruments:
            if instrument in price_data:
                # Get the last available closing price
                prices[instrument] = price_data[instrument].iloc[-1]['close']
            else:
                logging.warning(f"Could not retrieve historical price for {instrument}. It will be excluded from trade calculations.")

        # --- Calculate Trades ---
        for instrument in all_instruments:
            if instrument not in prices:
                continue

            target_weight = target_positions.get(instrument, 0)
            current_qty = current_positions.get(instrument, 0)
            
            target_value = net_liquidation * target_weight
            target_qty = math.ceil(target_value / prices[instrument])
            
            trade_qty = target_qty - current_qty

            if trade_qty != 0:
                action = "BUY" if trade_qty > 0 else "SELL"
                quantity = abs(trade_qty)
                price = prices.get(instrument)
                if price:
                    total_money = quantity * price
                    trades_to_make.append({
                        "action": action,
                        "symbol": instrument,
                        "quantity": quantity,
                        "estimated_total_cost": round(total_money, 2)
                    })
                logging.info(f"Trade calculated for {instrument}: {action} {quantity}")

    except Exception as e:
        logging.error(f"An error occurred during trade calculation: {e}", exc_info=True)

    # --- Write Output File ---
    if not trades_to_make:
        logging.info("No trades required. Target positions match current positions.")
    else:
        try:
            with open(OUTPUT_FILE_PATH, 'w') as f:
                json.dump(trades_to_make, f, indent=4)
            logging.info(f"Successfully wrote {len(trades_to_make)} trades to {OUTPUT_FILE_PATH}")
        except IOError as e:
            logging.error(f"Error writing output file. {e}")
            return

    logging.info("Trade calculation finished.")

if __name__ == "__main__":
    check_signals_and_order()