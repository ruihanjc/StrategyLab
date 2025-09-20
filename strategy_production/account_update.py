import logging
import sys
import os
import json
from datetime import datetime
from strategy_broker.ib_connection import IBConnection

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def account_update(config_arguments):
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        ib = IBConnection()
        ib.set_client_id(0)
        ib.connect()

        # Get account summary and orders
        account_data = ib.get_account_summary()
        account_summary = account_data['account_summary']

        # Get portfolio
        portfolio = ib.ib.portfolio()

        # Get orders
        trades = ib.ib.openTrades()

        # Prepare summary data
        summary_data = {
            'account_number': ib.account_number,
            'last_checked_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'net_liquidation': None,
            'available_funds': None,
            'total_cash_value': None,
            'positions': [],
            'orders': []
        }

        for item in account_summary:
            match item.tag:
                case 'NetLiquidation':
                    summary_data['net_liquidation'] = float(item.value)
                    continue
                case 'AvailableFunds':
                    summary_data['available_funds'] = float(item.value)
                    continue
                case 'TotalCashValue':
                    summary_data['total_cash_value'] = float(item.value)
                    continue
                case _:
                    continue

        if not portfolio:
            logger.info("No positions found in the account.")
        else:
            for p in portfolio:
                summary_data['positions'].append({
                    'symbol': p.contract.symbol,
                    'secType': p.contract.secType,
                    'exchange': p.contract.exchange,
                    'position': p.position,
                    'market_price': p.marketPrice,
                    'market_value': p.marketValue,
                    'average_cost': p.averageCost,
                    'unrealized_pnl': p.unrealizedPNL,
                    'realized_pnl': p.realizedPNL
                })

        # Add trades to summary data
        if not trades:
            logger.info("No open trades found in the account.")
        else:
            for trade in trades:
                summary_data['orders'].append({
                    'symbol': trade.contract.symbol,
                    'secType': trade.contract.secType,
                    'exchange': trade.contract.exchange,
                    'action': trade.order.action,
                    'total_quantity': trade.order.totalQuantity,
                    'order_type': trade.order.orderType,
                    'limit_price': trade.order.lmtPrice if hasattr(trade.order, 'lmtPrice') else None,
                    'stop_price': trade.order.auxPrice if hasattr(trade.order, 'auxPrice') else None,
                    'status': trade.orderStatus.status,
                    'filled': trade.orderStatus.filled,
                    'remaining': trade.orderStatus.remaining
                })

        # Define the output file path
        project_dir = os.path.abspath(__file__ + "/../")
        output_file = os.path.join(project_dir, "account/account_details.json")

        # Load existing data if file exists to preserve additional fields
        existing_data = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(f"Could not read existing {output_file}, creating new file")

        # Merge existing data with new summary data, preserving fields like original_cash
        final_data = {**existing_data, **summary_data}

        # Write updated data to JSON file
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=4)

        logger.info(f"Account summary successfully written to {output_file}")

        return True
    except Exception as error:
        logger.error(f"Application error: {str(error)}", exc_info=True)
        raise
    finally:
        if 'ib' in locals() and ib.ib.isConnected():
            ib.disconnect()


if __name__ == '__main__':
    try:
        success = account_update(sys.argv)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)
