import sys
import os
from strategy_brokers.ib_connection import IBConnection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    ib_conn = IBConnection(config_path='private_config.yaml')
    try:
        ib = ib_conn.connect()
        account_number = ib_conn.config['broker_account']

        # Fetching account summary
        account_summary = ib.accountSummary(account_number)

        print("Account Summary:")
        for account in account_summary:
            print(f"{account.tag}: {account.value} {account.currency}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        ib_conn.disconnect()


if __name__ == "__main__":
    main()
