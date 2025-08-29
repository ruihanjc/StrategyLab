from strategy_broker.ib_connection import IBConnection
from ib_insync import Forex


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

        forex = Forex("AUDUSD")

        ib.reqMarketDataType(1)

        bars = ib.reqHistoricalData(
            forex,
            endDateTime="",
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            timeout=20,
        )

        requested_data = ib.reqHistoricalData(forex, "", "100 D", "1 MIN", "TRADES", 1, 1, False, [])

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        ib_conn.disconnect()


if __name__ == "__main__":
    main()
