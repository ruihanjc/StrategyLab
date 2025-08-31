from strategy_broker.ib_connection import IBConnection
from ib_insync import *

def main():
    ib_conn = IBConnection()
    try:
        ib = ib_conn.connect("IBKR_DUMMY_ACCOUNT")
        account_number = ib_conn.config["IBKR_DUMMY_ACCOUNT"]['broker_account']

        # Fetching account summary
        account_summary = ib.accountSummary(account_number)

        print("Account Summary:")
        for account in account_summary:
            print(f"{account.tag}: {account.value} {account.currency}")

        forex = Forex("AUDUSD")
        stock = Stock("TSLA", exchange="SMART", currency="USD")

        ib.reqMarketDataType(1)

        bars = ib.reqHistoricalData(
            stock,
            endDateTime="",
            durationStr="4 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            timeout=20,
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        ib_conn.disconnect()


if __name__ == "__main__":
    main()
