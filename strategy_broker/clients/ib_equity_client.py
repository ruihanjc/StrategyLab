from strategy_broker.clients.ib_client import IBClient
import pandas as pd
import datetime


class IBEquityClient(IBClient):

    def __init__(self):
        super().__init__()
        self.barSizeSetting = "1d"  # We are always fetching EOD data

    def get_ib_data(self, ticker: str, service: str, backfill: bool = False):
        """
        Fetches historical EOD data for a given ticker.

        :param ticker: The stock ticker symbol.
        :param backfill: If True, fetches all data from 2020-01-01. 
                         If False (default), fetches the last 5 days of data for daily updates.
        """
        try:
            connectionId = self.ib_conn.get_conid(ticker)
            if not connectionId:
                self.logger.error(f"Could not find connectionId for ticker: {ticker}")
                return pd.DataFrame()

            if backfill:
                # Calculate years to go back to 2020
                start_year = 2020
                current_year = datetime.date.today().year
                years_to_fetch = current_year - start_year + 1
                period = f"{years_to_fetch}Y"
            else:
                # Fetch last few days for daily update
                period = "5d"

            bars_json = self.ib_conn.get_historical_data(
                conid=connectionId,
                period=period,
                bar=self.barSizeSetting
            )

            if not bars_json:
                self.logger.warning(f"No historical data returned for {ticker} with period {period}")
                return pd.DataFrame()

            df = self.process_bars_data(bars_json, ticker, service)

            # If backfilling, we should still ensure the date is on or after 2020-01-01,
            # just in case the API gives us more than we asked for.
            if backfill:
                df = df[df['date'] >= datetime.date(2020, 1, 1)]

            return df

        except Exception as e:
            self.logger.error(f"Error getting IB data for {ticker}: {e}")
            return pd.DataFrame()

    def process_bars_data(self, bars_json, ticker: str, service: str):
        # This method is still valid as it processes the JSON response.
        data = []
        for bar in bars_json:
            bar_date = datetime.datetime.fromtimestamp(bar['t'] / 1000).date()

            data.append({
                'date': bar_date,
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar['v'],
                'ticker': ticker,
                'timestamp': datetime.date.today(),
                'service': service.capitalize()
            })

        df = pd.DataFrame(data)
        return df
