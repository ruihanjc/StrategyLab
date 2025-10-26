from datetime import datetime
from threading import Lock
import pykx as kx
import time
import yfinance as yf

class QTickerFeed:
    def __init__(self, ticker):
        with kx.SyncQConnection('localhost', 5010) as self.q:
            self.latest_data = {}
            self.data_lock = Lock()
            self.fetch_stock_data(ticker)
            self.q.close()


    def fetch_stock_data(self,symbols):
        global latest_data

        while True:
            try:
                # Fetch all tickers in one call (more efficient)
                ticker = yf.Tickers(symbols)

                new_data = None
                timestamp = datetime.now().isoformat()

                try:
                        # Get fast quote data (lighter than .info)
                    fast_info = ticker.tickers[symbols].fast_info

                    new_data = kx.toq([symbols, fast_info.last_price, "B"])


                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")

                # Thread-safe update
                with self.data_lock:
                    latest_data = new_data

                print(latest_data)
                self.q.upd("trade", latest_data)

                print(f"Updated {len(new_data)} tickers at {timestamp}")

            except Exception as e:
                print(f"Error in fetch loop: {e}")

            time.sleep(5)



QTickerFeed("AAPL")