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
                timestamp = datetime.now()

                try:
                        # Get fast quote data (lighter than .info)
                    ticker_info = ticker.tickers[symbols].info

                    new_data = kx.toq([symbols, ticker_info["currentPrice"], ticker_info["bid"],
                                       ticker_info["ask"], ticker_info["bidSize"], ticker_info["askSize"],ticker_info["open"],
                                       ticker_info["dayLow"], ticker_info["dayHigh"], ticker_info["volume"]])


                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")

                # Thread-safe update
                with self.data_lock:
                    latest_data = new_data

                self.q.upd("tickerData", latest_data)

            except Exception as e:
                print(f"Error in fetch loop: {e}")

            time.sleep(600)


    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'q'):
            self.q.close()

QTickerFeed("AAPL")