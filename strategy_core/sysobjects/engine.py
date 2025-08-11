from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

from .backtest_results import BacktestResults
from .forecasts import Forecast


class TradingEngine(ABC):
    """
    Abstract base class for trading engines
    Handles common functionality between backtesting and live trading
    """

    def __init__(self, portfolio, strategy, data_handler=None):
        self.portfolio = portfolio
        self.strategy = strategy
        self.data_handler = data_handler
        self.current_positions = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def run(self, *args, **kwargs):
        """Main execution method - implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_current_prices(self, timestamp=None):
        """Get current market prices - implemented by subclasses"""
        pass
    
    def generate_signals(self, current_date: datetime, price_data: Dict) -> Dict[str, Forecast]:
        """Generate trading signals using the strategy"""
        return self.strategy.generate_signals(current_date, price_data)
    
    def calculate_positions(self, forecasts: Dict[str, Forecast], prices: Dict) -> Dict:
        """Calculate position sizes using the portfolio"""
        return self.portfolio.calculate_position_sizes(forecasts, prices)
    
    def update_positions(self, new_positions: Dict):
        """Update current positions"""
        self.current_positions = new_positions
        self.logger.info(f"Updated positions for {len(new_positions)} instruments")


class BacktestEngine(TradingEngine):
    """
    Backtesting implementation of TradingEngine
    Runs strategy against historical data
    """
    
    def __init__(self, portfolio, strategy, data_handler, start_date, end_date):
        super().__init__(portfolio, strategy, data_handler)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.results = None
        
    def run(self) -> BacktestResults:
        """
        Run the backtest over the specified date range - CTA style
        
        Returns:
        --------
        BacktestResults
            Complete backtest results and performance metrics
        """
        self.logger.info(f"Starting CTA-style backtest from {self.start_date} to {self.end_date}")
        
        # Get price data as DataFrame
        price_df = self._prepare_price_dataframe()
        if price_df.empty:
            raise ValueError("No price data available for backtest")
        
        self.logger.info(f"Backtesting over {len(price_df)} trading days")
        
        # Initialize position and forecast columns for each instrument
        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            price_df[f'pos_{ticker}'] = 0.0
            price_df[f'forecast_{ticker}'] = 0.0
        
        # Strategy execution loop - CTA style
        loadBars = 70  # Minimum bars needed for strategy
        
        for i, (current_date, row) in enumerate(price_df.iterrows()):
            if i < loadBars:
                continue
                
            try:
                # Get historical data up to current point
                historical_data = price_df.loc[:current_date].copy()
                
                # Get current prices for all instruments
                current_prices = self._extract_current_prices(historical_data, current_date)
                
                # Generate signals using strategy for all instruments
                forecasts = self.generate_signals(current_date, current_prices)
                
                # Process forecasts and positions for each instrument
                for instrument in self.portfolio.instruments:
                    ticker = instrument.ticker
                    
                    # Get forecast for this instrument
                    forecast_value = 0.0
                    if ticker in forecasts:
                        forecast_obj = forecasts[ticker]
                        if hasattr(forecast_obj, 'iloc') and len(forecast_obj) > 0:
                            forecast_value = forecast_obj.iloc[-1]
                        elif isinstance(forecast_obj, (int, float)):
                            forecast_value = forecast_obj
                    
                    # Store forecast
                    price_df.loc[current_date, f'forecast_{ticker}'] = forecast_value
                    
                    # Convert forecast to position using portfolio logic
                    if forecast_value != 0:
                        position = self._forecast_to_position(forecast_value, current_prices, ticker)
                        price_df.loc[current_date, f'pos_{ticker}'] = position
                
                # Forward fill positions for all instruments
                for instrument in self.portfolio.instruments:
                    ticker = instrument.ticker
                    price_df.loc[:current_date, f'pos_{ticker}'] = price_df.loc[:current_date, f'pos_{ticker}'].ffill()

                    
            except Exception as e:
                self.logger.error(f"Error processing {current_date}: {str(e)}")
                continue
        
        # Fill any remaining NaN positions for all instruments
        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            price_df[f'pos_{ticker}'] = price_df[f'pos_{ticker}'].fillna(0)
        
        # Calculate performance using CTA-style calculation
        results_df, stats = self._calculate_cta_results(price_df)
        
        # Create BacktestResults object with the calculated data
        self.results = BacktestResults(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.portfolio.initial_capital
        )
        
        # Populate results with calculated data
        self._populate_backtest_results(results_df, stats)
        
        # Store the price DataFrame for plotting
        self.results.price_data = price_df
        
        self.logger.info("CTA-style backtest completed successfully")
        return self.results
    
    def get_current_prices(self, timestamp: datetime) -> Dict:
        """Get historical prices for the given timestamp"""
        if not self.data_handler:
            return {}
            
        try:
            prices = {}
            for instrument in self.portfolio.instruments:
                price_data = self.data_handler.get(instrument.ticker)
                if price_data is not None and timestamp in price_data.index:
                    if hasattr(price_data, 'close'):
                        prices[instrument.ticker] = price_data.loc[timestamp, 'close']
                    else:
                        prices[instrument.ticker] = price_data.loc[timestamp]
            
            return prices
        except Exception as e:
            self.logger.error(f"Error getting prices for {timestamp}: {str(e)}")
            return {}
    
    def _get_available_dates(self) -> list:
        """Get all available trading dates from the data"""
        all_dates = set()
        
        for instrument in self.portfolio.instruments:
            if self.data_handler and instrument.ticker in self.data_handler:
                price_data = self.data_handler[instrument.ticker]
                if hasattr(price_data, 'index'):
                    all_dates.update(price_data.index)
        
        return sorted(list(all_dates))
    
    def _prepare_price_dataframe(self) -> pd.DataFrame:
        """Prepare price data for multiple instruments"""
        if not self.data_handler:
            return pd.DataFrame()
        
        # For multiple instruments, we need to create a combined approach
        # We'll use the first instrument's dates as the master calendar
        # and track positions/forecasts per instrument separately
        
        primary_instrument = list(self.portfolio.instruments)[0]
        price_data = self.data_handler.get(primary_instrument.ticker)
        
        if price_data is None:
            return pd.DataFrame()
        
        # Extract OHLC data for primary instrument
        if hasattr(price_data, 'data'):
            df = price_data.data.copy()
        elif hasattr(price_data, 'close'):
            df = pd.DataFrame({
                'open': price_data.open,
                'high': price_data.high, 
                'low': price_data.low,
                'close': price_data.close,
                'volume': price_data.volume if hasattr(price_data, 'volume') else 0
            })
        else:
            df = price_data.copy()
        
        # Add columns for each instrument's close prices
        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            if ticker != primary_instrument.ticker:  # Skip primary as it's already included
                inst_data = self.data_handler.get(ticker)
                if inst_data is not None:
                    if hasattr(inst_data, 'close'):
                        close_series = inst_data.close
                    elif hasattr(inst_data, 'data') and 'close' in inst_data.data.columns:
                        close_series = inst_data.data['close']
                    else:
                        close_series = inst_data
                    
                    # Align dates and add to main DataFrame
                    df[f'{ticker}_close'] = close_series
        
        # Clean data - remove invalid prices
        df = df[(df['close'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
        df = df.dropna(subset=['close'])  # Only require primary instrument to have valid data
        
        # Filter by date range
        mask = (df.index >= self.start_date) & (df.index <= self.end_date)
        df = df[mask]
        
        return df
    
    def _extract_current_prices(self, historical_data: pd.DataFrame, current_date) -> Dict:
        """Extract current prices for all instruments"""
        if current_date not in historical_data.index:
            return {}
        
        current_prices = {}
        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            
            # Primary instrument uses 'close' column
            if ticker == list(self.portfolio.instruments)[0].ticker:
                if 'close' in historical_data.columns:
                    current_prices[ticker] = historical_data.loc[current_date, 'close']
            else:
                # Other instruments use their specific close columns
                close_col = f'{ticker}_close'
                if close_col in historical_data.columns:
                    price = historical_data.loc[current_date, close_col]
                    if pd.notna(price) and price > 0:
                        current_prices[ticker] = price
        
        return current_prices
    
    def _forecast_to_position(self, forecast_value: float, current_prices: Dict, instrument_name: str) -> float:
        """Convert forecast to position using portfolio logic"""
        try:
            # Use simplified position sizing from portfolio
            # forecast_value is typically -20 to +20, convert to position size
            
            # Get volatility target from portfolio
            vol_target = self.portfolio.volatility_target
            
            # Simple position sizing: forecast/20 * volatility_target * leverage_factor
            # This gives us a position between -vol_target and +vol_target
            leverage_factor = 2.0  # Adjust based on your risk preference
            position = (forecast_value / 20.0) * vol_target * leverage_factor
            
            # Apply maximum leverage constraint
            max_position = self.portfolio.max_leverage
            position = max(-max_position, min(max_position, position))
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error converting forecast to position: {str(e)}")
            return 0.0
    
    def _calculate_cta_results(self, price_df: pd.DataFrame) -> tuple:
        """Calculate backtest results using CTA-style method for multiple instruments"""
        def get_max_drawdown(array):
            array = pd.Series(array)
            cummax = array.cummax()
            return array / cummax - 1

        df = price_df.copy()
        
        # Fill NaN positions for all instruments
        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            df[f'pos_{ticker}'] = df[f'pos_{ticker}'].fillna(0)
        
        df = df.ffill().fillna(0)
        
        # Calculate benchmark (using primary instrument)
        df['base_balance'] = df['close'] / df['close'].iloc[0]
        df['chg_primary'] = df['close'].pct_change().fillna(0)
        
        # Calculate returns for each instrument
        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            if ticker == list(self.portfolio.instruments)[0].ticker:
                # Primary instrument uses 'close'
                df[f'chg_{ticker}'] = df['close'].pct_change().fillna(0)
            else:
                # Other instruments use their close columns
                close_col = f'{ticker}_close'
                if close_col in df.columns:
                    df[f'chg_{ticker}'] = df[close_col].pct_change().fillna(0)
                else:
                    df[f'chg_{ticker}'] = 0.0
        
        # Calculate strategy equity (portfolio level)
        df['strategy_balance'] = self.portfolio.initial_capital
        for i in range(1, len(df)):
            total_return = 0.0
            
            # Sum returns from all instruments weighted by positions
            for instrument in self.portfolio.instruments:
                ticker = instrument.ticker
                prev_pos = df.iloc[i-1][f'pos_{ticker}']
                daily_return = df.iloc[i][f'chg_{ticker}']
                
                if pd.notna(prev_pos) and pd.notna(daily_return):
                    # Weight by position size (positions are already scaled by portfolio logic)
                    instrument_return = daily_return * prev_pos
                    total_return += instrument_return
            
            df.iloc[i, df.columns.get_loc('strategy_balance')] = (
                df.iloc[i-1]['strategy_balance'] * (1.0 + total_return)
            )
        
        # Calculate drawdown
        df['drawdown'] = get_max_drawdown(df['strategy_balance'] / self.portfolio.initial_capital)
        
        # Calculate statistics
        stats = {}
        final_balance = df['strategy_balance'].iloc[-1]
        initial_balance = self.portfolio.initial_capital
        
        stats['MaxDrawDown'] = min(df['drawdown'])
        stats['return'] = (final_balance - initial_balance) / initial_balance
        
        # Annualized return
        years = (df.index[-1] - df.index[0]).days / 365.25
        if years > 0:
            if years <= 1:
                stats['yearReturn'] = stats['return'] / years
            else:
                stats['yearReturn'] = (final_balance / initial_balance) ** (1 / years) - 1
        else:
            stats['yearReturn'] = 0
        
        # Return/Drawdown ratio
        if stats['MaxDrawDown'] != 0:
            stats['return_drawdown_ratio'] = -stats['return'] / stats['MaxDrawDown']
        else:
            stats['return_drawdown_ratio'] = 0
        
        # Sharpe ratio
        strategy_returns = (df['strategy_balance'] / df['strategy_balance'].shift(1) - 1).dropna()
        if len(strategy_returns) > 1 and strategy_returns.std() > 0:
            stats['sharpe_ratio'] = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            stats['sharpe_ratio'] = 0
        
        # Win rate (portfolio level)
        # Calculate daily portfolio returns
        daily_portfolio_returns = []
        for i in range(1, len(df)):
            daily_return = 0.0
            for instrument in self.portfolio.instruments:
                ticker = instrument.ticker
                pos = df.iloc[i-1][f'pos_{ticker}']
                ret = df.iloc[i][f'chg_{ticker}']
                if pd.notna(pos) and pd.notna(ret):
                    daily_return += ret * pos
            daily_portfolio_returns.append(daily_return)
        
        if len(daily_portfolio_returns) > 0:
            trading_returns = [r for r in daily_portfolio_returns if abs(r) > 0.0001]
            stats['win_rate'] = (pd.Series(trading_returns) > 0).mean() if len(trading_returns) > 0 else 0
        else:
            stats['win_rate'] = 0
        
        return df, stats
    
    def _populate_backtest_results(self, results_df: pd.DataFrame, stats: Dict):
        """Populate BacktestResults with calculated data for multiple instruments"""
        # Store the key metrics in results
        self.results.metrics = {
            'total_return': stats['return'],
            'annualized_return': stats['yearReturn'],
            'max_drawdown': stats['MaxDrawDown'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'win_rate': stats['win_rate'],
            'return_drawdown_ratio': stats['return_drawdown_ratio'],
            'final_capital': results_df['strategy_balance'].iloc[-1]
        }
        
        # Store daily data
        self.results.daily_capital = results_df['strategy_balance']
        self.results.daily_returns = (results_df['strategy_balance'] / results_df['strategy_balance'].shift(1) - 1).fillna(0)
        
        # Store positions and forecasts for all instruments
        positions_data = {}
        forecasts_data = {}
        
        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            positions_data[ticker] = results_df[f'pos_{ticker}']
            forecasts_data[ticker] = results_df[f'forecast_{ticker}']
        
        self.results.daily_positions = pd.DataFrame(positions_data, index=results_df.index)
        self.results.daily_forecasts = pd.DataFrame(forecasts_data, index=results_df.index)


class LiveTradingEngine(TradingEngine):
    """
    Live trading implementation of TradingEngine
    Executes trades in real-time with broker integration
    """
    
    def __init__(self, portfolio, strategy, broker_connection, data_feed):
        super().__init__(portfolio, strategy)
        self.broker = broker_connection
        self.data_feed = data_feed
        self.is_running = False
        
    def run(self, frequency: str = '1min'):
        """
        Run live trading with specified frequency
        
        Parameters:
        -----------
        frequency: str
            Trading frequency ('1min', '5min', '1hour', etc.)
        """
        self.logger.info(f"Starting live trading with {frequency} frequency")
        self.is_running = True
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Get real-time prices
                current_prices = self.get_current_prices()
                
                if not current_prices:
                    continue
                
                # Generate signals
                forecasts = self.generate_signals(current_time, current_prices)
                
                # Calculate new positions
                target_positions = self.calculate_positions(forecasts, current_prices)
                
                # Execute trades if needed
                self._execute_trades(target_positions)
                
                # Update positions
                self.update_positions(target_positions)
                
                # Wait for next cycle
                self._wait_for_next_cycle(frequency)
                
            except KeyboardInterrupt:
                self.logger.info("Live trading stopped by user")
                self.stop()
            except Exception as e:
                self.logger.error(f"Error in live trading loop: {str(e)}")
                continue
    
    def get_current_prices(self, timestamp=None) -> Dict:
        """Get real-time market prices from data feed"""
        try:
            prices = {}
            for instrument in self.portfolio.instruments:
                price = self.data_feed.get_price(instrument.ticker)
                if price is not None:
                    prices[instrument.ticker] = price
            return prices
        except Exception as e:
            self.logger.error(f"Error getting real-time prices: {str(e)}")
            return {}
    
    def _execute_trades(self, target_positions: Dict):
        """Execute trades through broker to reach target positions"""
        for instrument_name, target_size in target_positions.items():
            current_size = self.current_positions.get(instrument_name, 0)
            trade_size = target_size - current_size
            
            if abs(trade_size) > 0.01:  # Minimum trade threshold
                try:
                    order_id = self.broker.place_order(
                        symbol=instrument_name,
                        quantity=trade_size,
                        order_type='market'
                    )
                    self.logger.info(f"Placed order {order_id}: {trade_size} shares of {instrument_name}")
                except Exception as e:
                    self.logger.error(f"Failed to place order for {instrument_name}: {str(e)}")
    
    def _wait_for_next_cycle(self, frequency: str):
        """Wait for the next trading cycle"""
        import time
        
        freq_map = {
            '1sec': 1,
            '1min': 60,
            '5min': 300,
            '15min': 900,
            '1hour': 3600
        }
        
        sleep_time = freq_map.get(frequency, 60)
        time.sleep(sleep_time)
    
    def stop(self):
        """Stop the live trading engine"""
        self.is_running = False
        self.logger.info("Live trading engine stopped")


