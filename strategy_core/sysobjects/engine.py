from abc import ABC, abstractmethod
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

from .backtest_results import BacktestResults
from .forecasts import Forecast


class TradingEngine(ABC):
    """
    Abstract base class for trading engines
    Contains all shared functionality between backtesting and production trading
    """

    def __init__(self, portfolio, strategy, data_handler, start_date, end_date):
        self.portfolio = portfolio
        self.strategy = strategy
        self.data_handler = data_handler
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.current_positions = {}
        self.results = None
        self.load_bars = 70
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def _save_signals_to_file(self, portfolio_results: pd.DataFrame):
        """Save signals to file - path differs between backtest and production"""
        pass

    def run(self) -> BacktestResults:
        """
        Run the backtest over the specified date range - per-instrument approach
        
        Returns:
        --------
        BacktestResults
            Complete backtest results and performance metrics
        """
        self.logger.info(f"Starting multi-instrument backtest from {self.start_date} to {self.end_date}")

        # Prepare per-instrument data structures
        instrument_data = self._prepare_per_instrument_data()
        if not instrument_data:
            raise ValueError("No price data available for backtest")

        # Get common date range across all instruments
        common_dates = self._get_common_date_range(instrument_data)
        self.logger.info(f"Backtesting over {len(common_dates)} trading days")

        # Initialize results tracking
        portfolio_results = pd.DataFrame(index=common_dates)
        portfolio_results['portfolio_value'] = float(self.portfolio.initial_capital)

        # Initialize position and forecast tracking per instrument
        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            portfolio_results[f'pos_{ticker}'] = 0.0
            portfolio_results[f'forecast_{ticker}'] = 0.0

        # Strategy execution loop - per instrument
        for i, current_date in enumerate(common_dates):
            if i < self.load_bars:
                continue

            try:
                # Generate signals for each instrument independently
                instrument_forecasts = {}
                instrument_positions = {}

                for instrument in self.portfolio.instruments:
                    ticker = instrument.ticker
                    
                    if ticker not in instrument_data:
                        continue

                    # Check if this instrument has data for current date
                    ticker_data = instrument_data[ticker]
                    if current_date not in ticker_data.index:
                        # Instrument doesn't have data for this date, skip
                        continue
                    
                    # Get historical data up to current date for this instrument
                    historical_data = ticker_data.loc[:current_date]
                    
                    if len(historical_data) < self.load_bars:
                        continue

                    # Get current price for this instrument
                    current_price = historical_data.iloc[-1]['close'] if not historical_data.empty else None
                    
                    if current_price is None or current_price <= 0:
                        continue

                    # Generate forecast using instrument-specific data
                    current_prices = {ticker: current_price}
                    forecasts = self.generate_signals(current_date, current_prices)
                    
                    forecast_value = 0.0
                    if ticker in forecasts:
                        forecast_obj = forecasts[ticker]
                        if hasattr(forecast_obj, 'iloc') and len(forecast_obj) > 0:
                            forecast_value = forecast_obj.iloc[-1]
                        elif isinstance(forecast_obj, (int, float)):
                            forecast_value = forecast_obj

                    instrument_forecasts[ticker] = forecast_value

                    # Convert forecast to position using instrument-specific data
                    raw_position = self._forecast_to_position_per_instrument(
                        forecast_value, historical_data, instrument
                    )
                    
                    # Apply no-trade zone filters
                    filtered_position = self._apply_no_trade_filters(
                        raw_position, forecast_value, ticker, current_date
                    )
                    
                    instrument_positions[ticker] = filtered_position

                # Update portfolio results
                for ticker, forecast in instrument_forecasts.items():
                    portfolio_results.loc[current_date, f'forecast_{ticker}'] = forecast
                
                for ticker, position in instrument_positions.items():
                    portfolio_results.loc[current_date, f'pos_{ticker}'] = position

                # Forward fill positions
                for instrument in self.portfolio.instruments:
                    ticker = instrument.ticker
                    portfolio_results.loc[:current_date, f'pos_{ticker}'] = portfolio_results.loc[:current_date, f'pos_{ticker}'].ffill()

            except Exception as e:
                self.logger.error(f"Error processing {current_date}: {str(e)}")
                continue

        # Fill any remaining NaN positions
        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            portfolio_results[f'pos_{ticker}'] = portfolio_results[f'pos_{ticker}'].fillna(0)

        # Calculate portfolio performance
        portfolio_results = self._calculate_portfolio_performance(portfolio_results, instrument_data)
        
        # Calculate statistics
        stats = self._calculate_portfolio_stats(portfolio_results)

        # Create BacktestResults object
        self.results = BacktestResults(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.portfolio.initial_capital
        )

        # Populate results
        self._populate_backtest_results(portfolio_results, stats)

        # Store data for plotting
        self.results.price_data = portfolio_results
        self.results.instrument_data = instrument_data

        # Save signals to file
        self._save_signals_to_file(portfolio_results)

        self.logger.info("Multi-instrument backtest completed successfully")
        return self.results

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

    def _prepare_per_instrument_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare separate OHLC data for each instrument"""
        if not self.data_handler:
            return {}

        instrument_data = {}
        
        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            price_data = self.data_handler.get(ticker)
            
            if price_data is None:
                self.logger.warning(f"No data available for {ticker}")
                continue

            # Extract OHLC data for this instrument
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
                # Assume it's a Series of close prices
                df = pd.DataFrame({'close': price_data})
                df['open'] = df['close']
                df['high'] = df['close']
                df['low'] = df['close']
                df['volume'] = 0

            # Clean data - remove invalid prices
            df = df[(df['close'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
            df = df.dropna(subset=['close'])
            
            # Forward fill any remaining NaN values
            df = df.ffill()

            # Filter by date range
            mask = (df.index >= self.start_date) & (df.index <= self.end_date)
            df = df[mask]
            
            if not df.empty:
                instrument_data[ticker] = df
                self.logger.info(f"Loaded {len(df)} trading days for {ticker}")
            else:
                self.logger.warning(f"No valid data for {ticker} in date range")

        return instrument_data
    
    def _get_common_date_range(self, instrument_data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Get union of all trading dates across instruments (allows staggered starts)"""
        if not instrument_data:
            return pd.DatetimeIndex([])
        
        # Use union of all instruments' date ranges to allow staggered starts
        all_dates = set()
        for ticker, data in instrument_data.items():
            # Filter dates within our backtest range
            filtered_dates = data.index[(data.index >= self.start_date) & (data.index <= self.end_date)]
            all_dates = all_dates.union(set(filtered_dates))
        
        if all_dates:
            return pd.DatetimeIndex(sorted(all_dates))
        else:
            return pd.DatetimeIndex([])

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

    def _forecast_to_position_per_instrument(self, forecast_value: float, historical_data: pd.DataFrame, instrument) -> float:
        """Convert forecast to position using per-instrument data"""
        try:
            # Use instrument-specific volatility and characteristics
            if len(historical_data) < 25:
                return 0.0
            
            # Calculate instrument volatility using its own price data
            returns = historical_data['close'].pct_change().dropna()
            if len(returns) < 20:
                return 0.0
                
            instrument_vol = returns.rolling(window=min(25, len(returns))).std().iloc[-1] * np.sqrt(252)
            
            if pd.isna(instrument_vol) or instrument_vol <= 0:
                return 0.0

            # Get volatility target from portfolio
            vol_target = self.portfolio.volatility_target

            # Position sizing: forecast/forecast_cap * vol_target/instrument_vol * capital_multiplier
            forecast_cap = 20.0
            position = (forecast_value / forecast_cap) * (vol_target / instrument_vol)

            # Apply maximum leverage constraint
            max_position = self.portfolio.max_leverage
            position = max(-max_position, min(max_position, position))

            return position

        except Exception as e:
            self.logger.error(f"Error converting forecast to position for {instrument.ticker}: {str(e)}")
            return 0.0
    
    def _apply_no_trade_filters(self, raw_position: float, forecast_value: float, 
                              ticker: str, current_date) -> float:
        """Apply no-trade zone filters to position"""
        
        # Get current position from portfolio results
        current_position = 0.0
        # For the first calculation, we don't have previous position data
        # so we'll track it in a simple dict for now
        if not hasattr(self, '_previous_positions'):
            self._previous_positions = {}
        
        current_position = self._previous_positions.get(ticker, 0.0)
        
        # Filter 1: Minimum forecast threshold
        if abs(forecast_value) < self.portfolio.min_forecast_threshold:
            return current_position
        
        # Filter 2: Minimum position size threshold  
        if abs(raw_position) < self.portfolio.min_position_threshold:
            return current_position
        
        # Filter 3: No-trade buffer zone
        position_change = abs(raw_position - current_position)
        buffer_threshold = abs(current_position) * self.portfolio.no_trade_buffer
        
        if position_change < buffer_threshold and abs(current_position) > 0:
            filtered_position = current_position
        else:
            filtered_position = raw_position
        
        # Update previous position
        self._previous_positions[ticker] = filtered_position
        
        return filtered_position
    
    def _calculate_portfolio_performance(self, portfolio_results: pd.DataFrame, instrument_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate portfolio performance using per-instrument data"""
        
        # Initialize portfolio value tracking
        portfolio_results['portfolio_return'] = 0.0
        
        for i in range(1, len(portfolio_results)):
            current_date = portfolio_results.index[i]
            prev_date = portfolio_results.index[i-1]
            
            total_return = 0.0
            
            # Calculate returns for each instrument
            for instrument in self.portfolio.instruments:
                ticker = instrument.ticker
                
                if ticker not in instrument_data:
                    continue
                    
                # Get position from previous day
                prev_position = portfolio_results.loc[prev_date, f'pos_{ticker}']
                
                if pd.isna(prev_position) or prev_position == 0:
                    continue
                
                # Get current and previous prices for this instrument
                if current_date in instrument_data[ticker].index and prev_date in instrument_data[ticker].index:
                    current_price = instrument_data[ticker].loc[current_date, 'close']
                    prev_price = instrument_data[ticker].loc[prev_date, 'close']
                    
                    if pd.notna(current_price) and pd.notna(prev_price) and prev_price > 0:
                        # Calculate instrument return
                        instrument_return = (current_price - prev_price) / prev_price
                        
                        # Weight by position size
                        weighted_return = instrument_return * prev_position
                        total_return += weighted_return
            
            portfolio_results.loc[current_date, 'portfolio_return'] = total_return
            
            # Update portfolio value
            portfolio_results.loc[current_date, 'portfolio_value'] = (
                portfolio_results.loc[prev_date, 'portfolio_value'] * (1.0 + total_return)
            )
        
        return portfolio_results
    
    def _calculate_portfolio_stats(self, portfolio_results: pd.DataFrame) -> Dict:
        """Calculate portfolio statistics"""
        stats = {}
        
        returns = portfolio_results['portfolio_return'].dropna()
        portfolio_values = portfolio_results['portfolio_value'].dropna()
        
        if len(portfolio_values) == 0:
            return {'return': 0, 'yearReturn': 0, 'MaxDrawDown': 0, 'sharpe_ratio': 0, 'win_rate': 0}
        
        # Total and annualized returns
        final_value = portfolio_values.iloc[-1]
        initial_value = self.portfolio.initial_capital
        stats['return'] = (final_value - initial_value) / initial_value
        
        # Annualized return
        years = (portfolio_results.index[-1] - portfolio_results.index[0]).days / 365.25
        if years > 0:
            if years <= 1:
                stats['yearReturn'] = stats['return'] / years
            else:
                stats['yearReturn'] = (final_value / initial_value) ** (1 / years) - 1
        else:
            stats['yearReturn'] = 0
        
        # Maximum drawdown
        cummax = portfolio_values.cummax()
        drawdown = portfolio_values / cummax - 1
        stats['MaxDrawDown'] = drawdown.min()
        
        # Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            stats['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
        else:
            stats['sharpe_ratio'] = 0
        
        # Win rate
        trading_returns = returns[returns.abs() > 0.0001]
        stats['win_rate'] = (trading_returns > 0).mean() if len(trading_returns) > 0 else 0
        
        return stats

    def _populate_backtest_results(self, results_df: pd.DataFrame, stats: Dict):
        """Populate BacktestResults with calculated data for multiple instruments"""
        # Store the key metrics in results
        self.results.metrics = {
            'total_return': stats['return'],
            'annualized_return': stats['yearReturn'],
            'max_drawdown': stats['MaxDrawDown'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'win_rate': stats['win_rate'],
            'final_capital': results_df['portfolio_value'].iloc[-1]
        }

        # Store daily data
        self.results.daily_capital = results_df['portfolio_value']
        self.results.daily_returns = results_df['portfolio_return']

        # Store positions and forecasts for all instruments
        positions_data = {}
        forecasts_data = {}

        for instrument in self.portfolio.instruments:
            ticker = instrument.ticker
            positions_data[ticker] = results_df[f'pos_{ticker}']
            forecasts_data[ticker] = results_df[f'forecast_{ticker}']

        self.results.daily_positions = pd.DataFrame(positions_data, index=results_df.index)
        self.results.daily_forecasts = pd.DataFrame(forecasts_data, index=results_df.index)


class BacktestEngine(TradingEngine):
    """
    Backtesting implementation - saves signals to strategy_backtest/results
    """

    def _save_signals_to_file(self, portfolio_results: pd.DataFrame):
        """Save trading signals to backtest results directory"""
        
        # Create results directory if it doesn't exist
        results_dir = "strategy_backtest/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        signals_file = os.path.join(results_dir, f"signals_{timestamp}.txt")
        
        try:
            with open(signals_file, 'w') as f:
                f.write("=== TRADING SIGNALS LOG ===\n")
                f.write(f"Backtest Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Extract forecast and position columns
                forecast_cols = [col for col in portfolio_results.columns if col.startswith('forecast_')]
                position_cols = [col for col in portfolio_results.columns if col.startswith('pos_')]
                
                # Write header
                f.write("Date".ljust(12))
                for instrument in self.portfolio.instruments:
                    ticker = instrument.ticker
                    f.write(f"{ticker}_Signal".ljust(15))
                    f.write(f"{ticker}_Position".ljust(15))
                f.write("Portfolio_Value".ljust(15))
                f.write("\n")
                f.write("-" * (12 + len(self.portfolio.instruments) * 30 + 15) + "\n")
                
                # Write signals data - only record when positions change
                prev_positions = {}
                for instrument in self.portfolio.instruments:
                    prev_positions[instrument.ticker] = 0.0
                
                for date, row in portfolio_results.iterrows():
                    # Check if any position changed
                    position_changed = False
                    current_positions = {}
                    
                    for instrument in self.portfolio.instruments:
                        ticker = instrument.ticker
                        position_val = row.get(f'pos_{ticker}', 0.0)
                        current_positions[ticker] = position_val
                        
                        # Check if position changed significantly (avoid floating point noise)
                        if abs(position_val - prev_positions[ticker]) > 0.0001:
                            position_changed = True
                    
                    # Only record if position changed or it's the first/last day
                    if position_changed or date == portfolio_results.index[0] or date == portfolio_results.index[-1]:
                        f.write(f"{date.strftime('%Y-%m-%d')}".ljust(12))
                        
                        for instrument in self.portfolio.instruments:
                            ticker = instrument.ticker
                            
                            # Get forecast/signal
                            forecast_val = row.get(f'forecast_{ticker}', 0.0)
                            position_val = current_positions[ticker]
                            
                            # Format signal strength and check for no-trade zone
                            if abs(forecast_val) < 0.01:
                                signal_str = "NEUTRAL "
                            elif forecast_val > 0:
                                signal_str = f"LONG({forecast_val:.3f})"
                            else:
                                signal_str = f"SHORT({forecast_val:.3f})"
                            
                            # Check if position was filtered by no-trade zone
                            portfolio_config = getattr(self.portfolio, 'min_position_threshold', 0.01)
                            forecast_threshold = getattr(self.portfolio, 'min_forecast_threshold', 0.5)
                            
                            # Calculate what the raw position would have been
                            raw_position = 0.0
                            if abs(forecast_val) > 0.01:  # Only calculate if there's a meaningful forecast
                                try:
                                    # Estimate raw position using same logic as engine
                                    ticker_data = getattr(self, 'instrument_data', {}).get(ticker)
                                    if ticker_data is not None and date in ticker_data.index:
                                        historical_data = ticker_data.loc[:date]
                                        if len(historical_data) >= 25:
                                            returns = historical_data['close'].pct_change().dropna()
                                            if len(returns) >= 20:
                                                instrument_vol = returns.rolling(window=25).std().iloc[-1] * np.sqrt(252)
                                                if not pd.isna(instrument_vol) and instrument_vol > 0:
                                                    vol_target = self.portfolio.volatility_target
                                                    forecast_cap = 20.0
                                                    raw_position = (forecast_val / forecast_cap) * (vol_target / instrument_vol)
                                                    raw_position = max(-self.portfolio.max_leverage, min(self.portfolio.max_leverage, raw_position))
                                except:
                                    pass
                            
                            # Add filter indicators
                            if abs(forecast_val) < forecast_threshold:
                                signal_str += "*WEAK "
                            elif abs(raw_position) >= portfolio_config and abs(position_val) < portfolio_config:
                                signal_str += "*SMALL "
                            elif abs(raw_position) > abs(position_val) + 0.001:  # Position was reduced by buffer
                                signal_str += "*BUFFER "
                            
                            f.write(f"{signal_str}".ljust(15))
                            f.write(f"{position_val:.4f}".ljust(15))
                        
                        # Portfolio value
                        portfolio_val = row.get('portfolio_value', 0.0)
                        f.write(f"{portfolio_val:.2f}".ljust(15))
                        f.write("\n")
                    
                    # Update previous positions
                    prev_positions = current_positions.copy()
        except Exception as e:
            self.logger.error(f"Error saving signals to file: {str(e)}")


class ProductionEngine(TradingEngine):
    """
    Production trading engine - identical to BacktestEngine but saves signals to strategy_production/order_signal
    """

    def _save_signals_to_file(self, portfolio_results: pd.DataFrame):
        """Save trading signals to production order_signal directory - full history if empty, latest only if exists"""
        
        # Create results directory if it doesn't exist
        results_dir = "order_signal"
        os.makedirs(results_dir, exist_ok=True)
        
        # Use a consistent filename
        signals_file = os.path.join(results_dir, "signals.txt")
        
        try:
            # Check if file exists and has content
            file_exists = os.path.exists(signals_file) and os.path.getsize(signals_file) > 0
            
            if not file_exists:
                # File is empty or doesn't exist - write full history
                with open(signals_file, 'w') as f:
                    f.write("=== PRODUCTION TRADING SIGNALS LOG ===\n")
                    f.write("Date".ljust(12))
                    for instrument in self.portfolio.instruments:
                        ticker = instrument.ticker
                        f.write(f"{ticker}_Signal".ljust(15))
                        f.write(f"{ticker}_Position".ljust(15))
                    f.write("Portfolio_Value".ljust(15))
                    f.write("Generated".ljust(20))
                    f.write("\n")
                    f.write("-" * (12 + len(self.portfolio.instruments) * 30 + 15 + 20) + "\n")
                    
                    # Write all signals data - only record when positions change
                    prev_positions = {}
                    for instrument in self.portfolio.instruments:
                        prev_positions[instrument.ticker] = 0.0
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    for date, row in portfolio_results.iterrows():
                        # Check if any position changed
                        position_changed = False
                        current_positions = {}
                        
                        for instrument in self.portfolio.instruments:
                            ticker = instrument.ticker
                            position_val = row.get(f'pos_{ticker}', 0.0)
                            current_positions[ticker] = position_val
                            
                            # Check if position changed significantly (avoid floating point noise)
                            if abs(position_val - prev_positions[ticker]) > 0.0001:
                                position_changed = True
                        
                        # Only record if position changed or it's the first/last day
                        if position_changed or date == portfolio_results.index[0] or date == portfolio_results.index[-1]:
                            f.write(f"{date.strftime('%Y-%m-%d')}".ljust(12))
                            
                            for instrument in self.portfolio.instruments:
                                ticker = instrument.ticker
                                
                                # Get forecast/signal
                                forecast_val = row.get(f'forecast_{ticker}', 0.0)
                                position_val = current_positions[ticker]
                                
                                # Format signal strength and check for no-trade zone
                                if abs(forecast_val) < 0.01:
                                    signal_str = "NEUTRAL "
                                elif forecast_val > 0:
                                    signal_str = f"LONG({forecast_val:.3f})"
                                else:
                                    signal_str = f"SHORT({forecast_val:.3f})"
                                
                                # Check if position was filtered by no-trade zone
                                portfolio_config = getattr(self.portfolio, 'min_position_threshold', 0.01)
                                forecast_threshold = getattr(self.portfolio, 'min_forecast_threshold', 0.5)
                                
                                # Calculate what the raw position would have been
                                raw_position = 0.0
                                if abs(forecast_val) > 0.01:  # Only calculate if there's a meaningful forecast
                                    try:
                                        # Estimate raw position using same logic as engine
                                        ticker_data = getattr(self, 'instrument_data', {}).get(ticker)
                                        if ticker_data is not None and date in ticker_data.index:
                                            historical_data = ticker_data.loc[:date]
                                            if len(historical_data) >= 25:
                                                returns = historical_data['close'].pct_change().dropna()
                                                if len(returns) >= 20:
                                                    instrument_vol = returns.rolling(window=25).std().iloc[-1] * np.sqrt(252)
                                                    if not pd.isna(instrument_vol) and instrument_vol > 0:
                                                        vol_target = self.portfolio.volatility_target
                                                        forecast_cap = 20.0
                                                        raw_position = (forecast_val / forecast_cap) * (vol_target / instrument_vol)
                                                        raw_position = max(-self.portfolio.max_leverage, min(self.portfolio.max_leverage, raw_position))
                                    except:
                                        pass
                                
                                # Add filter indicators
                                if abs(forecast_val) < forecast_threshold:
                                    signal_str += "*WEAK "
                                elif abs(raw_position) >= portfolio_config > abs(position_val):
                                    signal_str += "*SMALL "
                                elif abs(raw_position) > abs(position_val) + 0.001:  # Position was reduced by buffer
                                    signal_str += "*BUFFER "
                                
                                f.write(f"{signal_str}".ljust(15))
                                f.write(f"{position_val:.4f}".ljust(15))
                            
                            # Portfolio value and timestamp
                            portfolio_val = row.get('portfolio_value', 0.0)
                            f.write(f"{portfolio_val:.2f}".ljust(15))
                            f.write(f"{timestamp}".ljust(20))
                            f.write("\n")
                        
                        # Update previous positions
                        prev_positions = current_positions.copy()
            else:
                # File exists with content - check if latest signal already exists
                latest_date = portfolio_results.index[-1]
                latest_date_str = latest_date.strftime('%Y-%m-%d')
                
                # Read the last line to check if this date already exists
                with open(signals_file, 'r') as f:
                    lines = f.readlines()
                    last_line = lines[-1].strip() if lines else ""
                    
                # Check if the last line already contains today's date
                if last_line.startswith(latest_date_str):
                    # Signal for this date already exists, skip writing
                    self.logger.info(f"Signal for {latest_date_str} already exists, skipping duplicate")
                    return
                
                # File exists but doesn't have today's signal - append it
                with open(signals_file, 'a') as f:
                    latest_row = portfolio_results.iloc[-1]
                    
                    # Write the latest signal line
                    f.write(f"{latest_date_str}".ljust(12))
                    
                    for instrument in self.portfolio.instruments:
                        ticker = instrument.ticker
                        
                        # Get forecast/signal
                        forecast_val = latest_row.get(f'forecast_{ticker}', 0.0)
                        position_val = latest_row.get(f'pos_{ticker}', 0.0)
                        
                        # Format signal strength and check for no-trade zone
                        if abs(forecast_val) < 0.01:
                            signal_str = "NEUTRAL "
                        elif forecast_val > 0:
                            signal_str = f"LONG({forecast_val:.3f})"
                        else:
                            signal_str = f"SHORT({forecast_val:.3f})"
                        
                        # Check if position was filtered by no-trade zone
                        portfolio_config = getattr(self.portfolio, 'min_position_threshold', 0.01)
                        forecast_threshold = getattr(self.portfolio, 'min_forecast_threshold', 0.5)
                        
                        # Calculate what the raw position would have been
                        raw_position = 0.0
                        if abs(forecast_val) > 0.01:  # Only calculate if there's a meaningful forecast
                            try:
                                # Estimate raw position using same logic as engine
                                ticker_data = getattr(self, 'instrument_data', {}).get(ticker)
                                if ticker_data is not None and latest_date in ticker_data.index:
                                    historical_data = ticker_data.loc[:latest_date]
                                    if len(historical_data) >= 25:
                                        returns = historical_data['close'].pct_change().dropna()
                                        if len(returns) >= 20:
                                            instrument_vol = returns.rolling(window=25).std().iloc[-1] * np.sqrt(252)
                                            if not pd.isna(instrument_vol) and instrument_vol > 0:
                                                vol_target = self.portfolio.volatility_target
                                                forecast_cap = 20.0
                                                raw_position = (forecast_val / forecast_cap) * (vol_target / instrument_vol)
                                                raw_position = max(-self.portfolio.max_leverage, min(self.portfolio.max_leverage, raw_position))
                            except:
                                pass
                        
                        # Add filter indicators
                        if abs(forecast_val) < forecast_threshold:
                            signal_str += "*WEAK "
                        elif abs(raw_position) >= portfolio_config > abs(position_val):
                            signal_str += "*SMALL "
                        elif abs(raw_position) > abs(position_val) + 0.001:  # Position was reduced by buffer
                            signal_str += "*BUFFER "
                        
                        f.write(f"{signal_str}".ljust(15))
                        f.write(f"{position_val:.4f}".ljust(15))
                    
                    # Portfolio value
                    portfolio_val = latest_row.get('portfolio_value', 0.0)
                    f.write(f"{portfolio_val:.2f}".ljust(15))
                    
                    # Timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{timestamp}".ljust(20))
                    f.write("\n")
                
        except Exception as e:
            self.logger.error(f"Error saving signals to file: {str(e)}")