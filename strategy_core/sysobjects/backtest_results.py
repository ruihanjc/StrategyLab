"""
BacktestResults class for storing and analyzing backtest performance
"""

import logging
from datetime import datetime
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd


class BacktestResults:
    """
    Stores and analyzes backtest results and performance metrics
    """
    
    def __init__(self, start_date: datetime, end_date: datetime, initial_capital: float):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Daily tracking
        self.daily_returns = pd.Series(dtype=float)
        self.daily_positions = pd.DataFrame()
        self.daily_prices = pd.DataFrame()
        self.daily_forecasts = pd.DataFrame()
        self.daily_pnl = pd.Series(dtype=float)
        self.daily_capital = pd.Series(dtype=float)
        
        # Performance metrics
        self.metrics = {}
        
        # Trades log
        self.trades = []
        
        # Price data for plotting
        self.price_data = None
        
        self.logger = logging.getLogger(f"{__name__}.BacktestResults")
        
    def update_daily_results(self, date: datetime, positions: Dict, prices: Dict, forecasts: Dict):
        """
        Update daily backtest results
        
        Parameters:
        -----------
        date: datetime
            Current date
        positions: Dict
            Current positions for each instrument
        prices: Dict  
            Current prices for each instrument
        forecasts: Dict
            Current forecasts for each instrument
        """
        # Store positions
        position_data = {}
        for instrument_name, position_value in positions.items():
            if hasattr(position_value, 'positions'):
                # If it's a PositionSeries object
                try:
                    latest_positions = position_value.get_portfolio_positions()
                    if not latest_positions.empty:
                        position_data[instrument_name] = latest_positions.iloc[-1]
                    else:
                        position_data[instrument_name] = 0
                except Exception:
                    position_data[instrument_name] = 0
            elif hasattr(position_value, '__len__') and not isinstance(position_value, str):
                # If it's an array-like object, take the last value
                try:
                    position_data[instrument_name] = position_value[-1] if len(position_value) > 0 else 0
                except Exception:
                    position_data[instrument_name] = 0
            else:
                # If it's a direct value (number)
                position_data[instrument_name] = float(position_value) if position_value is not None else 0
                
        if position_data:
            self.daily_positions.loc[date] = pd.Series(position_data)
        
        # Store prices
        if prices:
            self.daily_prices.loc[date] = pd.Series(prices)
        
        # Store forecasts  
        forecast_data = {}
        for instrument_name, forecast in forecasts.items():
            if hasattr(forecast, 'forecast_value'):
                forecast_data[instrument_name] = forecast.forecast_value
            else:
                forecast_data[instrument_name] = forecast
                
        if forecast_data:
            self.daily_forecasts.loc[date] = pd.Series(forecast_data)
        
        # Calculate daily P&L
        if not self.daily_positions.empty and not self.daily_prices.empty:
            self._calculate_daily_pnl(date)
    
    def _calculate_daily_pnl(self, date: datetime):
        """Calculate P&L for the given date"""
        try:
            if date not in self.daily_positions.index or date not in self.daily_prices.index:
                return

            self.daily_positions.loc[date].fillna(0)
            current_prices = self.daily_prices.loc[date].fillna(0)
            
            # Get previous day data for P&L calculation
            prev_dates = self.daily_positions.index[self.daily_positions.index < date]
            if len(prev_dates) > 0:
                prev_date = prev_dates[-1]
                prev_positions = self.daily_positions.loc[prev_date].fillna(0)
                
                if prev_date in self.daily_prices.index:
                    prev_prices = self.daily_prices.loc[prev_date].fillna(0)
                    
                    # Calculate P&L from price changes
                    common_instruments = set(current_prices.index) & set(prev_prices.index) & set(prev_positions.index)
                    daily_pnl = 0
                    
                    for instrument in common_instruments:
                        price_change = current_prices[instrument] - prev_prices[instrument]
                        pnl_contribution = prev_positions[instrument] * price_change
                        daily_pnl += pnl_contribution
                    
                    self.daily_pnl.loc[date] = daily_pnl
                    
                    # Update capital
                    prev_capital = self.daily_capital.iloc[-1] if len(self.daily_capital) > 0 else self.initial_capital
                    self.daily_capital.loc[date] = prev_capital + daily_pnl
                else:
                    self.daily_capital.loc[date] = self.daily_capital.iloc[-1] if len(self.daily_capital) > 0 else self.initial_capital
            else:
                # First day
                self.daily_pnl.loc[date] = 0
                self.daily_capital.loc[date] = self.initial_capital
                
        except Exception as e:
            self.logger.error(f"Error calculating P&L for {date}: {str(e)}")
    
    def calculate_final_metrics(self):
        """Calculate final performance metrics"""
        if self.daily_capital.empty or self.daily_returns.empty:
            self._calculate_returns()
        
        if self.daily_returns.empty:
            self.logger.warning("No returns data available for metrics calculation")
            return
            
        # Basic metrics
        total_return = (self.daily_capital.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized metrics
        trading_days = len(self.daily_returns)
        years = trading_days / 252
        
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        annualized_vol = self.daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + self.daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win/loss metrics
        positive_days = (self.daily_returns > 0).sum()
        negative_days = (self.daily_returns < 0).sum()
        win_rate = positive_days / len(self.daily_returns) if len(self.daily_returns) > 0 else 0
        
        # Average win/loss
        avg_win = self.daily_returns[self.daily_returns > 0].mean() if positive_days > 0 else 0
        avg_loss = self.daily_returns[self.daily_returns < 0].mean() if negative_days > 0 else 0
        
        self.metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(self.trades),
            'trading_days': trading_days,
            'final_capital': self.daily_capital.iloc[-1] if not self.daily_capital.empty else self.initial_capital
        }
        
        # Calculate yearly metrics
        yearly_metrics = self._calculate_yearly_metrics()
        self.metrics['yearly_performance'] = yearly_metrics
        
        self.logger.info(f"Backtest completed - Total Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}")
    
    def _calculate_returns(self):
        """Calculate daily returns from capital changes"""
        if not self.daily_capital.empty:
            self.daily_returns = self.daily_capital.pct_change().fillna(0)
    
    def _calculate_yearly_metrics(self):
        """Calculate year-by-year performance metrics"""
        if self.daily_returns.empty:
            self._calculate_returns()
        
        if self.daily_returns.empty:
            return {}
        
        yearly_metrics = {}
        
        # Group returns by year
        yearly_returns = self.daily_returns.groupby(self.daily_returns.index.year)
        
        for year, year_returns in yearly_returns:
            if len(year_returns) == 0:
                continue
                
            # Calculate yearly metrics
            year_total_return = (1 + year_returns).prod() - 1
            year_volatility = year_returns.std() * np.sqrt(252)
            year_sharpe = (year_total_return * 252 / len(year_returns)) / year_volatility if year_volatility > 0 else 0
            
            # Drawdown for the year
            year_cumulative = (1 + year_returns).cumprod()
            year_running_max = year_cumulative.cummax()
            year_drawdown = (year_cumulative - year_running_max) / year_running_max
            year_max_drawdown = year_drawdown.min()
            
            # Win/loss metrics for the year
            year_win_rate = (year_returns > 0).mean()
            year_positive_days = (year_returns > 0).sum()
            year_negative_days = (year_returns < 0).sum()
            
            yearly_metrics[year] = {
                'total_return': year_total_return,
                'annualized_return': year_total_return * 252 / len(year_returns),
                'volatility': year_volatility,
                'sharpe_ratio': year_sharpe,
                'max_drawdown': year_max_drawdown,
                'win_rate': year_win_rate,
                'positive_days': year_positive_days,
                'negative_days': year_negative_days,
                'trading_days': len(year_returns)
            }
        
        return yearly_metrics
    
    def get_performance_summary(self) -> Dict:
        """Get summary of performance metrics"""
        self.calculate_final_metrics()
        return self.metrics.copy()
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve (capital over time)"""
        return self.daily_capital.copy()
    
    def get_returns_series(self) -> pd.Series:
        """Get daily returns series"""
        if self.daily_returns.empty:
            self._calculate_returns()
        return self.daily_returns.copy()
    
    def get_positions_df(self) -> pd.DataFrame:
        """Get positions DataFrame"""
        return self.daily_positions.copy()
    
    def get_drawdown_series(self) -> pd.Series:
        """Get drawdown series"""
        if self.daily_returns.empty:
            self._calculate_returns()
            
        cumulative_returns = (1 + self.daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown
    
    def add_trade(self, date: datetime, instrument: str, quantity: float, price: float, side: str):
        """
        Add a trade to the trades log
        
        Parameters:
        -----------
        date: datetime
            Trade date
        instrument: str
            Instrument traded
        quantity: float
            Quantity traded
        price: float
            Execution price
        side: str
            'BUY' or 'SELL'
        """
        trade = {
            'date': date,
            'instrument': instrument,
            'quantity': quantity,
            'price': price,
            'side': side,
            'value': quantity * price
        }
        
        self.trades.append(trade)
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def plot_results(self, price_data=None):
        """Plot backtest results similar to CTA style (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            
            # 1. Equity curves - Strategy vs Benchmark
            if not self.daily_capital.empty:
                # Strategy equity curve (normalized to starting at 1)
                strategy_equity = self.daily_capital / self.initial_capital
                axes[0].plot(strategy_equity.index, strategy_equity.values, label='Strategy', linewidth=2)
                
                # Benchmark (buy and hold) if price data available
                if price_data is not None and not price_data.empty:
                    # Assume first column is the main price series
                    price_col = 'close' if 'close' in price_data.columns else price_data.columns[0]
                    benchmark = price_data[price_col] / price_data[price_col].iloc[0]
                    # Align dates with strategy
                    common_dates = strategy_equity.index.intersection(benchmark.index)
                    if len(common_dates) > 0:
                        axes[0].plot(common_dates, benchmark.loc[common_dates], 
                                   label='Benchmark (Buy & Hold)', alpha=0.7, linewidth=1)
                
                axes[0].set_title('Equity Curves')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[0].set_ylabel('Cumulative Return')
            
            # 2. Drawdown
            if not self.daily_capital.empty:
                drawdown = self.get_drawdown_series()
                axes[1].fill_between(drawdown.index, drawdown.values, 0, 
                                   alpha=0.3, color='red', label='Drawdown')
                axes[1].set_title('Drawdown')
                axes[1].set_ylabel('Drawdown')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
            
            # 3. Positions
            if not self.daily_positions.empty:
                # Plot positions for each instrument
                for col in self.daily_positions.columns:
                    positions = self.daily_positions[col].dropna()
                    if not positions.empty:
                        axes[2].plot(positions.index, positions.values, label=f'{col}', alpha=0.8)
                
                # Also plot combined position
                combined_pos = self.daily_positions.sum(axis=1)
                axes[2].plot(combined_pos.index, combined_pos.values, 
                           label='Portfolio Total', linewidth=2, color='black', linestyle='--')
                
                axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                axes[2].set_title('Position Allocation')
                axes[2].set_ylabel('Position Size')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            
            # 4. Price with signals (if price data provided)
            if price_data is not None and not price_data.empty and not self.daily_positions.empty:
                price_col = 'close' if 'close' in price_data.columns else price_data.columns[0]
                axes[3].plot(price_data.index, price_data[price_col], 
                           color='black', alpha=0.7, label='Price')
                
                # Sample positions to avoid overcrowding
                positions_col = self.daily_positions.columns[0]  # First instrument
                positions = self.daily_positions[positions_col].dropna()
                
                if len(positions) > 0:
                    # Sample some points for signal visualization
                    sample_size = min(100, len(positions))
                    sample_indices = np.linspace(0, len(positions)-1, sample_size).astype(int)
                    sample_positions = positions.iloc[sample_indices]
                    
                    # Get corresponding prices for the sampled dates
                    for date, pos in sample_positions.items():
                        if date in price_data.index:
                            price = price_data.loc[date, price_col]
                            if pos > 0.1:  # Long position threshold
                                axes[3].scatter(date, price, color='red', marker='^', 
                                              s=30, alpha=0.6, label='Long' if date == sample_positions.index[0] else "")
                            elif pos < -0.1:  # Short position threshold
                                axes[3].scatter(date, price, color='green', marker='v', 
                                              s=30, alpha=0.6, label='Short' if date == sample_positions.index[0] else "")
                
                axes[3].set_title('Price & Signals')
                axes[3].set_ylabel('Price')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
            elif not self.daily_forecasts.empty:
                # If no price data, plot forecasts instead
                for col in self.daily_forecasts.columns:
                    forecasts = self.daily_forecasts[col].dropna()
                    if not forecasts.empty:
                        axes[3].plot(forecasts.index, forecasts.values, label=f'Forecast - {col}')
                
                axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.3)
                axes[3].set_title('Forecasts Over Time')
                axes[3].set_ylabel('Forecast Value')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Error plotting results: {str(e)}")
            
    def plot_cta_style(self, price_df=None, save_path=None, show_plot=True):
        """Plot results in CTA style exactly like simple_cta_backtest"""
        try:

            
            # Check if we can use an interactive backend
            interactive_available = False
            current_backend = matplotlib.get_backend()
            
            if show_plot and current_backend.lower() == 'agg':
                # Try to set an interactive backend
                for backend in ['TkAgg', 'Qt5Agg', 'Qt4Agg']:
                    try:
                        matplotlib.use(backend)
                        interactive_available = True
                        self.logger.info(f"Using interactive backend: {backend}")
                        break
                    except:
                        continue
                
                if not interactive_available:
                    matplotlib.use('Agg')
                    self.logger.warning("No interactive display available. Plot will be saved only.")
                    show_plot = False  # Force save-only mode
                    if not save_path:
                        save_path = "strategy_backtest/results/backtest_results.png"  # Default save path
            
            import matplotlib.pyplot as plt
            
            # Create a combined DataFrame similar to CTA format
            if price_df is not None:
                df = price_df.copy()
                
                # Add strategy balance
                if not self.daily_capital.empty:
                    df['strategy_balance'] = self.daily_capital / self.initial_capital
                
                # Add combined position (sum of all positions)
                if not self.daily_positions.empty and len(self.daily_positions.columns) > 0:
                    df['pos'] = self.daily_positions.sum(axis=1)  # Sum all instrument positions
                
                # Add benchmark
                if 'close' in df.columns:
                    df['base_balance'] = df['close'] / df['close'].iloc[0]
                
            else:
                # Create minimal DataFrame from results
                df = pd.DataFrame(index=self.daily_capital.index)
                df['strategy_balance'] = self.daily_capital / self.initial_capital
                if not self.daily_positions.empty:
                    df['pos'] = self.daily_positions.iloc[:, 0]
            
            # Plotting - CTA style with 4 subplots
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            
            # 1. Equity curves
            if 'strategy_balance' in df.columns:
                axes[0].plot(df.index, df['strategy_balance'], label='Strategy', linewidth=2)
            if 'base_balance' in df.columns:
                axes[0].plot(df.index, df['base_balance'], label='Benchmark', alpha=0.7)
            axes[0].set_title('Equity Curves')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. Drawdown
            if 'strategy_balance' in df.columns:
                strategy_balance = df['strategy_balance'].fillna(1)
                cummax = strategy_balance.cummax()
                drawdown = strategy_balance / cummax - 1
                axes[1].fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
            axes[1].set_title('Drawdown')
            axes[1].grid(True, alpha=0.3)
            
            # 3. Positions
            if 'pos' in df.columns:
                axes[2].plot(df.index, df['pos'], label='Position')
                axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            axes[2].set_title('Position')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # 4. Price with signals
            if 'close' in df.columns and 'pos' in df.columns:
                axes[3].plot(df.index, df['close'], label='Price', color='black', alpha=0.7)
                
                # Sample signals to avoid overcrowding
                sample_size = min(100, len(df))
                sample_indices = np.linspace(0, len(df)-1, sample_size).astype(int)
                sample_df = df.iloc[sample_indices]
                
                long_entries = sample_df[sample_df['pos'] > 0.1]
                short_entries = sample_df[sample_df['pos'] < -0.1]
                
                if not long_entries.empty:
                    axes[3].scatter(long_entries.index, long_entries['close'], 
                                  color='red', marker='^', s=30, label='Long')
                if not short_entries.empty:
                    axes[3].scatter(short_entries.index, short_entries['close'], 
                                  color='green', marker='v', s=30, label='Short')
                
            axes[3].set_title('Price & Signals')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Plot saved to {save_path}")
            
            # Show if requested and interactive display is available
            if show_plot:
                try:
                    plt.show()
                except Exception as e:
                    self.logger.warning(f"Could not display plot interactively: {str(e)}")
                    self.logger.info("Plot saved to file instead.")
            else:
                plt.close()  # Only close if not showing
            
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")  
        except Exception as e:
            self.logger.error(f"Error plotting CTA style results: {str(e)}")