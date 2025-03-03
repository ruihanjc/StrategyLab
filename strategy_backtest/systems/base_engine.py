import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Callable


class BacktestEngine:
    def __init__(self,
                 initial_capital: float = 1000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0):
        """
        Initialize the backtesting engine

        Parameters:
        -----------
        initial_capital: float
            Starting capital for backtesting
        commission: float
            Commission as a percentage of trade value (0.001 = 0.1%)
        slippage: float
            Slippage as a percentage of price (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def run(self,
            data: pd.DataFrame,
            strategy,
            price_column: str = 'close'):
        """
        Run backtest on historical data

        Parameters:
        -----------
        data: pd.DataFrame
            Historical data with datetime index
        strategy: Strategy
            Strategy object that generates signals
        price_column: str
            Column name for price data

        Returns:
        --------
        pd.DataFrame
            Performance metrics
        """
        # Generate signals
        signals = strategy.generate_signals(data)

        # Initialize results
        results = data.copy()
        results['signal'] = signals
        results['position'] = signals.shift(1).fillna(0)
        results['returns'] = results[price_column].pct_change()
        results['strategy_returns'] = results['position'] * results['returns']

        # Apply commission and slippage
        # Only apply when position changes
        position_changes = results['position'].diff().fillna(0)
        results['trade_costs'] = abs(position_changes) * (self.commission + self.slippage)
        results['strategy_returns'] = results['strategy_returns'] - results['trade_costs']

        # Calculate equity curve
        results['equity'] = self.initial_capital * (1 + results['strategy_returns']).cumprod()
        self.equity_curve = results['equity']

        # Calculate metrics
        metrics = self._calculate_metrics(results)

        return results, metrics

    def _calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        returns = results['strategy_returns'].dropna()

        # Return metrics
        total_return = self.equity_curve.iloc[-1] / self.initial_capital - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1

        # Risk metrics
        daily_drawdown = self.equity_curve / self.equity_curve.cummax() - 1
        max_drawdown = daily_drawdown.min()

        # Risk-adjusted returns
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0

        # Trade metrics
        position_changes = results['position'].diff().fillna(0)
        trades = position_changes[position_changes != 0]
        num_trades = len(trades)

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'num_trades': num_trades,
            'win_rate': 0.0,  # Requires more detailed trade tracking
            'profit_factor': 0.0,  # Requires more detailed trade tracking
        }

    def plot_equity_curve(self):
        """Plot equity curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.show()

    def plot_drawdown(self):
        """Plot drawdown curve"""
        drawdown = self.equity_curve / self.equity_curve.cummax() - 1
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.show()