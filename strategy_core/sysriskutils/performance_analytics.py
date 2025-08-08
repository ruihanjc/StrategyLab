"""
Performance analytics and reporting system
Based on pysystemtrade performance analysis functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings

# Configure matplotlib for plotting
import matplotlib
import matplotlib.pyplot as plt

# Set up plotting backend - try multiple options
def setup_matplotlib_backend():
    backends_to_try = ['Qt5Agg', 'TkAgg', 'GTK3Agg', 'Agg']
    
    for backend in backends_to_try:
        try:
            matplotlib.use(backend, force=True)
            print(f"Using matplotlib backend: {backend}")
            if backend != 'Agg':
                plt.ion()  # Interactive mode for non-Agg backends
            return backend
        except (ImportError, RuntimeError):
            continue
    
    # If no backend works, use default
    print("Warning: Using default matplotlib backend")
    return matplotlib.get_backend()

# Setup backend
current_backend = setup_matplotlib_backend()

def _show_or_save_plot(filename_prefix='plot'):
    """Show plot interactively"""
    try:
        # Force interactive display regardless of backend
        plt.show(block=False)  # Non-blocking show
        print(f"Interactive plot displayed: {filename_prefix}")
    except Exception as e:
        print(f"Could not display interactive plot: {e}")
        try:
            # Fallback: save file and try to open it
            plt.savefig(f'{filename_prefix}.png', dpi=150, bbox_inches='tight')
            print(f"Plot saved as {filename_prefix}.png - open this file to view")
            plt.close()
        except Exception as e2:
            print(f"Plot display failed: {e}, {e2}")

from ..sysobjects.positions import Position, PositionSeries
from ..sysobjects.costs import CostCalculator
from ..sysobjects.instruments import InstrumentList


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis
    Similar to pysystemtrade performance analytics
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 trading_days_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
    
    def analyze_performance(self, 
                          positions: PositionSeries,
                          prices: Dict[str, pd.Series],
                          costs: Dict[str, pd.Series] = None,
                          initial_capital: float = 1000000) -> Dict:
        """
        Comprehensive performance analysis
        
        Parameters:
        -----------
        positions: PositionSeries
            Portfolio positions
        prices: Dict[str, pd.Series]
            Price data
        costs: Dict[str, pd.Series]
            Transaction costs (optional)
        initial_capital: float
            Initial capital
            
        Returns:
        --------
        Dict
            Performance metrics
        """
        # Calculate returns
        returns = self._calculate_portfolio_returns(positions, prices, costs)
        
        if returns.empty:
            return {}
        
        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(returns, initial_capital)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_performance_metrics(returns, equity_curve)
        
        # Add position-specific metrics
        position_metrics = self._calculate_position_metrics(positions, prices)
        metrics.update(position_metrics)
        
        # Add cost analysis if provided
        if costs:
            cost_metrics = self._calculate_cost_metrics(costs)
            metrics.update(cost_metrics)
        
        # Calculate individual instrument performance
        individual_performance = self._calculate_individual_performance(positions, prices, costs)
        metrics['individual_performance'] = individual_performance
        
        return {
            'returns': returns,
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    
    def _calculate_portfolio_returns(self, 
                                   positions: PositionSeries,
                                   prices: Dict[str, pd.Series],
                                   costs: Dict[str, pd.Series] = None) -> pd.Series:
        """Calculate portfolio returns"""
        positions_df = positions.get_portfolio_positions()
        
        if positions_df.empty:
            return pd.Series()
        
        # Calculate returns for each instrument
        portfolio_returns = pd.Series(0.0, index=positions_df.index)
        
        for instrument in positions_df.columns:
            if instrument in prices:
                position_series = positions_df[instrument]
                price_series = prices[instrument]
                
                # Align data
                aligned_position, aligned_price = position_series.align(price_series)
                aligned_position = aligned_position.ffill()
                aligned_price = aligned_price.ffill()
                
                # Calculate price returns
                price_returns = aligned_price.pct_change()
                
                # Calculate P&L
                instrument_returns = aligned_position.shift(1) * price_returns
                instrument_returns = instrument_returns.fillna(0)
                
                # Subtract costs if provided
                if costs and instrument in costs:
                    cost_series = costs[instrument]
                    aligned_costs = cost_series.reindex(instrument_returns.index, fill_value=0)
                    instrument_returns -= aligned_costs
                
                # Add to portfolio
                portfolio_returns = portfolio_returns.add(instrument_returns, fill_value=0)
        
        return portfolio_returns
    
    def _calculate_equity_curve(self, 
                              returns: pd.Series, 
                              initial_capital: float) -> pd.Series:
        """Calculate equity curve"""
        if returns.empty:
            return pd.Series()
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Convert to equity curve
        equity_curve = initial_capital * cum_returns
        
        return equity_curve
    
    def _calculate_performance_metrics(self, 
                                     returns: pd.Series,
                                     equity_curve: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        if returns.empty:
            return {}
        
        # Return metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(self.trading_days_per_year)
        downside_volatility = self._calculate_downside_volatility(returns)
        
        # Risk-adjusted returns
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown metrics
        drawdown = self._calculate_drawdown(equity_curve)
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown.mean()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win/Loss metrics
        win_rate = (returns > 0).mean()
        loss_rate = (returns < 0).mean()
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Other metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Trade analysis
        trade_metrics = self._analyze_trades(returns)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'profit_factor': profit_factor,
            'skewness': skewness,
            'kurtosis': kurtosis,
            **trade_metrics
        }
    
    def _calculate_position_metrics(self, 
                                  positions: PositionSeries,
                                  prices: Dict[str, pd.Series]) -> Dict:
        """Calculate position-specific metrics"""
        metrics = {}
        
        # Portfolio-level position metrics
        positions_df = positions.get_portfolio_positions()
        
        if not positions_df.empty:
            # Leverage metrics
            gross_leverage = positions_df.abs().sum(axis=1)
            net_leverage = positions_df.sum(axis=1)
            
            metrics.update({
                'avg_gross_leverage': gross_leverage.mean(),
                'max_gross_leverage': gross_leverage.max(),
                'avg_net_leverage': net_leverage.mean(),
                'max_net_leverage': net_leverage.max(),
                'leverage_volatility': gross_leverage.std()
            })
            
            # Turnover metrics
            turnover = positions.get_portfolio_turnover()
            metrics.update({
                'avg_turnover': turnover.mean(),
                'max_turnover': turnover.max(),
                'total_turnover': turnover.sum()
            })
            
            # Concentration metrics
            concentration = self._calculate_concentration_metrics(positions_df)
            metrics.update(concentration)
        
        # Instrument-level metrics
        instrument_metrics = {}
        for instrument in positions.get_instruments():
            position = positions.get_position(instrument)
            if position is not None:
                inst_stats = position.get_position_statistics()
                instrument_metrics[instrument] = inst_stats
        
        metrics['instrument_metrics'] = instrument_metrics
        
        return metrics
    
    def _calculate_cost_metrics(self, costs: Dict[str, pd.Series]) -> Dict:
        """Calculate cost-related metrics"""
        if not costs:
            return {}
        
        # Total costs
        total_costs = pd.DataFrame(costs).sum(axis=1)
        
        # Annualized costs
        annualized_costs = total_costs.sum() * (self.trading_days_per_year / len(total_costs))
        
        return {
            'total_costs': total_costs.sum(),
            'annualized_costs': annualized_costs,
            'avg_daily_costs': total_costs.mean(),
            'max_daily_costs': total_costs.max(),
            'cost_volatility': total_costs.std()
        }
    
    def _calculate_downside_volatility(self, returns: pd.Series) -> float:
        """Calculate downside volatility"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        
        return downside_returns.std() * np.sqrt(self.trading_days_per_year)
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        if equity_curve.empty:
            return pd.Series()
        
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown
    
    def _analyze_trades(self, returns: pd.Series) -> Dict:
        """Analyze individual trades"""
        if returns.empty:
            return {}
        
        # Find trade runs (consecutive periods of same sign)
        trade_runs = []
        current_run = []
        current_sign = None
        
        for ret in returns:
            if ret == 0:
                continue
            
            ret_sign = 1 if ret > 0 else -1
            
            if current_sign is None or ret_sign == current_sign:
                current_run.append(ret)
                current_sign = ret_sign
            else:
                if current_run:
                    trade_runs.append(sum(current_run))
                current_run = [ret]
                current_sign = ret_sign
        
        # Add final run
        if current_run:
            trade_runs.append(sum(current_run))
        
        if not trade_runs:
            return {}
        
        # Analyze trades
        winning_trades = [t for t in trade_runs if t > 0]
        losing_trades = [t for t in trade_runs if t < 0]
        
        return {
            'num_trades': len(trade_runs),
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades),
            'avg_winning_trade': np.mean(winning_trades) if winning_trades else 0,
            'avg_losing_trade': np.mean(losing_trades) if losing_trades else 0,
            'max_winning_trade': max(winning_trades) if winning_trades else 0,
            'max_losing_trade': min(losing_trades) if losing_trades else 0,
            'win_loss_ratio': (np.mean(winning_trades) / abs(np.mean(losing_trades))) 
                             if winning_trades and losing_trades else 0
        }
    
    def _calculate_concentration_metrics(self, positions_df: pd.DataFrame) -> Dict:
        """Calculate concentration metrics"""
        if positions_df.empty:
            return {}
        
        # Calculate average absolute weights
        avg_weights = positions_df.abs().mean()
        
        # Concentration ratio (top 3 / total)
        top_3_weight = avg_weights.nlargest(3).sum()
        total_weight = avg_weights.sum()
        concentration_ratio = top_3_weight / total_weight if total_weight > 0 else 0
        
        # Herfindahl index
        normalized_weights = avg_weights / total_weight if total_weight > 0 else avg_weights
        herfindahl_index = (normalized_weights ** 2).sum()
        
        return {
            'concentration_ratio': concentration_ratio,
            'herfindahl_index': herfindahl_index,
            'effective_instruments': 1.0 / herfindahl_index if herfindahl_index > 0 else 0
        }
    
    def _calculate_individual_performance(self, 
                                        positions: PositionSeries,
                                        prices: Dict[str, pd.Series],
                                        costs: Dict[str, pd.Series] = None) -> Dict:
        """Calculate performance metrics for each individual instrument"""
        individual_performance = {}
        positions_df = positions.get_portfolio_positions()
        
        for instrument in positions_df.columns:
            if instrument in prices:
                position_series = positions_df[instrument]
                price_series = prices[instrument]
                
                # Align data
                aligned_position, aligned_price = position_series.align(price_series)
                aligned_position = aligned_position.ffill()
                aligned_price = aligned_price.ffill()
                
                # Calculate price returns
                price_returns = aligned_price.pct_change()
                
                # Calculate P&L
                instrument_returns = aligned_position.shift(1) * price_returns
                instrument_returns = instrument_returns.fillna(0)
                
                # Subtract costs if provided
                if costs and instrument in costs:
                    cost_series = costs[instrument]
                    aligned_costs = cost_series.reindex(instrument_returns.index, fill_value=0)
                    instrument_returns = instrument_returns - aligned_costs
                
                # Calculate performance metrics for this instrument
                if not instrument_returns.empty and instrument_returns.std() > 0:
                    total_return = (1 + instrument_returns).prod() - 1
                    periods_per_year = 252  # Assuming daily data
                    annualized_return = (1 + total_return) ** (periods_per_year / len(instrument_returns)) - 1
                    volatility = instrument_returns.std() * np.sqrt(periods_per_year)
                    sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                    
                    # Calculate drawdown
                    cumulative_returns = (1 + instrument_returns).cumprod()
                    running_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns - running_max) / running_max
                    max_drawdown = drawdown.min()
                    
                    # Calculate win rate
                    winning_trades = (instrument_returns > 0).sum()
                    total_trades = (instrument_returns != 0).sum()
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0
                    
                    # Calculate profit factor
                    winning_returns = instrument_returns[instrument_returns > 0].sum()
                    losing_returns = abs(instrument_returns[instrument_returns < 0].sum())
                    profit_factor = winning_returns / losing_returns if losing_returns > 0 else np.inf
                    
                    individual_performance[instrument] = {
                        'total_return': total_return,
                        'annualized_return': annualized_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'num_trades': total_trades
                    }
                else:
                    # No meaningful returns for this instrument
                    individual_performance[instrument] = {
                        'total_return': 0.0,
                        'annualized_return': 0.0,
                        'volatility': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'win_rate': 0.0,
                        'profit_factor': 0.0,
                        'num_trades': 0
                    }
        
        return individual_performance


class PerformanceReporter:
    """
    Performance reporting functionality
    """
    
    def __init__(self, 
                 analyzer: PerformanceAnalyzer = None):
        self.analyzer = analyzer or PerformanceAnalyzer()
    
    def generate_performance_report(self, 
                                  positions: PositionSeries,
                                  prices: Dict[str, pd.Series],
                                  costs: Dict[str, pd.Series] = None,
                                  initial_capital: float = 1000000) -> str:
        """Generate comprehensive performance report"""
        
        # Analyze performance
        performance = self.analyzer.analyze_performance(
            positions, prices, costs, initial_capital
        )
        
        if not performance:
            return "No performance data available"
        
        metrics = performance['metrics']
        
        # Generate report
        report = ["=== PERFORMANCE REPORT ===\n"]
        
        # Return metrics
        report.append("RETURN METRICS:")
        report.append(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        report.append(f"  Volatility: {metrics.get('volatility', 0):.2%}")
        report.append(f"  Downside Volatility: {metrics.get('downside_volatility', 0):.2%}")
        
        # Risk-adjusted returns
        report.append("\nRISK-ADJUSTED RETURNS:")
        report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        
        # Drawdown metrics
        report.append("\nDRAWDOWN METRICS:")
        report.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"  Average Drawdown: {metrics.get('avg_drawdown', 0):.2%}")
        
        # Trade metrics
        report.append("\nTRADE METRICS:")
        report.append(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        report.append(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"  Number of Trades: {metrics.get('num_trades', 0)}")
        report.append(f"  Average Winning Trade: {metrics.get('avg_winning_trade', 0):.4f}")
        report.append(f"  Average Losing Trade: {metrics.get('avg_losing_trade', 0):.4f}")
        
        # Position metrics
        report.append("\nPOSITION METRICS:")
        report.append(f"  Average Gross Leverage: {metrics.get('avg_gross_leverage', 0):.2f}")
        report.append(f"  Maximum Gross Leverage: {metrics.get('max_gross_leverage', 0):.2f}")
        report.append(f"  Average Turnover: {metrics.get('avg_turnover', 0):.2f}")
        report.append(f"  Concentration Ratio: {metrics.get('concentration_ratio', 0):.2f}")
        
        # Cost metrics
        if 'total_costs' in metrics:
            report.append("\nCOST METRICS:")
            report.append(f"  Total Costs: {metrics.get('total_costs', 0):.2f}")
            report.append(f"  Annualized Costs: {metrics.get('annualized_costs', 0):.2f}")
            report.append(f"  Average Daily Costs: {metrics.get('avg_daily_costs', 0):.4f}")
        
        # Statistical metrics
        report.append("\nSTATISTICAL METRICS:")
        report.append(f"  Skewness: {metrics.get('skewness', 0):.2f}")
        report.append(f"  Kurtosis: {metrics.get('kurtosis', 0):.2f}")
        
        # Instrument breakdown
        inst_metrics = metrics.get('instrument_metrics', {})
        if inst_metrics:
            report.append("\nINSTRUMENT BREAKDOWN:")
            for instrument, inst_metric in inst_metrics.items():
                report.append(f"  {instrument}:")
                report.append(f"    Average Position: {inst_metric.get('average_position', 0):.2f}")
                report.append(f"    Total Trades: {inst_metric.get('total_trades', 0)}")
                report.append(f"    Turnover: {inst_metric.get('turnover', 0):.2f}")
        
        # Individual instrument performance
        individual_performance = metrics.get('individual_performance', {})
        if individual_performance:
            report.append("\n" + "=" * 60)
            report.append("INDIVIDUAL INSTRUMENT PERFORMANCE:")
            for instrument, perf in individual_performance.items():
                report.append(f"\n{instrument}:")
                report.append(f"  Total Return: {perf.get('total_return', 0):.2%}")
                report.append(f"  Annualized Return: {perf.get('annualized_return', 0):.2%}")
                report.append(f"  Volatility: {perf.get('volatility', 0):.2%}")
                report.append(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                report.append(f"  Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
                report.append(f"  Win Rate: {perf.get('win_rate', 0):.2%}")
                report.append(f"  Profit Factor: {perf.get('profit_factor', 0):.2f}")
                report.append(f"  Number of Trades: {perf.get('num_trades', 0)}")
        
        return "\n".join(report)
    
    def generate_summary_table(self, 
                             positions: PositionSeries,
                             prices: Dict[str, pd.Series],
                             costs: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """Generate summary table of key metrics"""
        
        performance = self.analyzer.analyze_performance(positions, prices, costs)
        
        if not performance:
            return pd.DataFrame()
        
        metrics = performance['metrics']
        
        # Key metrics for summary
        summary_metrics = {
            'Total Return': f"{metrics.get('total_return', 0):.2%}",
            'Annualized Return': f"{metrics.get('annualized_return', 0):.2%}",
            'Volatility': f"{metrics.get('volatility', 0):.2%}",
            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
            'Win Rate': f"{metrics.get('win_rate', 0):.2%}",
            'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}",
            'Average Leverage': f"{metrics.get('avg_gross_leverage', 0):.2f}",
            'Number of Trades': f"{metrics.get('num_trades', 0)}"
        }
        
        return pd.DataFrame(list(summary_metrics.items()), 
                          columns=['Metric', 'Value'])
    
    def plot_performance(self, 
                        positions: PositionSeries,
                        prices: Dict[str, pd.Series],
                        costs: Dict[str, pd.Series] = None):
        """Plot performance charts"""
        import matplotlib.pyplot as plt
        
        performance = self.analyzer.analyze_performance(positions, prices, costs)
        
        if not performance:
            print("No performance data available")
            return
        
        equity_curve = performance['equity_curve']
        returns = performance['returns']
        
        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Equity curve
        axes[0, 0].plot(equity_curve.index, equity_curve.values)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Equity')
        axes[0, 0].grid(True)
        
        # Drawdown
        drawdown = self.analyzer._calculate_drawdown(equity_curve)
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True)
        
        # Returns histogram
        axes[1, 0].hist(returns.values, bins=50, alpha=0.7)
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window=252).mean() / returns.rolling(window=252).std() * np.sqrt(252)
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[1, 1].set_title('Rolling Sharpe Ratio (1 Year)')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True)
        
        # Portfolio positions
        positions_df = positions.get_portfolio_positions()
        if not positions_df.empty:
            gross_leverage = positions_df.abs().sum(axis=1)
            axes[2, 0].plot(gross_leverage.index, gross_leverage.values)
            axes[2, 0].set_title('Gross Leverage')
            axes[2, 0].set_ylabel('Leverage')
            axes[2, 0].grid(True)
            
            # Turnover
            turnover = positions.get_portfolio_turnover()
            axes[2, 1].plot(turnover.index, turnover.values)
            axes[2, 1].set_title('Portfolio Turnover')
            axes[2, 1].set_ylabel('Turnover')
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        _show_or_save_plot('backtest_results')
        
        # Create individual ticker performance plots
        self.plot_individual_performance(positions, prices, costs)
    
    def plot_individual_performance(self,
                                  positions: PositionSeries,
                                  prices: Dict[str, pd.Series],
                                  costs: Dict[str, pd.Series] = None):
        """Plot individual ticker performance"""
        import matplotlib.pyplot as plt
        
        positions_df = positions.get_portfolio_positions()
        instruments = list(positions_df.columns)
        
        if not instruments:
            print("No instruments to plot")
            return
        
        # Calculate individual performance data
        individual_data = {}
        for instrument in instruments:
            if instrument in prices:
                position_series = positions_df[instrument]
                price_series = prices[instrument]
                
                # Align data
                aligned_position, aligned_price = position_series.align(price_series)
                aligned_position = aligned_position.ffill()
                aligned_price = aligned_price.ffill()
                
                # Calculate returns
                price_returns = aligned_price.pct_change()
                instrument_returns = aligned_position.shift(1) * price_returns
                instrument_returns = instrument_returns.fillna(0)
                
                # Subtract costs if provided
                if costs and instrument in costs:
                    cost_series = costs[instrument]
                    aligned_costs = cost_series.reindex(instrument_returns.index, fill_value=0)
                    instrument_returns = instrument_returns - aligned_costs
                
                # Calculate cumulative returns and positions
                cumulative_returns = (1 + instrument_returns).cumprod()
                
                individual_data[instrument] = {
                    'returns': instrument_returns,
                    'cumulative_returns': cumulative_returns,
                    'positions': aligned_position,
                    'prices': aligned_price
                }
        
        # Create plots
        n_instruments = len(individual_data)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Individual Ticker Performance', fontsize=16)
        
        # Plot 1: Cumulative Returns Comparison
        ax1 = axes[0, 0]
        for instrument, data in individual_data.items():
            ax1.plot(data['cumulative_returns'].index, data['cumulative_returns'].values, 
                    label=instrument, linewidth=2)
        ax1.set_title('Cumulative Returns by Ticker')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Price Performance
        ax2 = axes[0, 1]
        for instrument, data in individual_data.items():
            normalized_prices = data['prices'] / data['prices'].iloc[0]
            ax2.plot(normalized_prices.index, normalized_prices.values, 
                    label=instrument, linewidth=2)
        ax2.set_title('Normalized Price Performance')
        ax2.set_ylabel('Normalized Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position Sizes Over Time
        ax3 = axes[1, 0]
        for instrument, data in individual_data.items():
            ax3.plot(data['positions'].index, data['positions'].values, 
                    label=instrument, linewidth=2)
        ax3.set_title('Position Sizes Over Time')
        ax3.set_ylabel('Position Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rolling Sharpe Ratio (30-day window)
        ax4 = axes[1, 1]
        for instrument, data in individual_data.items():
            if not data['returns'].empty:
                rolling_sharpe = data['returns'].rolling(30).mean() / data['returns'].rolling(30).std() * np.sqrt(252)
                ax4.plot(rolling_sharpe.index, rolling_sharpe.values, 
                        label=f'{instrument} Sharpe', linewidth=2)
        ax4.set_title('30-Day Rolling Sharpe Ratio')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        _show_or_save_plot('individual_ticker_performance')
        
        # Create individual ticker comparison chart
        self.plot_ticker_comparison_chart(individual_data)
        
        # Create price chart
        self.plot_price_charts(prices)
        
        # Create combined backtest vs price chart
        self.plot_backtest_vs_prices(individual_data, prices)
    
    def plot_ticker_comparison_chart(self, individual_data: Dict):
        """Create a comparison chart for ticker performance metrics"""
        import matplotlib.pyplot as plt
        
        # Extract metrics for comparison
        metrics_data = {}
        for instrument, data in individual_data.items():
            returns = data['returns']
            if not returns.empty and returns.std() > 0:
                total_return = (1 + returns).prod() - 1
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = (returns.mean() * 252 - self.analyzer.risk_free_rate) / volatility if volatility > 0 else 0
                
                # Calculate max drawdown
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                
                metrics_data[instrument] = {
                    'Total Return': total_return * 100,
                    'Volatility': volatility * 100,
                    'Sharpe Ratio': sharpe_ratio,
                    'Max Drawdown': max_drawdown * 100
                }
        
        if not metrics_data:
            return
        
        # Create comparison bar charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Ticker Performance Comparison', fontsize=16)
        
        instruments = list(metrics_data.keys())
        metrics_names = list(next(iter(metrics_data.values())).keys())
        
        for i, metric in enumerate(metrics_names):
            ax = axes[i // 2, i % 2]
            values = [metrics_data[inst][metric] for inst in instruments]
            
            # Color bars based on performance
            colors = ['green' if v > 0 else 'red' for v in values] if metric in ['Total Return', 'Sharpe Ratio'] else ['red' if v < 0 else 'orange' for v in values]
            
            bars = ax.bar(instruments, values, color=colors, alpha=0.7)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(f'{metric} (%)' if 'Return' in metric or 'Volatility' in metric or 'Drawdown' in metric else metric)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(abs(min(values)), abs(max(values)))),
                       f'{value:.1f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        _show_or_save_plot('ticker_comparison')
    
    def plot_price_charts(self, prices: Dict[str, pd.Series]):
        """Plot raw price charts for all instruments"""
        import matplotlib.pyplot as plt
        
        if not prices:
            print("No price data to plot")
            return
        
        n_instruments = len(prices)
        
        # Create individual price charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Raw Price Charts', fontsize=16)
        
        instruments = list(prices.keys())
        
        # Plot 1: All prices on same chart (normalized)
        ax1 = axes[0, 0]
        for instrument, price_series in prices.items():
            normalized_prices = price_series / price_series.iloc[0]
            ax1.plot(normalized_prices.index, normalized_prices.values, 
                    label=instrument, linewidth=2)
        ax1.set_title('Normalized Price Comparison (Starting at 1.0)')
        ax1.set_ylabel('Normalized Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Price returns comparison
        ax2 = axes[0, 1]
        for instrument, price_series in prices.items():
            returns = price_series.pct_change().dropna()
            ax2.plot(returns.index, returns.values, 
                    label=f'{instrument} Returns', linewidth=1, alpha=0.7)
        ax2.set_title('Daily Returns Comparison')
        ax2.set_ylabel('Daily Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 3: Rolling volatility (30-day)
        ax3 = axes[1, 0]
        for instrument, price_series in prices.items():
            returns = price_series.pct_change().dropna()
            rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100  # Annualized %
            ax3.plot(rolling_vol.index, rolling_vol.values, 
                    label=f'{instrument} Vol', linewidth=2)
        ax3.set_title('30-Day Rolling Volatility (Annualized %)')
        ax3.set_ylabel('Volatility (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative returns
        ax4 = axes[1, 1]
        for instrument, price_series in prices.items():
            returns = price_series.pct_change().dropna()
            cumulative_returns = (1 + returns).cumprod()
            ax4.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=f'{instrument} Cum Return', linewidth=2)
        ax4.set_title('Cumulative Returns (Buy & Hold)')
        ax4.set_ylabel('Cumulative Return')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        _show_or_save_plot('price_analysis')
        
        # Create individual price subplots
        fig2, axes2 = plt.subplots(len(instruments), 1, figsize=(14, 4*len(instruments)))
        fig2.suptitle('Individual Price Charts', fontsize=16)
        
        if len(instruments) == 1:
            axes2 = [axes2]
        
        for i, (instrument, price_series) in enumerate(prices.items()):
            axes2[i].plot(price_series.index, price_series.values, linewidth=2, color='blue')
            axes2[i].set_title(f'{instrument} Price Chart')
            axes2[i].set_ylabel('Price ($)')
            axes2[i].grid(True, alpha=0.3)
            
            # Add basic statistics as text
            returns = price_series.pct_change().dropna()
            total_return = (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100
            
            stats_text = f'Total Return: {total_return:.2f}%\nVolatility: {volatility:.2f}%'
            axes2[i].text(0.02, 0.98, stats_text, transform=axes2[i].transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        _show_or_save_plot('individual_price_charts')
    
    def plot_backtest_vs_prices(self, individual_data: Dict, prices: Dict[str, pd.Series]):
        """Plot backtest performance vs underlying price movements with candlesticks if possible"""
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        if not individual_data or not prices:
            print("No data available for backtest vs price comparison")
            return
        
        n_instruments = len(individual_data)
        fig, axes = plt.subplots(n_instruments, 1, figsize=(16, 6*n_instruments))
        fig.suptitle('Backtest Performance vs Price Movement', fontsize=16)
        
        if n_instruments == 1:
            axes = [axes]
        
        for i, (instrument, data) in enumerate(individual_data.items()):
            if instrument not in prices:
                continue
                
            ax = axes[i]
            
            # Create secondary y-axis for prices
            ax2 = ax.twinx()
            
            # Plot cumulative strategy returns
            strategy_returns = data['cumulative_returns']
            ax.plot(strategy_returns.index, strategy_returns.values, 
                   label=f'{instrument} Strategy Returns', linewidth=3, color='red', alpha=0.8)
            
            # Plot buy & hold returns
            price_series = prices[instrument]
            buy_hold_returns = price_series / price_series.iloc[0]
            ax2.plot(buy_hold_returns.index, buy_hold_returns.values, 
                    label=f'{instrument} Buy & Hold', linewidth=2, color='blue', alpha=0.6)
            
            # Plot positions as filled area
            positions = data['positions']
            ax_pos = ax.twinx()
            ax_pos.spines['right'].set_position(('outward', 60))
            ax_pos.fill_between(positions.index, 0, positions.values, 
                              alpha=0.3, color='green', label=f'{instrument} Position')
            
            # Formatting
            ax.set_title(f'{instrument}: Strategy vs Buy & Hold Performance')
            ax.set_ylabel('Strategy Cumulative Return', color='red')
            ax2.set_ylabel('Buy & Hold Return (Normalized)', color='blue')
            ax_pos.set_ylabel('Position Size', color='green')
            
            ax.tick_params(axis='y', labelcolor='red')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax_pos.tick_params(axis='y', labelcolor='green')
            
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Break-even')
            ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            
            # Add performance statistics
            strategy_total_return = (strategy_returns.iloc[-1] - 1) * 100
            buyhold_total_return = (buy_hold_returns.iloc[-1] - 1) * 100
            alpha = strategy_total_return - buyhold_total_return
            
            stats_text = f'Strategy: {strategy_total_return:.1f}%\nBuy & Hold: {buyhold_total_return:.1f}%\nAlpha: {alpha:.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Format x-axis
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            
        plt.tight_layout()
        _show_or_save_plot('backtest_vs_prices')
        
    def plot_candlestick_with_signals(self, individual_data: Dict, ohlc_data: Dict = None):
        """Plot candlestick charts with trading signals if OHLC data is available"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        if not individual_data:
            print("No individual data available for candlestick plotting")
            return
            
        # Check if we have OHLC data, otherwise use close prices to simulate
        if ohlc_data is None:
            print("No OHLC data provided, using close prices for price visualization")
            return
        
        n_instruments = len(individual_data)
        fig, axes = plt.subplots(n_instruments, 1, figsize=(16, 8*n_instruments))
        fig.suptitle('Price Action with Trading Signals', fontsize=16)
        
        if n_instruments == 1:
            axes = [axes]
        
        for i, (instrument, data) in enumerate(individual_data.items()):
            if instrument not in ohlc_data:
                continue
                
            ax = axes[i]
            ohlc = ohlc_data[instrument]
            positions = data['positions']
            
            # Simple candlestick representation using rectangles
            for idx, (date, row) in enumerate(ohlc.iterrows()):
                if idx >= len(ohlc) - 1:
                    continue
                    
                open_price = row.get('open', row.get('close', 0))
                high_price = row.get('high', row.get('close', 0))
                low_price = row.get('low', row.get('close', 0))
                close_price = row.get('close', 0)
                
                # Determine color
                color = 'green' if close_price >= open_price else 'red'
                
                # Draw high-low line
                ax.plot([idx, idx], [low_price, high_price], color='black', linewidth=1)
                
                # Draw open-close rectangle
                height = abs(close_price - open_price)
                bottom = min(open_price, close_price)
                rect = Rectangle((idx-0.3, bottom), 0.6, height, 
                               facecolor=color, alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
            
            # Add position signals
            ax2 = ax.twinx()
            ax2.plot(range(len(positions)), positions.values, 
                    color='blue', linewidth=2, alpha=0.7, label='Position Size')
            ax2.set_ylabel('Position Size', color='blue')
            
            # Highlight entry/exit points
            position_changes = positions.diff().fillna(0)
            entries = position_changes > 0.01  # Significant position increase
            exits = position_changes < -0.01   # Significant position decrease
            
            if entries.any():
                entry_points = entries[entries].index
                for entry in entry_points:
                    if entry in ohlc.index:
                        entry_idx = ohlc.index.get_loc(entry)
                        entry_price = ohlc.loc[entry, 'close']
                        ax.scatter(entry_idx, entry_price, color='green', s=100, marker='^', 
                                 alpha=0.8, label='Long Entry' if 'Long Entry' not in [t.get_text() for t in ax.get_legend().get_texts() if ax.get_legend()] else "")
            
            if exits.any():
                exit_points = exits[exits].index
                for exit in exit_points:
                    if exit in ohlc.index:
                        exit_idx = ohlc.index.get_loc(exit)
                        exit_price = ohlc.loc[exit, 'close']
                        ax.scatter(exit_idx, exit_price, color='red', s=100, marker='v', 
                                 alpha=0.8, label='Exit' if 'Exit' not in [t.get_text() for t in ax.get_legend().get_texts() if ax.get_legend()] else "")
            
            ax.set_title(f'{instrument} - Price Action with Trading Signals')
            ax.set_ylabel('Price ($)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        _show_or_save_plot('candlestick_signals')


def create_sample_performance_analysis():
    """Create sample performance analysis for testing"""
    from ..systems.portfolio import create_sample_portfolio_system
    from ..sysobjects.costs import create_cost_models
    
    # Create sample portfolio system
    portfolio_system = create_sample_portfolio_system()
    
    # Create cost calculator
    cost_models = create_cost_models()
    cost_calculator = cost_models['equity']
    
    # Calculate costs
    costs = {}
    for instrument in portfolio_system['positions'].get_instruments():
        position = portfolio_system['positions'].get_position(instrument)
        price = portfolio_system['prices'][instrument]
        
        if position is not None and not price.empty:
            instrument_costs = cost_calculator.calculate_position_costs(
                position, price, instrument
            )
            costs[instrument] = instrument_costs
    
    # Create performance analyzer
    analyzer = PerformanceAnalyzer()
    reporter = PerformanceReporter(analyzer)
    
    # Analyze performance
    performance = analyzer.analyze_performance(
        portfolio_system['positions'],
        portfolio_system['prices'],
        costs
    )
    
    # Generate report
    report = reporter.generate_performance_report(
        portfolio_system['positions'],
        portfolio_system['prices'],
        costs
    )
    
    return {
        'performance': performance,
        'report': report,
        'analyzer': analyzer,
        'reporter': reporter,
        'costs': costs
    }