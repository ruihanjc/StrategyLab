"""
Performance analytics and reporting system
Based on pysystemtrade performance analysis functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings

try:
    from ..sysobjects.positions import Position, PositionSeries
    from ..sysobjects.costs import CostCalculator
    from ..sysobjects.instruments import InstrumentList
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from sysobjects.positions import Position, PositionSeries
    from sysobjects.costs import CostCalculator
    from sysobjects.instruments import InstrumentList


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
                aligned_position, aligned_price = position_series.align(
                    price_series, method='ffill'
                )
                
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
        plt.show()


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