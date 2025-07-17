"""
Risk management system
Based on pysystemtrade risk management functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings

try:
    from ..sysobjects.positions import Position, PositionSeries
    from ..sysobjects.instruments import Instrument, InstrumentList
    from ..sysobjects.prices import AdjustedPrices, MultiplePrices
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from sysobjects.positions import Position, PositionSeries
    from sysobjects.instruments import Instrument, InstrumentList
    from sysobjects.prices import AdjustedPrices, MultiplePrices


class RiskManager:
    """
    Comprehensive risk management system
    Similar to pysystemtrade risk management
    """
    
    def __init__(self, 
                 max_portfolio_leverage: float = 1.0,
                 max_instrument_weight: float = 0.5,
                 max_drawdown_threshold: float = 0.15,
                 volatility_target: float = 0.25,
                 lookback_days: int = 252):
        self.max_portfolio_leverage = max_portfolio_leverage
        self.max_instrument_weight = max_instrument_weight
        self.max_drawdown_threshold = max_drawdown_threshold
        self.volatility_target = volatility_target
        self.lookback_days = lookback_days
    
    def apply_risk_overlay(self, 
                          positions: PositionSeries,
                          prices: Dict[str, pd.Series],
                          instruments: InstrumentList) -> PositionSeries:
        """
        Apply comprehensive risk overlay to positions
        
        Parameters:
        -----------
        positions: PositionSeries
            Raw positions
        prices: Dict[str, pd.Series]
            Price data for each instrument
        instruments: InstrumentList
            Instrument metadata
            
        Returns:
        --------
        PositionSeries
            Risk-adjusted positions
        """
        # Start with original positions
        risk_adjusted_positions = positions
        
        # Apply leverage constraint
        risk_adjusted_positions = self._apply_leverage_constraint(risk_adjusted_positions)
        
        # Apply instrument weight constraints
        risk_adjusted_positions = self._apply_instrument_constraints(
            risk_adjusted_positions, instruments
        )
        
        # Apply drawdown control
        risk_adjusted_positions = self._apply_drawdown_control(
            risk_adjusted_positions, prices
        )
        
        # Apply volatility targeting
        risk_adjusted_positions = self._apply_volatility_targeting(
            risk_adjusted_positions, prices
        )
        
        return risk_adjusted_positions
    
    def _apply_leverage_constraint(self, positions: PositionSeries) -> PositionSeries:
        """Apply portfolio leverage constraint"""
        positions_df = positions.get_portfolio_positions()
        
        if positions_df.empty:
            return positions
        
        # Calculate gross leverage
        gross_leverage = positions_df.abs().sum(axis=1)
        
        # Scale positions if leverage exceeds limit
        excess_mask = gross_leverage > self.max_portfolio_leverage
        
        if excess_mask.any():
            scaling_factor = pd.Series(1.0, index=positions_df.index)
            scaling_factor[excess_mask] = (
                self.max_portfolio_leverage / gross_leverage[excess_mask]
            )
            
            # Apply scaling
            for col in positions_df.columns:
                positions_df[col] = positions_df[col] * scaling_factor
        
        # Convert back to PositionSeries
        adjusted_positions = {}
        for instrument in positions_df.columns:
            adjusted_positions[instrument] = Position(
                positions_df[instrument], 
                instrument=instrument
            )
        
        return PositionSeries(adjusted_positions)
    
    def _apply_instrument_constraints(self, 
                                    positions: PositionSeries,
                                    instruments: InstrumentList) -> PositionSeries:
        """Apply instrument-level constraints"""
        positions_df = positions.get_portfolio_positions()
        
        if positions_df.empty:
            return positions
        
        # Apply instrument weight constraints
        for instrument_name in positions_df.columns:
            positions_df[instrument_name] = positions_df[instrument_name].clip(
                -self.max_instrument_weight,
                self.max_instrument_weight
            )
        
        # Convert back to PositionSeries
        adjusted_positions = {}
        for instrument in positions_df.columns:
            adjusted_positions[instrument] = Position(
                positions_df[instrument], 
                instrument=instrument
            )
        
        return PositionSeries(adjusted_positions)
    
    def _apply_drawdown_control(self, 
                              positions: PositionSeries,
                              prices: Dict[str, pd.Series]) -> PositionSeries:
        """Apply drawdown-based position scaling"""
        # Calculate portfolio P&L
        portfolio_pnl = self._calculate_portfolio_pnl(positions, prices)
        
        if portfolio_pnl.empty:
            return positions
        
        # Calculate drawdown
        drawdown = self._calculate_drawdown(portfolio_pnl)
        
        # Calculate scaling factor based on drawdown
        scaling_factor = self._calculate_drawdown_scaling(drawdown)
        
        # Apply scaling to positions
        positions_df = positions.get_portfolio_positions()
        
        for col in positions_df.columns:
            aligned_scaling = scaling_factor.reindex(positions_df.index, method='ffill')
            positions_df[col] = positions_df[col] * aligned_scaling
        
        # Convert back to PositionSeries
        adjusted_positions = {}
        for instrument in positions_df.columns:
            adjusted_positions[instrument] = Position(
                positions_df[instrument], 
                instrument=instrument
            )
        
        return PositionSeries(adjusted_positions)
    
    def _apply_volatility_targeting(self, 
                                  positions: PositionSeries,
                                  prices: Dict[str, pd.Series]) -> PositionSeries:
        """Apply volatility targeting"""
        # Calculate portfolio volatility
        portfolio_returns = self._calculate_portfolio_returns(positions, prices)
        
        if portfolio_returns.empty:
            return positions
        
        # Calculate rolling volatility
        portfolio_vol = portfolio_returns.rolling(
            window=min(63, len(portfolio_returns))
        ).std() * np.sqrt(252)
        
        # Calculate volatility scaling
        vol_scaling = self.volatility_target / portfolio_vol
        vol_scaling = vol_scaling.clip(0.1, 3.0)  # Reasonable bounds
        
        # Apply scaling to positions
        positions_df = positions.get_portfolio_positions()
        
        for col in positions_df.columns:
            aligned_scaling = vol_scaling.reindex(positions_df.index, method='ffill')
            positions_df[col] = positions_df[col] * aligned_scaling
        
        # Convert back to PositionSeries
        adjusted_positions = {}
        for instrument in positions_df.columns:
            adjusted_positions[instrument] = Position(
                positions_df[instrument], 
                instrument=instrument
            )
        
        return PositionSeries(adjusted_positions)
    
    def _calculate_portfolio_pnl(self, 
                               positions: PositionSeries,
                               prices: Dict[str, pd.Series]) -> pd.Series:
        """Calculate portfolio P&L"""
        positions_df = positions.get_portfolio_positions()
        
        if positions_df.empty:
            return pd.Series()
        
        # Calculate P&L for each instrument
        portfolio_pnl = pd.Series(0.0, index=positions_df.index)
        
        for instrument in positions_df.columns:
            if instrument in prices:
                position_series = positions_df[instrument]
                price_series = prices[instrument]
                
                # Align position and price
                aligned_position, aligned_price = position_series.align(
                    price_series, method='ffill'
                )
                
                # Calculate returns
                price_returns = aligned_price.pct_change()
                
                # Calculate P&L contribution
                instrument_pnl = aligned_position.shift(1) * price_returns
                instrument_pnl = instrument_pnl.fillna(0)
                
                # Add to portfolio P&L
                portfolio_pnl = portfolio_pnl.add(instrument_pnl, fill_value=0)
        
        return portfolio_pnl
    
    def _calculate_portfolio_returns(self, 
                                   positions: PositionSeries,
                                   prices: Dict[str, pd.Series]) -> pd.Series:
        """Calculate portfolio returns"""
        return self._calculate_portfolio_pnl(positions, prices)
    
    def _calculate_drawdown(self, pnl: pd.Series) -> pd.Series:
        """Calculate drawdown from P&L series"""
        if pnl.empty:
            return pd.Series()
        
        # Calculate cumulative returns
        cum_returns = (1 + pnl).cumprod()
        
        # Calculate drawdown
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        
        return drawdown
    
    def _calculate_drawdown_scaling(self, drawdown: pd.Series) -> pd.Series:
        """Calculate scaling factor based on drawdown"""
        if drawdown.empty:
            return pd.Series(1.0, index=drawdown.index)
        
        # Simple linear scaling
        scaling = 1.0 - (drawdown.abs() / self.max_drawdown_threshold).clip(0, 1)
        
        # Ensure minimum scaling
        scaling = scaling.clip(0.1, 1.0)
        
        return scaling


class VolatilityTargeting:
    """
    Volatility targeting system
    """
    
    def __init__(self, 
                 target_volatility: float = 0.25,
                 vol_window: int = 63,
                 min_vol: float = 0.01,
                 max_vol: float = 2.0,
                 max_leverage: float = 3.0):
        self.target_volatility = target_volatility
        self.vol_window = vol_window
        self.min_vol = min_vol
        self.max_vol = max_vol
        self.max_leverage = max_leverage
    
    def calculate_vol_target_multiplier(self, 
                                      returns: pd.Series) -> pd.Series:
        """
        Calculate volatility targeting multiplier
        
        Parameters:
        -----------
        returns: pd.Series
            Returns series
            
        Returns:
        --------
        pd.Series
            Volatility targeting multiplier
        """
        if returns.empty:
            return pd.Series()
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(
            window=min(self.vol_window, len(returns))
        ).std() * np.sqrt(252)
        
        # Clip volatility to reasonable bounds
        rolling_vol = rolling_vol.clip(self.min_vol, self.max_vol)
        
        # Calculate multiplier
        multiplier = self.target_volatility / rolling_vol
        
        # Apply leverage constraint
        multiplier = multiplier.clip(1/self.max_leverage, self.max_leverage)
        
        return multiplier
    
    def apply_volatility_targeting(self, 
                                 positions: PositionSeries,
                                 prices: Dict[str, pd.Series]) -> PositionSeries:
        """Apply volatility targeting to positions"""
        # Calculate portfolio returns
        positions_df = positions.get_portfolio_positions()
        
        if positions_df.empty:
            return positions
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=positions_df.index)
        
        for instrument in positions_df.columns:
            if instrument in prices:
                position_series = positions_df[instrument]
                price_series = prices[instrument]
                
                # Align and calculate returns
                aligned_position, aligned_price = position_series.align(
                    price_series, method='ffill'
                )
                
                price_returns = aligned_price.pct_change()
                instrument_returns = aligned_position.shift(1) * price_returns
                
                portfolio_returns = portfolio_returns.add(instrument_returns, fill_value=0)
        
        # Calculate volatility targeting multiplier
        vol_multiplier = self.calculate_vol_target_multiplier(portfolio_returns)
        
        # Apply multiplier to positions
        adjusted_positions = {}
        for instrument in positions_df.columns:
            adjusted_series = positions_df[instrument] * vol_multiplier
            adjusted_positions[instrument] = Position(adjusted_series, instrument=instrument)
        
        return PositionSeries(adjusted_positions)


class CorrelationMonitor:
    """
    Correlation monitoring and adjustment
    """
    
    def __init__(self, 
                 correlation_window: int = 125,
                 max_correlation: float = 0.8,
                 min_correlation_periods: int = 20):
        self.correlation_window = correlation_window
        self.max_correlation = max_correlation
        self.min_correlation_periods = min_correlation_periods
    
    def calculate_rolling_correlations(self, 
                                     returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling correlations"""
        if returns.empty or returns.shape[1] < 2:
            return pd.DataFrame()
        
        # Calculate rolling correlations
        rolling_corr = returns.rolling(
            window=self.correlation_window,
            min_periods=self.min_correlation_periods
        ).corr()
        
        return rolling_corr
    
    def get_correlation_warnings(self, 
                               returns: pd.DataFrame) -> List[Dict]:
        """Get correlation warnings for high correlations"""
        correlations = self.calculate_rolling_correlations(returns)
        warnings = []
        
        if correlations.empty:
            return warnings
        
        # Check for high correlations
        for date in correlations.index.levels[0]:
            if date in correlations.index:
                corr_matrix = correlations.loc[date]
                
                # Find pairs with high correlation
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        instrument1 = corr_matrix.columns[i]
                        instrument2 = corr_matrix.columns[j]
                        correlation = corr_matrix.iloc[i, j]
                        
                        if abs(correlation) > self.max_correlation:
                            warnings.append({
                                'date': date,
                                'instrument1': instrument1,
                                'instrument2': instrument2,
                                'correlation': correlation,
                                'type': 'high_correlation'
                            })
        
        return warnings
    
    def calculate_diversification_multiplier(self, 
                                           correlations: pd.DataFrame) -> pd.Series:
        """Calculate diversification multiplier from correlations"""
        if correlations.empty:
            return pd.Series()
        
        # Simple approach: based on average correlation
        avg_correlations = []
        dates = []
        
        for date in correlations.index.levels[0]:
            if date in correlations.index:
                corr_matrix = correlations.loc[date]
                
                # Calculate average off-diagonal correlation
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                avg_corr = corr_matrix.values[mask].mean()
                
                avg_correlations.append(avg_corr)
                dates.append(date)
        
        avg_corr_series = pd.Series(avg_correlations, index=dates)
        
        # Calculate diversification multiplier
        # Higher correlation -> lower diversification
        div_multiplier = 1.0 / np.sqrt(1 + avg_corr_series * (len(correlations.columns) - 1))
        
        return div_multiplier


class RiskReporter:
    """
    Risk reporting and monitoring
    """
    
    def __init__(self):
        self.risk_metrics = {}
    
    def calculate_risk_metrics(self, 
                             positions: PositionSeries,
                             prices: Dict[str, pd.Series],
                             instruments: InstrumentList) -> Dict:
        """Calculate comprehensive risk metrics"""
        metrics = {}
        
        # Portfolio-level metrics
        positions_df = positions.get_portfolio_positions()
        
        if not positions_df.empty:
            # Leverage metrics
            metrics['gross_leverage'] = positions_df.abs().sum(axis=1).mean()
            metrics['net_leverage'] = positions_df.sum(axis=1).mean()
            
            # Concentration metrics
            metrics['max_instrument_weight'] = positions_df.abs().max().max()
            metrics['concentration_ratio'] = self._calculate_concentration_ratio(positions_df)
            
            # Turnover metrics
            turnover = positions.get_portfolio_turnover()
            metrics['average_turnover'] = turnover.mean()
            metrics['max_turnover'] = turnover.max()
            
            # Volatility metrics
            portfolio_returns = self._calculate_portfolio_returns(positions, prices)
            if not portfolio_returns.empty:
                metrics['portfolio_volatility'] = portfolio_returns.std() * np.sqrt(252)
                metrics['sharpe_ratio'] = (
                    portfolio_returns.mean() * 252 / (portfolio_returns.std() * np.sqrt(252))
                )
                
                # Drawdown metrics
                drawdown = self._calculate_drawdown(portfolio_returns)
                metrics['max_drawdown'] = drawdown.min()
                metrics['current_drawdown'] = drawdown.iloc[-1] if not drawdown.empty else 0
        
        # Instrument-level metrics
        metrics['instrument_metrics'] = {}
        for instrument_name in positions.get_instruments():
            position = positions.get_position(instrument_name)
            if position is not None:
                inst_metrics = position.get_position_statistics()
                metrics['instrument_metrics'][instrument_name] = inst_metrics
        
        return metrics
    
    def _calculate_concentration_ratio(self, positions_df: pd.DataFrame) -> float:
        """Calculate concentration ratio (top 3 instruments / total)"""
        if positions_df.empty:
            return 0.0
        
        # Calculate average absolute weights
        avg_weights = positions_df.abs().mean()
        
        # Calculate concentration ratio
        top_3_weight = avg_weights.nlargest(3).sum()
        total_weight = avg_weights.sum()
        
        return top_3_weight / total_weight if total_weight > 0 else 0.0
    
    def _calculate_portfolio_returns(self, 
                                   positions: PositionSeries,
                                   prices: Dict[str, pd.Series]) -> pd.Series:
        """Calculate portfolio returns"""
        positions_df = positions.get_portfolio_positions()
        
        if positions_df.empty:
            return pd.Series()
        
        portfolio_returns = pd.Series(0.0, index=positions_df.index)
        
        for instrument in positions_df.columns:
            if instrument in prices:
                position_series = positions_df[instrument]
                price_series = prices[instrument]
                
                # Align and calculate returns
                aligned_position, aligned_price = position_series.align(
                    price_series, method='ffill'
                )
                
                price_returns = aligned_price.pct_change()
                instrument_returns = aligned_position.shift(1) * price_returns
                
                portfolio_returns = portfolio_returns.add(instrument_returns, fill_value=0)
        
        return portfolio_returns
    
    def _calculate_drawdown(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown from returns"""
        if returns.empty:
            return pd.Series()
        
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        
        return drawdown
    
    def generate_risk_report(self, 
                           positions: PositionSeries,
                           prices: Dict[str, pd.Series],
                           instruments: InstrumentList) -> str:
        """Generate comprehensive risk report"""
        metrics = self.calculate_risk_metrics(positions, prices, instruments)
        
        report = ["=== RISK REPORT ===\n"]
        
        # Portfolio metrics
        report.append("Portfolio Metrics:")
        report.append(f"  Gross Leverage: {metrics.get('gross_leverage', 0):.2f}")
        report.append(f"  Net Leverage: {metrics.get('net_leverage', 0):.2f}")
        report.append(f"  Max Instrument Weight: {metrics.get('max_instrument_weight', 0):.2f}")
        report.append(f"  Concentration Ratio: {metrics.get('concentration_ratio', 0):.2f}")
        report.append(f"  Portfolio Volatility: {metrics.get('portfolio_volatility', 0):.2%}")
        report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"  Current Drawdown: {metrics.get('current_drawdown', 0):.2%}")
        
        # Instrument metrics summary
        report.append("\nInstrument Summary:")
        inst_metrics = metrics.get('instrument_metrics', {})
        for instrument, inst_metric in inst_metrics.items():
            report.append(f"  {instrument}:")
            report.append(f"    Avg Position: {inst_metric.get('average_position', 0):.2f}")
            report.append(f"    Total Trades: {inst_metric.get('total_trades', 0)}")
            report.append(f"    Turnover: {inst_metric.get('turnover', 0):.2f}")
        
        return "\n".join(report)


def create_sample_risk_system():
    """Create sample risk management system for testing"""
    from ..systems.portfolio import create_sample_portfolio_system
    
    # Create sample portfolio
    portfolio_system = create_sample_portfolio_system()
    
    # Create risk management components
    risk_manager = RiskManager()
    vol_targeting = VolatilityTargeting()
    corr_monitor = CorrelationMonitor()
    risk_reporter = RiskReporter()
    
    # Apply risk management
    risk_adjusted_positions = risk_manager.apply_risk_overlay(
        portfolio_system['positions'],
        portfolio_system['prices'],
        portfolio_system['instruments']
    )
    
    # Generate risk report
    risk_report = risk_reporter.generate_risk_report(
        risk_adjusted_positions,
        portfolio_system['prices'],
        portfolio_system['instruments']
    )
    
    return {
        'original_positions': portfolio_system['positions'],
        'risk_adjusted_positions': risk_adjusted_positions,
        'risk_manager': risk_manager,
        'vol_targeting': vol_targeting,
        'corr_monitor': corr_monitor,
        'risk_reporter': risk_reporter,
        'risk_report': risk_report
    }