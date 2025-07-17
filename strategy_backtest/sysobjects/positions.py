"""
Position data structures for backtesting
Based on pysystemtrade position handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
import warnings


class Position(pd.Series):
    """
    Single position series with validation and analysis
    Similar to pysystemtrade positions
    """
    
    def __init__(self, data, instrument: str = None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        
        self.instrument = instrument
        
        # Ensure index is datetime
        if not isinstance(self.index, pd.DatetimeIndex):
            if hasattr(self.index, 'dtype') and self.index.dtype == 'object':
                self.index = pd.to_datetime(self.index)
        
        # Validate position data
        self._validate_position()
    
    def _validate_position(self):
        """Validate position data"""
        if self.empty:
            return
        
        # Check for extreme positions
        if abs(self).max() > 10:
            warnings.warn("Position sizes exceed 10 units")
        
        # Check for missing values
        if self.isna().any():
            warnings.warn("Missing values detected in position data")
    
    def get_position_changes(self) -> pd.Series:
        """Get position changes (trades)"""
        return self.diff().fillna(0)
    
    def get_trades(self) -> pd.Series:
        """Get non-zero position changes"""
        changes = self.get_position_changes()
        return changes[changes != 0]
    
    def get_trade_count(self) -> int:
        """Get total number of trades"""
        return len(self.get_trades())
    
    def get_turnover(self) -> float:
        """Calculate turnover (sum of absolute position changes)"""
        return abs(self.get_position_changes()).sum()
    
    def get_average_position(self) -> float:
        """Get average absolute position"""
        return abs(self).mean()
    
    def get_position_statistics(self) -> Dict:
        """Get comprehensive position statistics"""
        trades = self.get_trades()
        
        return {
            'total_trades': len(trades),
            'turnover': self.get_turnover(),
            'average_position': self.get_average_position(),
            'max_position': self.max(),
            'min_position': self.min(),
            'position_volatility': self.std(),
            'average_trade_size': abs(trades).mean() if len(trades) > 0 else 0,
            'max_trade_size': abs(trades).max() if len(trades) > 0 else 0,
            'long_percentage': (self > 0).mean() * 100,
            'short_percentage': (self < 0).mean() * 100,
            'flat_percentage': (self == 0).mean() * 100,
        }
    
    def apply_buffer(self, buffer_size: float = 0.1) -> 'Position':
        """Apply position buffering to reduce turnover"""
        if buffer_size <= 0:
            return Position(self, instrument=self.instrument)
        
        buffered = self.copy()
        
        # Apply buffer logic
        for i in range(1, len(buffered)):
            current_pos = buffered.iloc[i]
            prev_pos = buffered.iloc[i-1]
            
            # Only change position if move is larger than buffer
            if abs(current_pos - prev_pos) < buffer_size:
                buffered.iloc[i] = prev_pos
        
        return Position(buffered, instrument=self.instrument)
    
    def get_data_for_period(self, start_date: Union[str, datetime], 
                           end_date: Union[str, datetime] = None) -> 'Position':
        """Get position for specific period"""
        if end_date is None:
            end_date = self.index[-1]
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        mask = (self.index >= start_date) & (self.index <= end_date)
        return Position(self[mask], instrument=self.instrument)


class PositionSeries:
    """
    Container for multiple position series with portfolio-level analysis
    """
    
    def __init__(self, positions: Dict[str, Position] = None):
        self.positions = positions or {}
    
    def add_position(self, instrument: str, position: Position):
        """Add position for instrument"""
        self.positions[instrument] = position
    
    def get_position(self, instrument: str) -> Optional[Position]:
        """Get position for instrument"""
        return self.positions.get(instrument)
    
    def get_instruments(self) -> List[str]:
        """Get list of instruments"""
        return list(self.positions.keys())
    
    def get_common_index(self) -> pd.DatetimeIndex:
        """Get common index for all positions"""
        if not self.positions:
            return pd.DatetimeIndex([])
        
        # Find intersection of all indices
        common_index = None
        for position in self.positions.values():
            if common_index is None:
                common_index = position.index
            else:
                common_index = common_index.intersection(position.index)
        
        return common_index.sort_values()
    
    def get_portfolio_positions(self) -> pd.DataFrame:
        """Get all positions as DataFrame"""
        if not self.positions:
            return pd.DataFrame()
        
        # Align all positions to common index
        common_index = self.get_common_index()
        aligned_positions = {}
        
        for instrument, position in self.positions.items():
            aligned_positions[instrument] = position.reindex(common_index, method='ffill').fillna(0)
        
        return pd.DataFrame(aligned_positions)
    
    def get_portfolio_turnover(self) -> pd.Series:
        """Get portfolio-level turnover"""
        positions_df = self.get_portfolio_positions()
        if positions_df.empty:
            return pd.Series()
        
        # Calculate turnover for each instrument
        turnovers = {}
        for instrument in positions_df.columns:
            position_changes = positions_df[instrument].diff().fillna(0)
            turnovers[instrument] = abs(position_changes)
        
        turnover_df = pd.DataFrame(turnovers)
        return turnover_df.sum(axis=1)
    
    def get_portfolio_statistics(self) -> Dict:
        """Get portfolio-level position statistics"""
        if not self.positions:
            return {}
        
        positions_df = self.get_portfolio_positions()
        portfolio_turnover = self.get_portfolio_turnover()
        
        # Calculate statistics for each instrument
        instrument_stats = {}
        for instrument, position in self.positions.items():
            instrument_stats[instrument] = position.get_position_statistics()
        
        # Portfolio-level statistics
        portfolio_stats = {
            'total_instruments': len(self.positions),
            'total_portfolio_turnover': portfolio_turnover.sum(),
            'average_portfolio_turnover': portfolio_turnover.mean(),
            'max_portfolio_turnover': portfolio_turnover.max(),
            'instrument_statistics': instrument_stats
        }
        
        return portfolio_stats
    
    def apply_portfolio_buffer(self, buffer_size: float = 0.1) -> 'PositionSeries':
        """Apply buffering to all positions"""
        buffered_positions = {}
        
        for instrument, position in self.positions.items():
            buffered_positions[instrument] = position.apply_buffer(buffer_size)
        
        return PositionSeries(buffered_positions)
    
    def get_gross_exposure(self) -> pd.Series:
        """Get gross exposure (sum of absolute positions)"""
        positions_df = self.get_portfolio_positions()
        if positions_df.empty:
            return pd.Series()
        
        return positions_df.abs().sum(axis=1)
    
    def get_net_exposure(self) -> pd.Series:
        """Get net exposure (sum of positions)"""
        positions_df = self.get_portfolio_positions()
        if positions_df.empty:
            return pd.Series()
        
        return positions_df.sum(axis=1)
    
    def get_long_short_exposure(self) -> pd.DataFrame:
        """Get long and short exposure separately"""
        positions_df = self.get_portfolio_positions()
        if positions_df.empty:
            return pd.DataFrame()
        
        long_exposure = positions_df.where(positions_df > 0, 0).sum(axis=1)
        short_exposure = positions_df.where(positions_df < 0, 0).sum(axis=1)
        
        return pd.DataFrame({
            'long_exposure': long_exposure,
            'short_exposure': abs(short_exposure),
            'net_exposure': long_exposure + short_exposure
        })
    
    def get_data_for_period(self, start_date: Union[str, datetime], 
                           end_date: Union[str, datetime] = None) -> 'PositionSeries':
        """Get positions for specific period"""
        period_positions = {}
        
        for instrument, position in self.positions.items():
            period_positions[instrument] = position.get_data_for_period(start_date, end_date)
        
        return PositionSeries(period_positions)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return self.get_portfolio_positions()


def create_sample_position(instrument: str = "AAPL", 
                          start_date: str = "2020-01-01",
                          end_date: str = "2023-12-31") -> Position:
    """Create sample position data for testing"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic position data
    np.random.seed(42)
    n_periods = len(dates)
    
    # Generate trending position with occasional reversals
    base_position = 0.5  # Base long position
    trend = np.sin(np.arange(n_periods) * 2 * np.pi / 252) * 0.3  # Annual cycle
    noise = np.random.normal(0, 0.1, n_periods)
    
    # Add some regime changes
    regime_changes = np.random.random(n_periods) < 0.01  # 1% chance per day
    regime_multiplier = np.where(regime_changes, np.random.choice([-1, 1]), 1)
    regime_multiplier = np.cumprod(regime_multiplier)
    
    position_values = (base_position + trend + noise) * regime_multiplier
    
    # Clip to reasonable range
    position_values = np.clip(position_values, -2, 2)
    
    position = pd.Series(position_values, index=dates)
    return Position(position, instrument=instrument)


def create_sample_position_series() -> PositionSeries:
    """Create sample position series for testing"""
    instruments = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    positions = {}
    
    for instrument in instruments:
        positions[instrument] = create_sample_position(instrument)
    
    return PositionSeries(positions)