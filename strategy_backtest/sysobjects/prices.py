"""
Price data structures for backtesting
Enhanced with pysystemtrade-style price handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import warnings


class AdjustedPrices(pd.Series):
    """
    Single price series with enhanced functionality
    Similar to pysystemtrade adjusted prices
    """
    
    def __init__(self, data, *args, **kwargs):
        # Ensure index is datetime
        if hasattr(data, 'index') and not isinstance(data.index, pd.DatetimeIndex):
            if data.index.dtype == 'object':
                data.index = pd.to_datetime(data.index)
        
        super().__init__(data, *args, **kwargs)
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate price data"""
        if self.empty:
            return
        
        # Check for negative prices
        if (self < 0).any():
            warnings.warn("Negative prices detected")
        
        # Check for missing values
        if self.isna().any():
            warnings.warn("Missing values detected in price data")
    
    def returns(self, periods: int = 1) -> pd.Series:
        """Calculate returns"""
        return self.pct_change(periods=periods)
    
    def log_returns(self, periods: int = 1) -> pd.Series:
        """Calculate log returns"""
        return np.log(self / self.shift(periods))
    
    def volatility(self, window: int = 20, annualize: bool = True) -> pd.Series:
        """Calculate rolling volatility"""
        returns = self.returns()
        vol = returns.rolling(window=window).std()
        if annualize:
            vol = vol * np.sqrt(252)  # Assuming daily data
        return vol
    
    def drawdown(self) -> pd.Series:
        """Calculate drawdown from running maximum"""
        running_max = self.cummax()
        return (self - running_max) / running_max
    
    def resample_to_frequency(self, frequency: str) -> 'AdjustedPrices':
        """Resample to different frequency"""
        resampled = self.resample(frequency).last()
        return AdjustedPrices(resampled)
    
    def get_data_for_period(self, start_date: Union[str, datetime], 
                           end_date: Union[str, datetime] = None) -> 'AdjustedPrices':
        """Get data for specific period"""
        if end_date is None:
            end_date = self.index[-1]
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        mask = (self.index >= start_date) & (self.index <= end_date)
        return AdjustedPrices(self[mask])
    
    def align_to_index(self, target_index: pd.DatetimeIndex, 
                      method: str = 'ffill') -> 'AdjustedPrices':
        """Align price series to target index"""
        aligned = self.reindex(target_index, method=method)
        return AdjustedPrices(aligned)


class MultiplePrices(pd.DataFrame):
    """
    Multiple price series (OHLCV) with enhanced functionality
    Similar to pysystemtrade multiple prices
    """
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']
    OPTIONAL_COLUMNS = ['volume', 'adjusted_close']
    
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        
        # Ensure index is datetime
        if not isinstance(self.index, pd.DatetimeIndex):
            if self.index.dtype == 'object':
                self.index = pd.to_datetime(self.index)
        
        # Validate structure
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate DataFrame structure"""
        if self.empty:
            return
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for logical consistency
        if not self.empty:
            # High should be >= Low
            if (self['high'] < self['low']).any():
                warnings.warn("High prices less than low prices detected")
            
            # Open and Close should be between High and Low
            if ((self['open'] > self['high']) | (self['open'] < self['low'])).any():
                warnings.warn("Open prices outside high-low range detected")
            
            if ((self['close'] > self['high']) | (self['close'] < self['low'])).any():
                warnings.warn("Close prices outside high-low range detected")
    
    def adjusted_prices(self, price_type: str = 'close') -> AdjustedPrices:
        """Get adjusted prices for specific price type"""
        if price_type not in self.columns:
            raise ValueError(f"Price type '{price_type}' not found in data")
        
        return AdjustedPrices(self[price_type])
    
    def typical_price(self) -> AdjustedPrices:
        """Calculate typical price (HLC/3)"""
        typical = (self['high'] + self['low'] + self['close']) / 3
        return AdjustedPrices(typical)
    
    def true_range(self) -> pd.Series:
        """Calculate true range"""
        prev_close = self['close'].shift(1)
        tr1 = self['high'] - self['low']
        tr2 = abs(self['high'] - prev_close)
        tr3 = abs(self['low'] - prev_close)
        
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def average_true_range(self, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        return self.true_range().rolling(window=window).mean()
    
    def returns(self, price_type: str = 'close', periods: int = 1) -> pd.Series:
        """Calculate returns for specific price type"""
        return self[price_type].pct_change(periods=periods)
    
    def volatility(self, price_type: str = 'close', window: int = 20, 
                  annualize: bool = True) -> pd.Series:
        """Calculate rolling volatility"""
        returns = self.returns(price_type)
        vol = returns.rolling(window=window).std()
        if annualize:
            vol = vol * np.sqrt(252)
        return vol
    
    def resample_to_frequency(self, frequency: str) -> 'MultiplePrices':
        """Resample to different frequency"""
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }
        
        # Add volume aggregation if present
        if 'volume' in self.columns:
            agg_dict['volume'] = 'sum'
        
        # Add other columns as last value
        for col in self.columns:
            if col not in agg_dict:
                agg_dict[col] = 'last'
        
        resampled = self.resample(frequency).agg(agg_dict)
        return MultiplePrices(resampled)
    
    def get_data_for_period(self, start_date: Union[str, datetime], 
                           end_date: Union[str, datetime] = None) -> 'MultiplePrices':
        """Get data for specific period"""
        if end_date is None:
            end_date = self.index[-1]
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        mask = (self.index >= start_date) & (self.index <= end_date)
        return MultiplePrices(self[mask])
    
    def align_to_index(self, target_index: pd.DatetimeIndex, 
                      method: str = 'ffill') -> 'MultiplePrices':
        """Align price data to target index"""
        aligned = self.reindex(target_index, method=method)
        return MultiplePrices(aligned)
    
    def add_technical_indicators(self):
        """Add common technical indicators"""
        # Simple moving averages
        self['sma_20'] = self['close'].rolling(window=20).mean()
        self['sma_50'] = self['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        self['ema_12'] = self['close'].ewm(span=12).mean()
        self['ema_26'] = self['close'].ewm(span=26).mean()
        
        # RSI
        self['rsi'] = self._calculate_rsi()
        
        # Bollinger Bands
        sma_20 = self['close'].rolling(window=20).mean()
        std_20 = self['close'].rolling(window=20).std()
        self['bb_upper'] = sma_20 + (std_20 * 2)
        self['bb_lower'] = sma_20 - (std_20 * 2)
        
        return self
    
    def _calculate_rsi(self, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = self['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def create_sample_price_data(instrument: str = "AAPL", 
                           start_date: str = "2020-01-01",
                           end_date: str = "2023-12-31") -> MultiplePrices:
    """Create sample price data for testing"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    n_periods = len(dates)
    
    # Start with base price
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_periods)  # Daily returns
    
    # Generate close prices
    close_prices = [base_price]
    for ret in returns[1:]:
        close_prices.append(close_prices[-1] * (1 + ret))
    
    close_prices = np.array(close_prices)
    
    # Generate OHLC from close prices
    high_mult = np.random.uniform(1.001, 1.05, n_periods)
    low_mult = np.random.uniform(0.95, 0.999, n_periods)
    open_mult = np.random.uniform(0.98, 1.02, n_periods)
    
    data = pd.DataFrame({
        'open': close_prices * open_mult,
        'high': close_prices * high_mult,
        'low': close_prices * low_mult,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, n_periods)
    }, index=dates)
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
    
    return MultiplePrices(data)