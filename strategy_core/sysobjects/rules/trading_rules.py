"""
Core trading rules implementation
Based on pysystemtrade trading rules architecture
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from datetime import datetime
import warnings

from strategy_core.sysobjects.prices import AdjustedPrices
from strategy_core.sysobjects.forecasts import Forecast
from strategy_core.sysutils.math_algorithms import robust_vol_calc, ewmac_calc


class TradingRuleBase:
    """
    Base class for trading rules
    Similar to pysystemtrade rule structure
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.parameters = {}
    
    def __call__(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """
        Execute the trading rule
        
        Parameters:
        -----------
        price_data: pd.Series
            Price data for the instrument
        **kwargs: dict
            Strategy-specific parameters
            
        Returns:
        --------
        pd.Series
            Raw forecast values
        """
        raise NotImplementedError("Subclasses must implement __call__")
    
    def validate_parameters(self, **kwargs):
        """Validate rule parameters"""
        pass
    
    def get_data_requirements(self) -> List[str]:
        """Get list of required data series"""
        return ['price']


class EWMACRule(TradingRuleBase):
    """
    Exponential Weighted Moving Average Crossover
    The core trend-following rule from pysystemtrade
    """
    
    def __init__(self, Lfast: int = 16, Lslow: int = 64, vol_days: int = 35):
        super().__init__("EWMAC", "Exponential Weighted Moving Average Crossover")
        self.Lfast = Lfast
        self.Lslow = Lslow
        self.vol_days = vol_days
    
    def __call__(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """
        Calculate EWMAC forecast
        
        Parameters:
        -----------
        price_data: pd.Series
            Price data
        **kwargs: dict
            Optional parameters to override defaults
            
        Returns:
        --------
        pd.Series
            EWMAC forecast
        """
        # Get parameters
        Lfast = kwargs.get('Lfast', self.Lfast)
        Lslow = kwargs.get('Lslow', self.Lslow)
        vol_days = kwargs.get('vol_days', self.vol_days)
        
        # Validate parameters
        self.validate_parameters(Lfast=Lfast, Lslow=Lslow, vol_days=vol_days)
        
        # Calculate EWMAC
        return ewmac_calc(price_data, Lfast, Lslow, vol_days)
    
    def validate_parameters(self, **kwargs):
        """Validate EWMAC parameters"""
        Lfast = kwargs.get('Lfast', self.Lfast)
        Lslow = kwargs.get('Lslow', self.Lslow)
        vol_days = kwargs.get('vol_days', self.vol_days)
        
        if Lfast >= Lslow:
            raise ValueError("Fast period must be less than slow period")
        if Lfast <= 0 or Lslow <= 0:
            raise ValueError("Periods must be positive")
        if vol_days <= 0:
            raise ValueError("Volatility days must be positive")


class BreakoutRule(TradingRuleBase):
    """
    Breakout trading rule
    Based on pysystemtrade breakout implementation
    """
    
    def __init__(self, lookback: int = 20, smooth: int = None):
        super().__init__("Breakout", "Breakout trading rule")
        self.lookback = lookback
        self.smooth = smooth or max(1, lookback // 4)
    
    def __call__(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """
        Calculate breakout forecast
        
        Parameters:
        -----------
        price_data: pd.Series
            Price data
        **kwargs: dict
            Optional parameters
            
        Returns:
        --------
        pd.Series
            Breakout forecast
        """
        # Get parameters
        lookback = kwargs.get('lookback', self.lookback)
        smooth = kwargs.get('smooth', self.smooth)
        
        # Validate parameters
        self.validate_parameters(lookback=lookback, smooth=smooth)
        
        # Calculate breakout
        return self._calculate_breakout(price_data, lookback, smooth)
    
    def _calculate_breakout(self, price_data: pd.Series, lookback: int, smooth: int) -> pd.Series:
        """Calculate breakout forecast"""
        # Calculate rolling high and low
        rolling_high = price_data.rolling(window=lookback).max()
        rolling_low = price_data.rolling(window=lookback).min()
        
        # Calculate midpoint and range
        midpoint = (rolling_high + rolling_low) / 2
        range_val = rolling_high - rolling_low
        
        # Avoid division by zero
        range_val = range_val.replace(0, np.nan)
        
        # Calculate breakout signal
        raw_signal = 40 * (price_data - midpoint) / range_val
        
        # Apply smoothing
        if smooth > 1:
            raw_signal = raw_signal.ewm(span=smooth).mean()
        
        return raw_signal
    
    def validate_parameters(self, **kwargs):
        """Validate breakout parameters"""
        lookback = kwargs.get('lookback', self.lookback)
        smooth = kwargs.get('smooth', self.smooth)
        
        if lookback <= 0:
            raise ValueError("Lookback period must be positive")
        if smooth <= 0:
            raise ValueError("Smooth period must be positive")


class MomentumRule(TradingRuleBase):
    """
    Simple momentum rule
    Based on price momentum over specified period
    """
    
    def __init__(self, lookback: int = 20, vol_days: int = 35):
        super().__init__("Momentum", "Simple momentum rule")
        self.lookback = lookback
        self.vol_days = vol_days
    
    def __call__(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """
        Calculate momentum forecast
        
        Parameters:
        -----------
        price_data: pd.Series
            Price data
        **kwargs: dict
            Optional parameters
            
        Returns:
        --------
        pd.Series
            Momentum forecast
        """
        # Get parameters
        lookback = kwargs.get('lookback', self.lookback)
        vol_days = kwargs.get('vol_days', self.vol_days)
        
        # Validate parameters
        self.validate_parameters(lookback=lookback, vol_days=vol_days)
        
        # Calculate momentum
        return self._calculate_momentum(price_data, lookback, vol_days)
    
    def _calculate_momentum(self, price_data: pd.Series, lookback: int, vol_days: int) -> pd.Series:
        """Calculate momentum forecast"""
        # Calculate returns
        returns = price_data.pct_change(periods=lookback)
        
        # Calculate volatility
        vol = robust_vol_calc(price_data, vol_days)
        
        # Normalize by volatility
        vol_normalized = returns / vol * 100
        
        return vol_normalized
    
    def validate_parameters(self, **kwargs):
        """Validate momentum parameters"""
        lookback = kwargs.get('lookback', self.lookback)
        vol_days = kwargs.get('vol_days', self.vol_days)
        
        if lookback <= 0:
            raise ValueError("Lookback period must be positive")
        if vol_days <= 0:
            raise ValueError("Volatility days must be positive")


class AccelerationRule(TradingRuleBase):
    """
    Acceleration rule - rate of change of EWMAC
    Based on pysystemtrade acceleration implementation
    """
    
    def __init__(self, Lfast: int = 4, vol_days: int = 35):
        super().__init__("Acceleration", "Momentum acceleration rule")
        self.Lfast = Lfast
        self.vol_days = vol_days
    
    def __call__(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """
        Calculate acceleration forecast
        
        Parameters:
        -----------
        price_data: pd.Series
            Price data
        **kwargs: dict
            Optional parameters
            
        Returns:
        --------
        pd.Series
            Acceleration forecast
        """
        # Get parameters
        Lfast = kwargs.get('Lfast', self.Lfast)
        vol_days = kwargs.get('vol_days', self.vol_days)
        
        # Validate parameters
        self.validate_parameters(Lfast=Lfast, vol_days=vol_days)
        
        # Calculate acceleration
        return self._calculate_acceleration(price_data, Lfast, vol_days)
    
    def _calculate_acceleration(self, price_data: pd.Series, Lfast: int, vol_days: int) -> pd.Series:
        """Calculate acceleration forecast"""
        # Calculate base EWMAC signal
        Lslow = Lfast * 4  # Standard ratio
        ewmac_signal = ewmac_calc(price_data, Lfast, Lslow, vol_days)
        
        # Calculate acceleration (rate of change)
        acceleration = ewmac_signal - ewmac_signal.shift(Lfast)
        
        return acceleration
    
    def validate_parameters(self, **kwargs):
        """Validate acceleration parameters"""
        Lfast = kwargs.get('Lfast', self.Lfast)
        vol_days = kwargs.get('vol_days', self.vol_days)
        
        if Lfast <= 0:
            raise ValueError("Fast period must be positive")
        if vol_days <= 0:
            raise ValueError("Volatility days must be positive")


class MeanReversionRule(TradingRuleBase):
    """
    Mean reversion rule
    Fades extreme moves
    """
    
    def __init__(self, lookback: int = 20, threshold: float = 2.0, vol_days: int = 35):
        super().__init__("MeanReversion", "Mean reversion rule")
        self.lookback = lookback
        self.threshold = threshold
        self.vol_days = vol_days
    
    def __call__(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """
        Calculate mean reversion forecast
        
        Parameters:
        -----------
        price_data: pd.Series
            Price data
        **kwargs: dict
            Optional parameters
            
        Returns:
        --------
        pd.Series
            Mean reversion forecast
        """
        # Get parameters
        lookback = kwargs.get('lookback', self.lookback)
        threshold = kwargs.get('threshold', self.threshold)
        vol_days = kwargs.get('vol_days', self.vol_days)
        
        # Validate parameters
        self.validate_parameters(lookback=lookback, threshold=threshold, vol_days=vol_days)
        
        # Calculate mean reversion
        return self._calculate_mean_reversion(price_data, lookback, threshold, vol_days)
    
    def _calculate_mean_reversion(self, price_data: pd.Series, lookback: int, 
                                threshold: float, vol_days: int) -> pd.Series:
        """Calculate mean reversion forecast"""
        # Calculate returns
        returns = price_data.pct_change()
        
        # Calculate rolling mean and std
        rolling_mean = returns.rolling(window=lookback).mean()
        rolling_std = returns.rolling(window=lookback).std()
        
        # Calculate z-score
        z_score = (returns - rolling_mean) / rolling_std
        
        # Apply threshold - only trade when z-score exceeds threshold
        signal = np.where(abs(z_score) > threshold, -z_score * 10, 0)
        
        return pd.Series(signal, index=price_data.index)
    
    def validate_parameters(self, **kwargs):
        """Validate mean reversion parameters"""
        lookback = kwargs.get('lookback', self.lookback)
        threshold = kwargs.get('threshold', self.threshold)
        vol_days = kwargs.get('vol_days', self.vol_days)
        
        if lookback <= 0:
            raise ValueError("Lookback period must be positive")
        if threshold <= 0:
            raise ValueError("Threshold must be positive")
        if vol_days <= 0:
            raise ValueError("Volatility days must be positive")


class CarryRule(TradingRuleBase):
    """
    Carry rule for instruments with yield data
    Based on pysystemtrade carry implementation
    """
    
    def __init__(self, smooth_days: int = 90):
        super().__init__("Carry", "Carry trading rule")
        self.smooth_days = smooth_days
    
    def __call__(self, carry_data: pd.Series, **kwargs) -> pd.Series:
        """
        Calculate carry forecast
        
        Parameters:
        -----------
        carry_data: pd.Series
            Carry data (yield differential, etc.)
        **kwargs: dict
            Optional parameters
            
        Returns:
        --------
        pd.Series
            Carry forecast
        """
        # Get parameters
        smooth_days = kwargs.get('smooth_days', self.smooth_days)
        
        # Validate parameters
        self.validate_parameters(smooth_days=smooth_days)
        
        # Calculate carry
        return self._calculate_carry(carry_data, smooth_days)
    
    def _calculate_carry(self, carry_data: pd.Series, smooth_days: int) -> pd.Series:
        """Calculate carry forecast"""
        # Smooth the carry data
        if smooth_days > 1:
            smoothed_carry = carry_data.ewm(span=smooth_days).mean()
        else:
            smoothed_carry = carry_data
        
        # Scale to forecast units (multiply by 30 as per pysystemtrade)
        forecast = smoothed_carry * 30
        
        return forecast
    
    def validate_parameters(self, **kwargs):
        """Validate carry parameters"""
        smooth_days = kwargs.get('smooth_days', self.smooth_days)
        
        if smooth_days <= 0:
            raise ValueError("Smooth days must be positive")
    
    def get_data_requirements(self) -> List[str]:
        """Get list of required data series"""
        return ['carry']


class RelativeMomentumRule(TradingRuleBase):
    """
    Relative momentum rule
    Momentum relative to asset class or benchmark
    """
    
    def __init__(self, horizon: int = 40, ewma_span: int = 10):
        super().__init__("RelativeMomentum", "Relative momentum rule")
        self.horizon = horizon
        self.ewma_span = ewma_span
    
    def __call__(self, price_data: pd.Series, benchmark_data: pd.Series = None, **kwargs) -> pd.Series:
        """
        Calculate relative momentum forecast
        
        Parameters:
        -----------
        price_data: pd.Series
            Price data for the instrument
        benchmark_data: pd.Series
            Benchmark price data (optional)
        **kwargs: dict
            Optional parameters
            
        Returns:
        --------
        pd.Series
            Relative momentum forecast
        """
        # Get parameters
        horizon = kwargs.get('horizon', self.horizon)
        ewma_span = kwargs.get('ewma_span', self.ewma_span)
        
        # Validate parameters
        self.validate_parameters(horizon=horizon, ewma_span=ewma_span)
        
        # Calculate relative momentum
        return self._calculate_relative_momentum(price_data, benchmark_data, horizon, ewma_span)
    
    def _calculate_relative_momentum(self, price_data: pd.Series, benchmark_data: pd.Series,
                                   horizon: int, ewma_span: int) -> pd.Series:
        """Calculate relative momentum forecast"""
        # Calculate returns
        instrument_returns = price_data.pct_change(periods=horizon)
        
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change(periods=horizon)
            # Relative performance
            relative_returns = instrument_returns - benchmark_returns
        else:
            # Use absolute returns if no benchmark
            relative_returns = instrument_returns
        
        # Smooth the signal
        if ewma_span > 1:
            smoothed_signal = relative_returns.ewm(span=ewma_span).mean()
        else:
            smoothed_signal = relative_returns
        
        # Scale to forecast units
        forecast = smoothed_signal * 100
        
        return forecast
    
    def validate_parameters(self, **kwargs):
        """Validate relative momentum parameters"""
        horizon = kwargs.get('horizon', self.horizon)
        ewma_span = kwargs.get('ewma_span', self.ewma_span)
        
        if horizon <= 0:
            raise ValueError("Horizon must be positive")
        if ewma_span <= 0:
            raise ValueError("EWMA span must be positive")
    
    def get_data_requirements(self) -> List[str]:
        """Get list of required data series"""
        return ['price', 'benchmark']


class VolatilityRule(TradingRuleBase):
    """
    Volatility-based trading rule
    Trades based on volatility regime changes
    """
    
    def __init__(self, vol_lookback: int = 20, signal_lookback: int = 5):
        super().__init__("Volatility", "Volatility-based trading rule")
        self.vol_lookback = vol_lookback
        self.signal_lookback = signal_lookback
    
    def __call__(self, price_data: pd.Series, **kwargs) -> pd.Series:
        """
        Calculate volatility forecast
        
        Parameters:
        -----------
        price_data: pd.Series
            Price data
        **kwargs: dict
            Optional parameters
            
        Returns:
        --------
        pd.Series
            Volatility forecast
        """
        # Get parameters
        vol_lookback = kwargs.get('vol_lookback', self.vol_lookback)
        signal_lookback = kwargs.get('signal_lookback', self.signal_lookback)
        
        # Validate parameters
        self.validate_parameters(vol_lookback=vol_lookback, signal_lookback=signal_lookback)
        
        # Calculate volatility signal
        return self._calculate_volatility_signal(price_data, vol_lookback, signal_lookback)
    
    def _calculate_volatility_signal(self, price_data: pd.Series, vol_lookback: int, 
                                   signal_lookback: int) -> pd.Series:
        """Calculate volatility-based forecast"""
        # Calculate returns
        returns = price_data.pct_change()
        
        # Calculate rolling volatility
        vol = returns.rolling(window=vol_lookback).std()
        
        # Calculate volatility momentum
        vol_momentum = vol.pct_change(periods=signal_lookback)
        
        # Generate signal (fade volatility spikes)
        signal = -vol_momentum * 50  # Negative because we fade vol spikes
        
        return signal
    
    def validate_parameters(self, **kwargs):
        """Validate volatility parameters"""
        vol_lookback = kwargs.get('vol_lookback', self.vol_lookback)
        signal_lookback = kwargs.get('signal_lookback', self.signal_lookback)
        
        if vol_lookback <= 0:
            raise ValueError("Volatility lookback must be positive")
        if signal_lookback <= 0:
            raise ValueError("Signal lookback must be positive")


def create_standard_trading_rules() -> Dict[str, TradingRuleBase]:
    """Create standard set of trading rules"""
    rules = {}
    
    # EWMAC rules (different speeds)
    rules['ewmac_2_8'] = EWMACRule(Lfast=2, Lslow=8)
    rules['ewmac_4_16'] = EWMACRule(Lfast=4, Lslow=16)
    rules['ewmac_8_32'] = EWMACRule(Lfast=8, Lslow=32)
    rules['ewmac_16_64'] = EWMACRule(Lfast=16, Lslow=64)
    rules['ewmac_32_128'] = EWMACRule(Lfast=32, Lslow=128)
    rules['ewmac_64_256'] = EWMACRule(Lfast=64, Lslow=256)
    
    # Breakout rules
    rules['breakout_10'] = BreakoutRule(lookback=10)
    rules['breakout_20'] = BreakoutRule(lookback=20)
    rules['breakout_40'] = BreakoutRule(lookback=40)
    rules['breakout_80'] = BreakoutRule(lookback=80)
    rules['breakout_160'] = BreakoutRule(lookback=160)
    rules['breakout_320'] = BreakoutRule(lookback=320)
    
    # Momentum rules
    rules['momentum_10'] = MomentumRule(lookback=10)
    rules['momentum_20'] = MomentumRule(lookback=20)
    rules['momentum_40'] = MomentumRule(lookback=40)
    rules['momentum_80'] = MomentumRule(lookback=80)
    
    # Other rules
    rules['acceleration_4'] = AccelerationRule(Lfast=4)
    rules['mean_reversion_20'] = MeanReversionRule(lookback=20)
    rules['carry_90'] = CarryRule(smooth_days=90)
    rules['relative_momentum_40'] = RelativeMomentumRule(horizon=40)
    rules['volatility_20'] = VolatilityRule(vol_lookback=20)
    
    return rules


def get_rule_config_template() -> Dict[str, Dict]:
    """Get template for rule configuration"""
    return {
        'ewmac_16_64': {
            'rule_class': 'EWMACRule',
            'parameters': {
                'Lfast': 16,
                'Lslow': 64,
                'vol_days': 35
            },
            'forecast_scalar': 7.5,
            'data_requirements': ['price']
        },
        'breakout_20': {
            'rule_class': 'BreakoutRule',
            'parameters': {
                'lookback': 20,
                'smooth': 5
            },
            'forecast_scalar': 1.0,
            'data_requirements': ['price']
        },
        'momentum_20': {
            'rule_class': 'MomentumRule',
            'parameters': {
                'lookback': 20,
                'vol_days': 35
            },
            'forecast_scalar': 2.0,
            'data_requirements': ['price']
        }
    }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    import pandas as pd
    import numpy as np
    
    # Generate sample price data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    
    # Test EWMAC rule
    ewmac_rule = EWMACRule(16, 64)
    ewmac_forecast = ewmac_rule(prices)
    print(f"EWMAC forecast stats: mean={ewmac_forecast.mean():.3f}, std={ewmac_forecast.std():.3f}")
    
    # Test Breakout rule
    breakout_rule = BreakoutRule(20)
    breakout_forecast = breakout_rule(prices)
    print(f"Breakout forecast stats: mean={breakout_forecast.mean():.3f}, std={breakout_forecast.std():.3f}")
    
    # Test Momentum rule
    momentum_rule = MomentumRule(20)
    momentum_forecast = momentum_rule(prices)
    print(f"Momentum forecast stats: mean={momentum_forecast.mean():.3f}, std={momentum_forecast.std():.3f}")
    
    print("All trading rules implemented successfully!")