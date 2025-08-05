"""
Proper EWMAC implementation matching pysystemtrade's approach
Returns continuous forecasts, not discrete signals
"""

import pandas as pd
import numpy as np
from strategy_backtest.sysobjects.forecasts import Forecast


def robust_vol_calc(price_series, days=35):
    """
    Calculate robust volatility similar to pysystemtrade
    
    Parameters:
    -----------
    price_series: pd.Series
        Price series to calculate volatility for
    days: int
        Lookback period for volatility calculation
        
    Returns:
    --------
    pd.Series
        Volatility series
    """
    # Calculate price differences (returns in points, not percentage)
    price_diffs = price_series.diff()
    
    # Calculate rolling standard deviation of price differences
    vol = price_diffs.rolling(window=days, min_periods=10).std()
    
    # Use expanding window for early periods where we don't have enough data
    vol = vol.fillna(price_diffs.expanding(min_periods=2).std())
    
    return vol


def ewmac_calc_vol(price, Lfast=16, Lslow=64, vol_days=35):
    """
    Calculate EWMAC forecast with volatility calculation
    This is the main function that matches pysystemtrade's ewmac_forecast_with_defaults
    
    Parameters:
    -----------
    price: pd.Series
        Price series
    Lfast: int
        Fast EWMA lookback period
    Lslow: int  
        Slow EWMA lookback period
    vol_days: int
        Days for volatility calculation
        
    Returns:
    --------
    pd.Series
        Raw forecast values (continuous, not discrete signals)
    """
    # Calculate volatility
    vol = robust_vol_calc(price, days=vol_days)
    
    # Calculate EWMAC forecast
    forecast = ewmac(price, vol, Lfast, Lslow)
    
    return forecast


def ewmac(price, vol, Lfast, Lslow):
    """
    Core EWMAC calculation matching pysystemtrade
    
    Parameters:
    -----------
    price: pd.Series
        Price series
    vol: pd.Series
        Volatility series  
    Lfast: int
        Fast EWMA lookback
    Lslow: int
        Slow EWMA lookback
        
    Returns:
    --------
    pd.Series
        Raw forecast values (properly scaled to ~-10 to +10 range)
    """
    # Calculate exponentially weighted moving averages
    # Using pandas ewm with span parameter (equivalent to pysystemtrade's approach)
    fast_ewma = price.ewm(span=Lfast, min_periods=2).mean()
    slow_ewma = price.ewm(span=Lslow, min_periods=2).mean()
    
    # Calculate raw ewmac signal
    raw_ewmac = fast_ewma - slow_ewma
    
    # Normalize by volatility to get forecast
    # This is the key step that makes it volatility-adjusted
    forecast = raw_ewmac / vol
    
    # Handle any infinite or NaN values
    forecast = forecast.replace([np.inf, -np.inf], np.nan)
    forecast = forecast.fillna(0)
    
    # Apply scaling factor to get typical -10 to +10 range
    # This is similar to pysystemtrade's forecast scalar approach
    # Different EWMAC speeds need different scalars:
    # 16/64 EWMAC typically uses scalar around 7.5
    # 32/128 EWMAC typically uses scalar around 2.65
    
    # Calculate dynamic scalar based on lookback periods
    # Longer periods need higher scalars
    if Lslow <= 32:
        scalar = 5.3  # For fast EWMAC like 8/32
    elif Lslow <= 64:
        scalar = 7.5  # For medium EWMAC like 16/64  
    elif Lslow <= 128:
        scalar = 2.65 # For slow EWMAC like 32/128
    else:
        scalar = 1.5  # For very slow EWMAC like 64/256
    
    scaled_forecast = forecast * scalar
    
    # Apply forecast capping to keep within reasonable bounds (-20 to +20)
    # This matches pysystemtrade's forecast capping approach
    forecast_cap = 20.0
    capped_forecast = scaled_forecast.clip(-forecast_cap, forecast_cap)
    
    return capped_forecast


class ProperEWMAC:
    """
    Proper EWMAC trading rule that returns continuous forecasts
    This matches pysystemtrade's approach
    """
    
    def __init__(self, Lfast=16, Lslow=64, vol_days=35):
        """
        Initialize EWMAC rule
        
        Parameters:
        -----------
        Lfast: int
            Fast EWMA period
        Lslow: int  
            Slow EWMA period
        vol_days: int
            Volatility calculation period
        """
        self.Lfast = Lfast
        self.Lslow = Lslow
        self.vol_days = vol_days
        self.description = f"EWMAC({Lfast},{Lslow})"
    
    def get_data_requirements(self):
        """Get data requirements for this rule"""
        return ['price']  # Only requires price data
        
    def __call__(self, price_data, **kwargs):
        """
        Generate EWMAC forecast
        
        Parameters:
        -----------
        price_data: pd.Series or pd.DataFrame
            Price data (if DataFrame, uses 'close' column)
        **kwargs: dict
            Additional parameters (ignored for compatibility)
            
        Returns:
        --------
        Forecast
            Forecast object with continuous values
        """
        # Extract price series
        if isinstance(price_data, pd.DataFrame):
            if 'close' in price_data.columns:
                price = price_data['close']
            else:
                # Use first column if no 'close' column
                price = price_data.iloc[:, 0]
        else:
            price = price_data
            
        # Calculate EWMAC forecast
        forecast_values = ewmac_calc_vol(
            price, 
            Lfast=self.Lfast, 
            Lslow=self.Lslow, 
            vol_days=self.vol_days
        )
        
        # Return as Forecast object
        return Forecast(forecast_values)
    
    def generate_forecast(self, price_data, additional_data=None):
        """
        Alternative interface for generating forecasts
        
        Parameters:
        -----------
        price_data: pd.Series or pd.DataFrame
            Price data
        additional_data: dict
            Additional data (ignored)
            
        Returns:
        --------
        Forecast
            Forecast object
        """
        return self.__call__(price_data)


# Factory function for rule creation
def create_ewmac_rule(Lfast=16, Lslow=64, vol_days=35):
    """
    Factory function to create EWMAC rule
    
    Parameters:
    -----------
    Lfast: int
        Fast EWMA period
    Lslow: int
        Slow EWMA period  
    vol_days: int
        Volatility calculation period
        
    Returns:
    --------
    ProperEWMAC
        EWMAC rule instance
    """
    return ProperEWMAC(Lfast=Lfast, Lslow=Lslow, vol_days=vol_days)


# For backward compatibility and easy imports
def ewmac_16_64(price_data, **kwargs):
    """16/64 EWMAC rule"""
    rule = ProperEWMAC(Lfast=16, Lslow=64)
    return rule(price_data, **kwargs)


def ewmac_32_128(price_data, **kwargs):
    """32/128 EWMAC rule"""
    rule = ProperEWMAC(Lfast=32, Lslow=128)
    return rule(price_data, **kwargs)


def ewmac_8_32(price_data, **kwargs):
    """8/32 EWMAC rule"""
    rule = ProperEWMAC(Lfast=8, Lslow=32)
    return rule(price_data, **kwargs)