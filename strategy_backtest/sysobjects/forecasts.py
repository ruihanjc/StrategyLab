"""
Forecast data structures for backtesting
Based on pysystemtrade forecast handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
import warnings


class Forecast(pd.Series):
    """
    Single forecast series with validation and scaling
    Similar to pysystemtrade forecasts
    """
    
    DEFAULT_FORECAST_CAP = 20.0
    
    def __init__(self, data, forecast_cap: float = None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        
        self.forecast_cap = forecast_cap or self.DEFAULT_FORECAST_CAP
        
        # Ensure index is datetime
        if not isinstance(self.index, pd.DatetimeIndex):
            if hasattr(self.index, 'dtype') and self.index.dtype == 'object':
                self.index = pd.to_datetime(self.index)
        
        # Validate and process forecast
        self._validate_forecast()
    
    def _validate_forecast(self):
        """Validate forecast data"""
        if self.empty:
            return
        
        # Check for extreme values
        if abs(self).max() > self.forecast_cap * 2:
            warnings.warn(f"Forecast values exceed 2x cap ({self.forecast_cap})")
        
        # Check for missing values
        if self.isna().any():
            warnings.warn("Missing values detected in forecast data")
    
    def cap_forecast(self, cap: float = None) -> 'Forecast':
        """Cap forecast values"""
        if cap is None:
            cap = self.forecast_cap
        
        capped = self.clip(-cap, cap)
        return Forecast(capped, forecast_cap=cap)
    
    def scale_forecast(self, target_volatility: float = 0.25) -> 'Forecast':
        """Scale forecast to target volatility"""
        if self.empty:
            return self
        
        # Calculate forecast volatility
        forecast_vol = abs(self).rolling(window=32).mean()
        
        # Avoid division by zero
        forecast_vol = forecast_vol.replace(0, np.nan)
        
        # Scale to target volatility
        scaling_factor = target_volatility / forecast_vol
        scaled = self * scaling_factor
        
        return Forecast(scaled, forecast_cap=self.forecast_cap)
    
    def smooth_forecast(self, window: int = 5) -> 'Forecast':
        """Apply smoothing to forecast"""
        smoothed = self.rolling(window=window, center=True).mean()
        return Forecast(smoothed, forecast_cap=self.forecast_cap)
    
    def get_signal_strength(self) -> pd.Series:
        """Get signal strength (absolute value normalized)"""
        return abs(self) / self.forecast_cap
    
    def get_position_from_forecast(self, volatility_scalar: float = 1.0) -> pd.Series:
        """Convert forecast to position"""
        # Simple linear mapping: position = forecast / forecast_cap
        return self / self.forecast_cap * volatility_scalar
    
    def get_data_for_period(self, start_date: Union[str, datetime], 
                           end_date: Union[str, datetime] = None) -> 'Forecast':
        """Get forecast for specific period"""
        if end_date is None:
            end_date = self.index[-1]
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        mask = (self.index >= start_date) & (self.index <= end_date)
        return Forecast(self[mask], forecast_cap=self.forecast_cap)


class ForecastCombination:
    """
    Combines multiple forecasts with weights
    Similar to pysystemtrade forecast combination
    """
    
    def __init__(self, forecasts: Dict[str, Forecast], 
                 weights: Dict[str, float] = None,
                 forecast_cap: float = 20.0):
        self.forecasts = forecasts
        self.forecast_cap = forecast_cap
        
        # Default equal weights
        if weights is None:
            n_forecasts = len(forecasts)
            weights = {name: 1.0 / n_forecasts for name in forecasts.keys()}
        
        self.weights = weights
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate forecast weights"""
        # Check that all forecasts have weights
        forecast_names = set(self.forecasts.keys())
        weight_names = set(self.weights.keys())
        
        if forecast_names != weight_names:
            raise ValueError("Forecast names and weight names must match")
        
        # Check weights sum to 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            warnings.warn(f"Weights sum to {total_weight}, not 1.0")
    
    def get_combined_forecast(self, 
                            apply_cap: bool = True,
                            apply_diversification_multiplier: bool = True) -> Forecast:
        """Combine forecasts using weights"""
        if not self.forecasts:
            raise ValueError("No forecasts to combine")
        
        # Align all forecasts to common index
        common_index = self._get_common_index()
        aligned_forecasts = {}
        
        for name, forecast in self.forecasts.items():
            aligned_forecasts[name] = forecast.reindex(common_index, method='ffill')
        
        # Calculate weighted combination
        combined = pd.Series(0.0, index=common_index)
        
        for name, forecast in aligned_forecasts.items():
            weight = self.weights[name]
            combined += weight * forecast
        
        # Apply diversification multiplier
        if apply_diversification_multiplier:
            div_mult = self._calculate_diversification_multiplier()
            combined *= div_mult
        
        # Create forecast object
        result = Forecast(combined, forecast_cap=self.forecast_cap)
        
        # Apply cap if requested
        if apply_cap:
            result = result.cap_forecast()
        
        return result
    
    def _get_common_index(self) -> pd.DatetimeIndex:
        """Get common index for all forecasts"""
        if not self.forecasts:
            return pd.DatetimeIndex([])
        
        # Find intersection of all indices
        common_index = None
        for forecast in self.forecasts.values():
            if common_index is None:
                common_index = forecast.index
            else:
                common_index = common_index.intersection(forecast.index)
        
        return common_index.sort_values()
    
    def _calculate_diversification_multiplier(self) -> float:
        """Calculate diversification multiplier"""
        # Simple approach: sqrt(number of forecasts)
        # More sophisticated would use correlation matrix
        n_forecasts = len(self.forecasts)
        return np.sqrt(n_forecasts)
    
    def get_forecast_correlations(self) -> pd.DataFrame:
        """Calculate correlations between forecasts"""
        if len(self.forecasts) < 2:
            return pd.DataFrame()
        
        # Align forecasts
        common_index = self._get_common_index()
        aligned_data = {}
        
        for name, forecast in self.forecasts.items():
            aligned_data[name] = forecast.reindex(common_index, method='ffill')
        
        # Create DataFrame and calculate correlations
        forecast_df = pd.DataFrame(aligned_data)
        return forecast_df.corr()
    
    def get_forecast_contributions(self) -> pd.DataFrame:
        """Get contribution of each forecast to combined forecast"""
        common_index = self._get_common_index()
        contributions = {}
        
        for name, forecast in self.forecasts.items():
            weight = self.weights[name]
            aligned_forecast = forecast.reindex(common_index, method='ffill')
            contributions[name] = weight * aligned_forecast
        
        return pd.DataFrame(contributions)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update forecast weights"""
        self.weights = new_weights
        self._validate_weights()
    
    def add_forecast(self, name: str, forecast: Forecast, weight: float):
        """Add a new forecast"""
        self.forecasts[name] = forecast
        self.weights[name] = weight
        self._validate_weights()
    
    def remove_forecast(self, name: str):
        """Remove a forecast"""
        if name in self.forecasts:
            del self.forecasts[name]
        if name in self.weights:
            del self.weights[name]


def create_sample_forecast(instrument: str = "AAPL", 
                          rule_name: str = "ewmac",
                          start_date: str = "2020-01-01",
                          end_date: str = "2023-12-31") -> Forecast:
    """Create sample forecast data for testing"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic forecast data
    np.random.seed(42)
    n_periods = len(dates)
    
    # Generate trending forecast with noise
    trend = np.sin(np.arange(n_periods) * 2 * np.pi / 252) * 10  # Annual cycle
    noise = np.random.normal(0, 5, n_periods)
    forecast_values = trend + noise
    
    # Clip to reasonable range
    forecast_values = np.clip(forecast_values, -20, 20)
    
    forecast = pd.Series(forecast_values, index=dates)
    return Forecast(forecast)


def create_sample_forecast_combination() -> ForecastCombination:
    """Create sample forecast combination for testing"""
    # Create multiple forecasts
    forecasts = {
        'ewmac_fast': create_sample_forecast(rule_name='ewmac_fast'),
        'ewmac_slow': create_sample_forecast(rule_name='ewmac_slow'),
        'momentum': create_sample_forecast(rule_name='momentum'),
        'mean_reversion': create_sample_forecast(rule_name='mean_reversion')
    }
    
    # Define weights
    weights = {
        'ewmac_fast': 0.3,
        'ewmac_slow': 0.3,
        'momentum': 0.2,
        'mean_reversion': 0.2
    }
    
    return ForecastCombination(forecasts, weights)