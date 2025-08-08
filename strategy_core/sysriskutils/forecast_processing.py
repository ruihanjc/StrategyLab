"""
Forecast processing and combination system
Based on pysystemtrade forecast processing functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime
import warnings

from ..sysobjects.forecasts import Forecast, ForecastCombination
from ..sysobjects.instruments import Instrument, InstrumentList
from ..sysobjects.prices import AdjustedPrices


class ForecastScaler:
    """
    Forecast scaling functionality
    Similar to pysystemtrade forecast scaling
    """
    
    def __init__(self, 
                 target_abs_forecast: float = 10.0,
                 forecast_cap: float = 20.0,
                 min_periods: int = 500,
                 max_lookback: int = 2500):
        self.target_abs_forecast = target_abs_forecast
        self.forecast_cap = forecast_cap
        self.min_periods = min_periods
        self.max_lookback = max_lookback
    
    def scale_forecast(self, 
                      forecast: Forecast, 
                      prices: pd.Series = None) -> Forecast:
        """
        Scale forecast to target absolute forecast
        
        Parameters:
        -----------
        forecast: Forecast
            Raw forecast to scale
        prices: pd.Series
            Price data for calculating returns (optional)
            
        Returns:
        --------
        Forecast
            Scaled forecast
        """
        if forecast.empty:
            return forecast
        
        # Calculate scaling factor
        scaling_factor = self._calculate_scaling_factor(forecast, prices)
        
        # Apply scaling
        scaled_forecast = forecast * scaling_factor
        
        # Create new forecast object with cap
        return Forecast(scaled_forecast, forecast_cap=self.forecast_cap)
    
    def _calculate_scaling_factor(self, 
                                forecast: Forecast, 
                                prices: pd.Series = None) -> float:
        """Calculate scaling factor for forecast"""
        # Use recent periods for scaling calculation
        lookback_periods = min(self.max_lookback, len(forecast))
        recent_forecast = forecast.tail(lookback_periods)
        
        if len(recent_forecast) < self.min_periods:
            # Not enough data, use default scaling
            return 1.0
        
        # Calculate average absolute forecast
        avg_abs_forecast = recent_forecast.abs().mean()
        
        if avg_abs_forecast == 0:
            return 1.0
        
        # Calculate scaling factor
        scaling_factor = self.target_abs_forecast / avg_abs_forecast
        
        # Apply reasonable bounds
        scaling_factor = np.clip(scaling_factor, 0.1, 10.0)
        
        return scaling_factor
    
    def calculate_forecast_scalar(self, 
                                forecast: Forecast, 
                                prices: pd.Series) -> pd.Series:
        """
        Calculate time-varying forecast scalar
        
        Parameters:
        -----------
        forecast: Forecast
            Raw forecast
        prices: pd.Series
            Price data
            
        Returns:
        --------
        pd.Series
            Time-varying forecast scalar
        """
        if forecast.empty or prices.empty:
            return pd.Series()
        
        # Align forecast and prices
        aligned_forecast, aligned_prices = forecast.align(prices)
        aligned_forecast = aligned_forecast.ffill()
        aligned_prices = aligned_prices.ffill()
        
        # Calculate returns
        returns = aligned_prices.pct_change()
        
        # Calculate rolling forecast scalar
        scalar_series = pd.Series(1.0, index=aligned_forecast.index)
        
        for i in range(self.min_periods, len(aligned_forecast)):
            # Get window data
            window_forecast = aligned_forecast.iloc[max(0, i-self.max_lookback):i]
            window_returns = returns.iloc[max(0, i-self.max_lookback):i]
            
            # Calculate scalar
            scalar = self._calculate_scalar_from_windows(window_forecast, window_returns)
            scalar_series.iloc[i] = scalar
        
        return scalar_series
    
    def _calculate_scalar_from_windows(self, 
                                     forecast_window: pd.Series,
                                     returns_window: pd.Series) -> float:
        """Calculate scalar from forecast and returns windows"""
        if len(forecast_window) < self.min_periods:
            return 1.0
        
        # Calculate average absolute forecast
        avg_abs_forecast = forecast_window.abs().mean()
        
        if avg_abs_forecast == 0:
            return 1.0
        
        # Calculate target from returns volatility
        returns_vol = returns_window.std() * np.sqrt(252)
        target_forecast = self.target_abs_forecast * (returns_vol / 0.25)  # Scale by vol
        
        # Calculate scalar
        scalar = target_forecast / avg_abs_forecast
        
        return np.clip(scalar, 0.1, 10.0)


class ForecastCombiner:
    """
    Forecast combination functionality
    Enhanced version of pysystemtrade forecast combination
    """
    
    def __init__(self, 
                 forecast_cap: float = 20.0,
                 diversification_multiplier: float = 1.0,
                 correlation_window: int = 125,
                 min_correlation_periods: int = 20):
        self.forecast_cap = forecast_cap
        self.diversification_multiplier = diversification_multiplier
        self.correlation_window = correlation_window
        self.min_correlation_periods = min_correlation_periods
    
    def combine_forecasts(self, 
                         forecasts: Dict[str, Forecast],
                         weights: Dict[str, float] = None,
                         calculate_div_mult: bool = True) -> Forecast:
        """
        Combine multiple forecasts with weights
        
        Parameters:
        -----------
        forecasts: Dict[str, Forecast]
            Dictionary of forecasts by rule name
        weights: Dict[str, float]
            Weights for each forecast (optional, defaults to equal weights)
        calculate_div_mult: bool
            Whether to calculate diversification multiplier
            
        Returns:
        --------
        Forecast
            Combined forecast
        """
        if not forecasts:
            return Forecast(pd.Series())
        
        # Default to equal weights
        if weights is None:
            n_forecasts = len(forecasts)
            weights = {name: 1.0 / n_forecasts for name in forecasts.keys()}
        
        # Create combination object
        combination = ForecastCombination(forecasts, weights, self.forecast_cap)
        
        # Get combined forecast
        combined = combination.get_combined_forecast(
            apply_cap=False,  # Apply cap later
            apply_diversification_multiplier=False  # Calculate our own
        )
        
        # Calculate diversification multiplier if requested
        if calculate_div_mult:
            div_mult = self._calculate_diversification_multiplier(forecasts)
            combined = combined * div_mult
        
        # Apply cap - create new Forecast object if needed
        if hasattr(combined, 'cap_forecast'):
            combined = combined.cap_forecast(self.forecast_cap)
        else:
            # combined is a pandas Series, create Forecast object
            combined = Forecast(combined, forecast_cap=self.forecast_cap)
        
        return combined
    
    def _calculate_diversification_multiplier(self, 
                                           forecasts: Dict[str, Forecast]) -> float:
        """Calculate diversification multiplier based on correlations"""
        if len(forecasts) < 2:
            return 1.0
        
        # Calculate correlation matrix
        forecast_df = pd.DataFrame(forecasts)
        forecast_df = forecast_df.dropna()
        
        if len(forecast_df) < self.min_correlation_periods:
            # Not enough data, use simple sqrt approach
            return np.sqrt(len(forecasts))
        
        # Calculate correlation matrix
        corr_matrix = forecast_df.corr()
        
        # Calculate diversification multiplier
        # This is a simplified approach - more sophisticated methods exist
        n_forecasts = len(forecasts)
        avg_correlation = self._calculate_average_correlation(corr_matrix)
        
        # Diversification multiplier formula
        div_mult = n_forecasts / (1 + (n_forecasts - 1) * avg_correlation)
        div_mult = np.sqrt(div_mult)
        
        return np.clip(div_mult, 1.0, n_forecasts)
    
    def _calculate_average_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate average off-diagonal correlation"""
        if corr_matrix.empty:
            return 0.0
        
        # Get off-diagonal elements
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diagonal = corr_matrix.values[mask]
        
        # Calculate average, handling NaN values
        avg_corr = np.nanmean(off_diagonal)
        
        return avg_corr if not np.isnan(avg_corr) else 0.0
    
    def calculate_optimal_weights(self, 
                                forecasts: Dict[str, Forecast],
                                target_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate optimal forecast weights
        
        Parameters:
        -----------
        forecasts: Dict[str, Forecast]
            Dictionary of forecasts
        target_weights: Dict[str, float]
            Target weights (optional)
            
        Returns:
        --------
        Dict[str, float]
            Optimal weights
        """
        if not forecasts:
            return {}
        
        # Default to equal weights
        if target_weights is None:
            n_forecasts = len(forecasts)
            target_weights = {name: 1.0 / n_forecasts for name in forecasts.keys()}
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(forecasts)
        forecast_df = forecast_df.dropna()
        
        if len(forecast_df) < self.min_correlation_periods:
            # Not enough data, use target weights
            return target_weights
        
        # Calculate correlation matrix
        corr_matrix = forecast_df.corr()
        
        # Simple optimization: risk parity approach
        # More sophisticated optimization could be implemented
        optimal_weights = self._calculate_risk_parity_weights(corr_matrix)
        
        # Convert to dictionary
        weight_dict = {}
        for i, forecast_name in enumerate(corr_matrix.columns):
            weight_dict[forecast_name] = optimal_weights[i]
        
        return weight_dict
    
    def _calculate_risk_parity_weights(self, corr_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate risk parity weights"""
        n_assets = len(corr_matrix)
        
        if n_assets == 1:
            return np.array([1.0])
        
        # Simple risk parity: inverse of average correlation with others
        weights = np.zeros(n_assets)
        
        for i in range(n_assets):
            # Average correlation with other assets
            other_corrs = corr_matrix.iloc[i, :].drop(corr_matrix.columns[i])
            avg_corr = other_corrs.mean()
            
            # Weight is inverse of average correlation
            weights[i] = 1.0 / (1.0 + avg_corr)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights


class ForecastMapper:
    """
    Maps forecasts to positions
    Similar to pysystemtrade forecast mapping
    """
    
    def __init__(self, 
                 forecast_cap: float = 20.0,
                 position_multiplier: float = 1.0):
        self.forecast_cap = forecast_cap
        self.position_multiplier = position_multiplier
    
    def map_forecast_to_position(self, 
                               forecast: Forecast,
                               volatility_scalar: float = 1.0) -> pd.Series:
        """
        Map forecast to position
        
        Parameters:
        -----------
        forecast: Forecast
            Input forecast
        volatility_scalar: float
            Volatility scalar for position sizing
            
        Returns:
        --------
        pd.Series
            Position series
        """
        if forecast.empty:
            return pd.Series()
        
        # Simple linear mapping
        position = (forecast / self.forecast_cap) * volatility_scalar * self.position_multiplier
        
        return position
    
    def map_forecast_to_target_position(self, 
                                      forecast: Forecast,
                                      target_volatility: float = 0.25,
                                      instrument_volatility: pd.Series = None) -> pd.Series:
        """
        Map forecast to target position with volatility adjustment
        
        Parameters:
        -----------
        forecast: Forecast
            Input forecast
        target_volatility: float
            Target portfolio volatility
        instrument_volatility: pd.Series
            Instrument volatility for scaling
            
        Returns:
        --------
        pd.Series
            Target position series
        """
        if forecast.empty:
            return pd.Series()
        
        # Basic position from forecast
        base_position = forecast / self.forecast_cap
        
        # Apply volatility scaling if provided
        if instrument_volatility is not None:
            aligned_base, aligned_vol = base_position.align(instrument_volatility)
            aligned_base = aligned_base.ffill()
            aligned_vol = aligned_vol.ffill()
            
            # Volatility scalar
            vol_scalar = target_volatility / aligned_vol
            vol_scalar = vol_scalar.clip(0.1, 10.0)  # Reasonable bounds
            
            target_position = aligned_base * vol_scalar
        else:
            target_position = base_position
        
        return target_position * self.position_multiplier


class ForecastProcessor:
    """
    Main forecast processing pipeline
    Combines scaling, combination, and mapping
    """
    
    def __init__(self, 
                 scaler: ForecastScaler = None,
                 combiner: ForecastCombiner = None,
                 mapper: ForecastMapper = None):
        self.scaler = scaler or ForecastScaler()
        self.combiner = combiner or ForecastCombiner()
        self.mapper = mapper or ForecastMapper()
    
    def process_forecasts(self, 
                         raw_forecasts: Dict[str, Forecast],
                         prices: Dict[str, pd.Series],
                         forecast_weights: Dict[str, float] = None) -> Dict[str, Forecast]:
        """
        Process raw forecasts through complete pipeline
        
        Parameters:
        -----------
        raw_forecasts: Dict[str, Forecast]
            Raw forecasts by rule name
        prices: Dict[str, pd.Series]
            Price data for scaling
        forecast_weights: Dict[str, float]
            Weights for combining forecasts
            
        Returns:
        --------
        Dict[str, Forecast]
            Processed forecasts
        """
        processed_forecasts = {}
        
        # Step 1: Scale individual forecasts
        scaled_forecasts = {}
        for rule_name, forecast in raw_forecasts.items():
            # Get corresponding price data
            price_data = prices.get(rule_name.split('_')[0])  # Assume rule name contains instrument
            
            # Scale forecast
            scaled_forecast = self.scaler.scale_forecast(forecast, price_data)
            scaled_forecasts[rule_name] = scaled_forecast
        
        # Step 2: Combine forecasts if multiple rules
        if len(scaled_forecasts) > 1:
            combined_forecast = self.combiner.combine_forecasts(
                scaled_forecasts, forecast_weights
            )
            processed_forecasts['combined'] = combined_forecast
        else:
            # Single forecast
            processed_forecasts = scaled_forecasts
        
        return processed_forecasts
    
    def generate_positions_from_forecasts(self, 
                                        forecasts: Dict[str, Forecast],
                                        prices: Dict[str, pd.Series],
                                        target_volatility: float = 0.25) -> Dict[str, pd.Series]:
        """
        Generate positions from processed forecasts
        
        Parameters:
        -----------
        forecasts: Dict[str, Forecast]
            Processed forecasts
        prices: Dict[str, pd.Series]
            Price data
        target_volatility: float
            Target volatility for position sizing
            
        Returns:
        --------
        Dict[str, pd.Series]
            Position series by instrument
        """
        positions = {}
        
        for instrument_name, forecast in forecasts.items():
            # Get price data
            if instrument_name in prices:
                price_data = prices[instrument_name]
                
                # Calculate instrument volatility
                returns = price_data.pct_change()
                volatility = returns.rolling(window=25).std() * np.sqrt(252)
                
                # Map to position
                position = self.mapper.map_forecast_to_target_position(
                    forecast, target_volatility, volatility
                )
                
                positions[instrument_name] = position
        
        return positions


def create_sample_forecast_processing():
    """Create sample forecast processing system for testing"""
    from ..sysobjects.forecasts import create_sample_forecast_combination
    from ..sysobjects.prices import create_sample_price_data
    
    # Create sample data
    forecast_combination = create_sample_forecast_combination()
    price_data = create_sample_price_data()
    prices = {'AAPL': price_data.adjusted_prices('close')}
    
    # Create processing components
    scaler = ForecastScaler()
    combiner = ForecastCombiner()
    mapper = ForecastMapper()
    processor = ForecastProcessor(scaler, combiner, mapper)
    
    # Process forecasts
    raw_forecasts = forecast_combination.forecasts
    processed_forecasts = processor.process_forecasts(raw_forecasts, prices)
    
    # Generate positions
    positions = processor.generate_positions_from_forecasts(processed_forecasts, prices)
    
    return {
        'raw_forecasts': raw_forecasts,
        'processed_forecasts': processed_forecasts,
        'positions': positions,
        'scaler': scaler,
        'combiner': combiner,
        'mapper': mapper,
        'processor': processor
    }