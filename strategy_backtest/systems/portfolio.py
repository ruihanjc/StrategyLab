"""
Portfolio management system
Based on pysystemtrade portfolio functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import warnings

try:
    from ..sysobjects.instruments import Instrument, InstrumentList
    from ..sysobjects.positions import Position, PositionSeries
    from ..sysobjects.forecasts import Forecast, ForecastCombination
    from ..sysobjects.costs import CostCalculator, TradingCosts
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from sysobjects.instruments import Instrument, InstrumentList
    from sysobjects.positions import Position, PositionSeries
    from sysobjects.forecasts import Forecast, ForecastCombination
    from sysobjects.costs import CostCalculator, TradingCosts


class PortfolioOptimizer:
    """
    Portfolio optimization functionality
    Similar to pysystemtrade portfolio optimization
    """
    
    def __init__(self, 
                 correlation_window: int = 125,
                 min_correlation_periods: int = 20,
                 max_portfolio_leverage: float = 1.0):
        self.correlation_window = correlation_window
        self.min_correlation_periods = min_correlation_periods
        self.max_portfolio_leverage = max_portfolio_leverage
    
    def calculate_instrument_weights(self, 
                                   returns: pd.DataFrame,
                                   target_risk: float = 0.25) -> pd.DataFrame:
        """
        Calculate optimal instrument weights based on returns
        
        Parameters:
        -----------
        returns: pd.DataFrame
            Returns for each instrument
        target_risk: float
            Target portfolio risk (volatility)
            
        Returns:
        --------
        pd.DataFrame
            Optimal weights for each instrument over time
        """
        if returns.empty:
            return pd.DataFrame()
        
        # Calculate rolling correlations and volatilities
        correlations = self._calculate_rolling_correlations(returns)
        volatilities = self._calculate_rolling_volatilities(returns)
        
        # Calculate weights
        weights = {}
        
        for date in returns.index:
            if date in correlations.index:
                corr_matrix = correlations.loc[date]
                vol_vector = volatilities.loc[date]
                
                # Calculate optimal weights
                date_weights = self._optimize_weights(corr_matrix, vol_vector, target_risk)
                weights[date] = date_weights
        
        # Convert to DataFrame
        weights_df = pd.DataFrame(weights).T
        weights_df.index = pd.to_datetime(weights_df.index)
        
        return weights_df
    
    def _calculate_rolling_correlations(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate rolling correlation matrices"""
        correlations = {}
        
        for date in returns.index[self.correlation_window:]:
            window_returns = returns.loc[
                returns.index <= date
            ].tail(self.correlation_window)
            
            if len(window_returns) >= self.min_correlation_periods:
                corr_matrix = window_returns.corr()
                correlations[date] = corr_matrix
        
        return pd.Series(correlations)
    
    def _calculate_rolling_volatilities(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling volatilities"""
        return returns.rolling(window=self.correlation_window).std() * np.sqrt(252)
    
    def _optimize_weights(self, 
                         correlation_matrix: pd.DataFrame,
                         volatilities: pd.Series,
                         target_risk: float) -> pd.Series:
        """
        Optimize weights for given correlation matrix and volatilities
        
        Uses simple risk parity approach
        """
        if correlation_matrix.empty or volatilities.empty:
            return pd.Series()
        
        # Remove missing values
        valid_instruments = volatilities.dropna().index
        corr_matrix = correlation_matrix.loc[valid_instruments, valid_instruments]
        vol_vector = volatilities.loc[valid_instruments]
        
        if len(valid_instruments) == 0:
            return pd.Series()
        
        # Risk parity weights (inverse volatility)
        inv_vol = 1.0 / vol_vector
        weights = inv_vol / inv_vol.sum()
        
        # Scale to target risk
        portfolio_vol = self._calculate_portfolio_volatility(weights, corr_matrix, vol_vector)
        if portfolio_vol > 0:
            scaling_factor = target_risk / portfolio_vol
            weights = weights * scaling_factor
        
        # Apply leverage constraint
        total_leverage = weights.abs().sum()
        if total_leverage > self.max_portfolio_leverage:
            weights = weights * (self.max_portfolio_leverage / total_leverage)
        
        return weights
    
    def _calculate_portfolio_volatility(self, 
                                      weights: pd.Series,
                                      correlation_matrix: pd.DataFrame,
                                      volatilities: pd.Series) -> float:
        """Calculate portfolio volatility"""
        try:
            # Portfolio variance
            portfolio_var = np.dot(weights.values, 
                                 np.dot(correlation_matrix.values * 
                                       np.outer(volatilities.values, volatilities.values),
                                       weights.values))
            return np.sqrt(portfolio_var)
        except:
            return 0.0


class PositionSizer:
    """
    Position sizing functionality
    Similar to pysystemtrade position sizing
    """
    
    def __init__(self, 
                 volatility_target: float = 0.25,
                 base_currency: str = "USD",
                 capital_multiplier: float = 1.0):
        self.volatility_target = volatility_target
        self.base_currency = base_currency
        self.capital_multiplier = capital_multiplier
    
    def calculate_position_size(self, 
                              forecast: Forecast,
                              price: pd.Series,
                              instrument: Instrument,
                              fx_rate: pd.Series = None,
                              volatility: pd.Series = None) -> Position:
        """
        Calculate position size from forecast
        
        Parameters:
        -----------
        forecast: Forecast
            Trading forecast
        price: pd.Series
            Price series for the instrument
        instrument: Instrument
            Instrument being traded
        fx_rate: pd.Series
            FX rate to base currency (optional)
        volatility: pd.Series
            Price volatility (optional, calculated if not provided)
            
        Returns:
        --------
        Position
            Position series
        """
        # Align forecast and price
        aligned_forecast, aligned_price = forecast.align(price, method='ffill')
        
        # Calculate volatility if not provided
        if volatility is None:
            returns = aligned_price.pct_change()
            volatility = returns.rolling(window=25).std() * np.sqrt(252)
        
        # Align volatility
        aligned_volatility = volatility.reindex(aligned_forecast.index, method='ffill')
        
        # Calculate FX rate if needed
        if fx_rate is None:
            fx_rate = pd.Series(1.0, index=aligned_forecast.index)
        else:
            fx_rate = fx_rate.reindex(aligned_forecast.index, method='ffill')
        
        # Calculate position size
        # Position = (forecast / forecast_cap) * (target_vol / instrument_vol) * capital_multiplier
        position_series = pd.Series(0.0, index=aligned_forecast.index)
        
        for date in aligned_forecast.index:
            if (pd.notna(aligned_forecast.loc[date]) and 
                pd.notna(aligned_price.loc[date]) and 
                pd.notna(aligned_volatility.loc[date]) and
                aligned_volatility.loc[date] > 0):
                
                # Forecast component
                forecast_component = aligned_forecast.loc[date] / forecast.forecast_cap
                
                # Volatility scaling
                vol_scalar = self.volatility_target / aligned_volatility.loc[date]
                
                # FX adjustment
                fx_adjustment = 1.0 / fx_rate.loc[date]
                
                # Point size adjustment
                point_size = instrument.point_size
                
                # Calculate position
                position = (forecast_component * vol_scalar * 
                           fx_adjustment * self.capital_multiplier * point_size)
                
                position_series.loc[date] = position
        
        return Position(position_series, instrument=instrument.name)
    
    def calculate_portfolio_positions(self, 
                                    forecasts: Dict[str, Forecast],
                                    prices: Dict[str, pd.Series],
                                    instruments: InstrumentList,
                                    weights: pd.DataFrame = None,
                                    fx_rates: Dict[str, pd.Series] = None) -> PositionSeries:
        """
        Calculate positions for entire portfolio
        
        Parameters:
        -----------
        forecasts: Dict[str, Forecast]
            Forecasts for each instrument
        prices: Dict[str, pd.Series]
            Prices for each instrument
        instruments: InstrumentList
            List of instruments
        weights: pd.DataFrame
            Instrument weights (optional)
        fx_rates: Dict[str, pd.Series]
            FX rates for each instrument (optional)
            
        Returns:
        --------
        PositionSeries
            Portfolio positions
        """
        positions = {}
        
        for instrument_name in forecasts.keys():
            if instrument_name in prices and instrument_name in instruments:
                instrument = instruments[instrument_name]
                forecast = forecasts[instrument_name]
                price = prices[instrument_name]
                
                # Get FX rate
                fx_rate = None
                if fx_rates and instrument_name in fx_rates:
                    fx_rate = fx_rates[instrument_name]
                
                # Calculate raw position
                position = self.calculate_position_size(
                    forecast, price, instrument, fx_rate
                )
                
                # Apply weight if provided
                if weights is not None and instrument_name in weights.columns:
                    weight_series = weights[instrument_name].reindex(
                        position.index, method='ffill'
                    ).fillna(0)
                    position = Position(position * weight_series, instrument=instrument_name)
                
                positions[instrument_name] = position
        
        return PositionSeries(positions)


class VolatilityEstimator:
    """
    Volatility estimation for position sizing
    """
    
    def __init__(self, 
                 min_periods: int = 10,
                 vol_lookback: int = 25,
                 vol_floor: float = 0.01,
                 vol_ceiling: float = 5.0):
        self.min_periods = min_periods
        self.vol_lookback = vol_lookback
        self.vol_floor = vol_floor
        self.vol_ceiling = vol_ceiling
    
    def estimate_volatility(self, prices: pd.Series) -> pd.Series:
        """
        Estimate volatility from price series
        
        Parameters:
        -----------
        prices: pd.Series
            Price series
            
        Returns:
        --------
        pd.Series
            Volatility series (annualized)
        """
        # Calculate returns
        returns = prices.pct_change()
        
        # Calculate rolling volatility
        vol = returns.rolling(
            window=self.vol_lookback,
            min_periods=self.min_periods
        ).std() * np.sqrt(252)
        
        # Apply floor and ceiling
        vol = vol.clip(self.vol_floor, self.vol_ceiling)
        
        return vol
    
    def estimate_portfolio_volatilities(self, 
                                      prices: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Estimate volatilities for multiple instruments
        
        Parameters:
        -----------
        prices: Dict[str, pd.Series]
            Price series for each instrument
            
        Returns:
        --------
        Dict[str, pd.Series]
            Volatility series for each instrument
        """
        volatilities = {}
        
        for instrument, price_series in prices.items():
            volatilities[instrument] = self.estimate_volatility(price_series)
        
        return volatilities


class RiskBudgeter:
    """
    Risk budgeting functionality
    """
    
    def __init__(self, 
                 max_instrument_weight: float = 0.5,
                 max_asset_class_weight: float = 0.8,
                 max_leverage: float = 1.0):
        self.max_instrument_weight = max_instrument_weight
        self.max_asset_class_weight = max_asset_class_weight
        self.max_leverage = max_leverage
    
    def apply_risk_budgets(self, 
                          positions: PositionSeries,
                          instruments: InstrumentList) -> PositionSeries:
        """
        Apply risk budget constraints to positions
        
        Parameters:
        -----------
        positions: PositionSeries
            Raw positions
        instruments: InstrumentList
            Instrument metadata
            
        Returns:
        --------
        PositionSeries
            Risk-budgeted positions
        """
        # Get positions as DataFrame
        positions_df = positions.get_portfolio_positions()
        
        if positions_df.empty:
            return positions
        
        # Apply instrument-level constraints
        for instrument_name in positions_df.columns:
            positions_df[instrument_name] = positions_df[instrument_name].clip(
                -self.max_instrument_weight,
                self.max_instrument_weight
            )
        
        # Apply asset class constraints
        positions_df = self._apply_asset_class_constraints(positions_df, instruments)
        
        # Apply leverage constraint
        positions_df = self._apply_leverage_constraint(positions_df)
        
        # Convert back to PositionSeries
        budgeted_positions = {}
        for instrument_name in positions_df.columns:
            budgeted_positions[instrument_name] = Position(
                positions_df[instrument_name], 
                instrument=instrument_name
            )
        
        return PositionSeries(budgeted_positions)
    
    def _apply_asset_class_constraints(self, 
                                     positions_df: pd.DataFrame,
                                     instruments: InstrumentList) -> pd.DataFrame:
        """Apply asset class level constraints"""
        # Group by asset class
        asset_classes = {}
        for instrument_name in positions_df.columns:
            if instrument_name in instruments:
                instrument = instruments[instrument_name]
                asset_class = instrument.asset_class
                
                if asset_class not in asset_classes:
                    asset_classes[asset_class] = []
                asset_classes[asset_class].append(instrument_name)
        
        # Apply constraints for each asset class
        for asset_class, instrument_names in asset_classes.items():
            if len(instrument_names) > 1:
                # Calculate total exposure for asset class
                asset_class_exposure = positions_df[instrument_names].sum(axis=1)
                
                # Scale if exceeds limit
                excess_mask = asset_class_exposure.abs() > self.max_asset_class_weight
                if excess_mask.any():
                    for instrument_name in instrument_names:
                        scaling_factor = self.max_asset_class_weight / asset_class_exposure.abs()
                        scaling_factor = scaling_factor.clip(upper=1.0)
                        positions_df[instrument_name] = (
                            positions_df[instrument_name] * scaling_factor
                        )
        
        return positions_df
    
    def _apply_leverage_constraint(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """Apply portfolio leverage constraint"""
        # Calculate gross leverage
        gross_leverage = positions_df.abs().sum(axis=1)
        
        # Scale if exceeds limit
        excess_mask = gross_leverage > self.max_leverage
        if excess_mask.any():
            scaling_factor = self.max_leverage / gross_leverage
            scaling_factor = scaling_factor.clip(upper=1.0)
            
            for col in positions_df.columns:
                positions_df[col] = positions_df[col] * scaling_factor
        
        return positions_df


def create_sample_portfolio_system():
    """Create sample portfolio system for testing"""
    from ..sysobjects.instruments import create_sample_instruments
    from ..sysobjects.forecasts import create_sample_forecast
    from ..sysobjects.prices import create_sample_price_data
    
    # Create sample data
    instruments = create_sample_instruments()
    
    # Create forecasts and prices
    forecasts = {}
    prices = {}
    
    for instrument in instruments.get_instrument_list()[:4]:  # First 4 instruments
        forecasts[instrument] = create_sample_forecast(instrument)
        price_data = create_sample_price_data(instrument)
        prices[instrument] = price_data.adjusted_prices('close')
    
    # Create portfolio components
    optimizer = PortfolioOptimizer()
    position_sizer = PositionSizer()
    risk_budgeter = RiskBudgeter()
    
    # Calculate positions
    positions = position_sizer.calculate_portfolio_positions(
        forecasts, prices, instruments
    )
    
    # Apply risk budgets
    budgeted_positions = risk_budgeter.apply_risk_budgets(positions, instruments)
    
    return {
        'instruments': instruments,
        'forecasts': forecasts,
        'prices': prices,
        'positions': positions,
        'budgeted_positions': budgeted_positions,
        'optimizer': optimizer,
        'position_sizer': position_sizer,
        'risk_budgeter': risk_budgeter
    }