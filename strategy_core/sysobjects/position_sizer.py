"""
Position sizing functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from .instruments import Instrument, InstrumentList
from .positions import Position, PositionSeries
from .forecasts import Forecast


class PositionSizer:
    """
    Position sizing functionality
    """

    def __init__(self,
                 volatility_target: float = 0.25,
                 capital_multiplier: float = 1.0,
                 min_position_threshold: float = 0.01,
                 min_forecast_threshold: float = 0.5,
                 no_trade_buffer: float = 0.1):
        self.volatility_target = volatility_target
        self.capital_multiplier = capital_multiplier
        self.min_position_threshold = min_position_threshold
        self.min_forecast_threshold = min_forecast_threshold
        self.no_trade_buffer = no_trade_buffer

    def calculate_position_size(self,
                                forecast: Forecast,
                                price: pd.Series,
                                instrument: Instrument,
                                fx_rate: pd.Series = None,
                                volatility: pd.Series = None) -> Position:
        # Align forecast and price
        aligned_forecast, aligned_price = forecast.align(price)
        aligned_forecast = aligned_forecast.ffill()
        aligned_price = aligned_price.ffill()

        # Calculate volatility if not provided
        if volatility is None:
            returns = aligned_price.pct_change()
            volatility = returns.rolling(window=25).std() * np.sqrt(252)

        # Align volatility
        aligned_volatility = volatility.reindex(aligned_forecast.index).ffill()

        # Calculate FX rate if needed
        if fx_rate is None:
            fx_rate = pd.Series(1.0, index=aligned_forecast.index)
        else:
            fx_rate = fx_rate.reindex(aligned_forecast.index).ffill()

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
    
    def apply_no_trade_zone(self, 
                           new_position: Position, 
                           current_position: Position = None,
                           forecast: Forecast = None) -> Position:
        """
        Apply no-trade zone logic to position sizing
        
        Parameters:
        -----------
        new_position: Position
            Target position from strategy
        current_position: Position
            Current held position
        forecast: Forecast
            Original forecast for filtering weak signals
            
        Returns:
        --------
        Position
            Position after applying no-trade zone filters
        """
        filtered_position = new_position.copy()
        
        if current_position is None:
            current_position = Position(pd.Series(0.0, index=new_position.index), 
                                      instrument=new_position.instrument)
        
        # Align positions
        aligned_new, aligned_current = new_position.align(current_position)
        aligned_new = aligned_new.ffill().fillna(0)
        aligned_current = aligned_current.ffill().fillna(0)
        
        # Apply filters
        for date in aligned_new.index:
            new_pos = aligned_new.loc[date]
            current_pos = aligned_current.loc[date]
            
            # Filter 1: Minimum forecast threshold
            if forecast is not None:
                if date in forecast.index:
                    forecast_val = abs(forecast.loc[date])
                    if forecast_val < self.min_forecast_threshold:
                        filtered_position.iloc[filtered_position.index.get_loc(date)] = current_pos
                        continue
            
            # Filter 2: Minimum position size threshold
            if abs(new_pos) < self.min_position_threshold:
                filtered_position.iloc[filtered_position.index.get_loc(date)] = current_pos
                continue
            
            # Filter 3: No-trade buffer zone
            position_change = abs(new_pos - current_pos)
            buffer_threshold = abs(current_pos) * self.no_trade_buffer
            
            if position_change < buffer_threshold and abs(current_pos) > 0:
                # Stay in current position if change is within buffer
                filtered_position.iloc[filtered_position.index.get_loc(date)] = current_pos
            else:
                # Trade to new position
                filtered_position.iloc[filtered_position.index.get_loc(date)] = new_pos
        
        return filtered_position

    def calculate_portfolio_positions(self,
                                      forecasts: Dict[str, Forecast],
                                      prices: Dict[str, pd.Series],
                                      instruments: InstrumentList,
                                      weights: pd.DataFrame = None,
                                      fx_rates: Dict[str, pd.Series] = None) -> PositionSeries:
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
