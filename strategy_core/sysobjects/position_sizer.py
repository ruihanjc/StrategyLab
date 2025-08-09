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
                 capital_multiplier: float = 1.0):
        self.volatility_target = volatility_target
        self.capital_multiplier = capital_multiplier

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
