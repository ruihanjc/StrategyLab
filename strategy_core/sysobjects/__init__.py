"""
Enhanced data structures for backtesting system
Inspired by pysystemtrade architecture
"""

from .instruments import Instrument, InstrumentList
from .prices import AdjustedPrices, MultiplePrices  
from .forecasts import Forecast, ForecastCombination
from .positions import Position, PositionSeries
from .costs import TradingCosts, CostCalculator
from .portfolio import Portfolio
from .position_sizer import PositionSizer

__all__ = [
    'Instrument', 'InstrumentList',
    'AdjustedPrices', 'MultiplePrices',
    'Forecast', 'ForecastCombination', 
    'Position', 'PositionSeries',
    'TradingCosts', 'CostCalculator',
    'Portfolio', 'PositionSizer'
]