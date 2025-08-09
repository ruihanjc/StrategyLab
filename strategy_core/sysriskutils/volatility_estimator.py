"""
Volatility estimation for position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


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