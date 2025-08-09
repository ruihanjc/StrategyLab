"""
Portfolio optimization functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple


class PortfolioOptimizer:
    """
    Portfolio optimization functionality
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

    @staticmethod
    def _calculate_portfolio_volatility(weights: pd.Series,
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