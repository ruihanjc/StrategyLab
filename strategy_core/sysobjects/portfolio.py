"""
Main Portfolio class - orchestrates portfolio management
Contains instruments and coordinates position sizing, optimization, and risk management
"""

import logging
from typing import Dict

import pandas as pd

from .forecasts import Forecast
from .instruments import Instrument, InstrumentList
from .position_sizer import PositionSizer
from .positions import PositionSeries
from ..sysriskutils.portfolio_optimizer import PortfolioOptimizer
from ..sysriskutils.risk_budgeter import RiskBudgeter
from ..sysriskutils.volatility_estimator import VolatilityEstimator


class Portfolio:
    """
    Main Portfolio class that orchestrates portfolio management
    
    Contains:
    - List of instruments
    - Position sizing logic
    - Portfolio optimization
    - Risk management
    - Portfolio-level calculations
    """

    def __init__(self,
                 instruments: InstrumentList,
                 initial_capital: float = 1000000,
                 volatility_target: float = 0.25,
                 max_leverage: float = 1.0):
        """
        Initialize Portfolio
        
        Parameters:
        -----------
        instruments: InstrumentList
            List of instruments in the portfolio
        initial_capital: float
            Initial portfolio capital
        volatility_target: float
            Target portfolio volatility
        max_leverage: float
            Maximum portfolio leverage
        base_currency: str
            Base currency for portfolio
        """
        self.instruments = instruments
        self.initial_capital = initial_capital
        self.volatility_target = volatility_target
        self.max_leverage = max_leverage

        # Current portfolio state
        self.current_positions = PositionSeries({})
        self.current_weights = pd.DataFrame()
        self.current_capital = initial_capital

        # Initialize utility components
        self.position_sizer = PositionSizer(
            volatility_target=volatility_target,
        )
        self.optimizer = PortfolioOptimizer(
            max_portfolio_leverage=max_leverage
        )
        self.volatility_estimator = VolatilityEstimator()
        self.risk_budgeter = RiskBudgeter(
            max_leverage=max_leverage
        )

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.Portfolio")

    def calculate_position_sizes(self,
                                 forecasts: Dict[str, Forecast],
                                 prices: Dict[str, pd.Series],
                                 optimize_weights: bool = True,
                                 apply_risk_budget: bool = True) -> PositionSeries:
        """
        Main method to calculate position sizes for the portfolio
        
        Parameters:
        -----------
        forecasts: Dict[str, Forecast]
            Trading forecasts for each instrument
        prices: Dict[str, pd.Series]
            Price data for each instrument
        optimize_weights: bool
            Whether to optimize instrument weights
        apply_risk_budget: bool
            Whether to apply risk budgeting constraints
            
        Returns:
        --------
        PositionSeries
            Final position sizes for the portfolio
        """
        self.logger.info("Calculating portfolio position sizes...")

        # Step 1: Calculate optimal weights if requested
        weights = None
        if optimize_weights and len(prices) > 1:
            returns = self._calculate_returns(prices)
            weights = self.optimizer.calculate_instrument_weights(
                returns, self.volatility_target
            )
            self.current_weights = weights

        # Step 2: Calculate raw positions using position sizer
        raw_positions = self.position_sizer.calculate_portfolio_positions(
            forecasts=forecasts,
            prices=prices,
            instruments=self.instruments,
            weights=weights
        )

        # Step 3: Apply risk budgeting if requested
        if apply_risk_budget:
            final_positions = self.risk_budgeter.apply_risk_budgets(
                raw_positions, self.instruments
            )
        else:
            final_positions = raw_positions

        # Update current positions
        self.current_positions = final_positions

        self.logger.info(f"Calculated positions for {len(final_positions.get_instruments())} instruments")
        return final_positions

    def rebalance(self,
                  new_forecasts: Dict[str, Forecast],
                  prices: Dict[str, pd.Series],
                  rebalance_threshold: float = 0.1) -> PositionSeries:
        """
        Rebalance the portfolio based on new forecasts
        
        Parameters:
        -----------
        new_forecasts: Dict[str, Forecast]
            New trading forecasts
        prices: Dict[str, pd.Series]
            Current price data
        rebalance_threshold: float
            Minimum change threshold to trigger rebalance
            
        Returns:
        --------
        PositionSeries
            New position sizes after rebalancing
        """
        self.logger.info("Rebalancing portfolio...")

        # Calculate new target positions
        new_positions = self.calculate_position_sizes(
            new_forecasts, prices
        )

        # Check if rebalancing is needed
        if self._should_rebalance(new_positions, rebalance_threshold):
            self.logger.info("Rebalancing triggered")
            self.current_positions = new_positions
        else:
            self.logger.info("No rebalancing needed")

        return self.current_positions

    def add_instrument(self, instrument: Instrument):
        """
        Add a new instrument to the portfolio
        
        Parameters:
        -----------
        instrument: Instrument
            New instrument to add
        """
        self.instruments.add_instrument(instrument)
        self.logger.info(f"Added instrument {instrument.name} to portfolio")

    def remove_instrument(self, instrument_name: str):
        """
        Remove an instrument from the portfolio
        
        Parameters:
        -----------
        instrument_name: str
            Name of instrument to remove
        """
        # Close position in removed instrument
        if instrument_name in self.current_positions.get_instruments():
            self.current_positions.positions.pop(instrument_name, None)

        self.instruments.remove_instrument(instrument_name)
        self.logger.info(f"Removed instrument {instrument_name} from portfolio")

    def get_portfolio_summary(self) -> Dict:
        """
        Get summary of current portfolio state
        
        Returns:
        --------
        Dict
            Portfolio summary information
        """
        positions_df = self.current_positions.get_portfolio_positions()

        summary = {
            'total_instruments': len(self.instruments),
            'active_positions': len(positions_df.columns) if not positions_df.empty else 0,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'volatility_target': self.volatility_target,
            'max_leverage': self.max_leverage,
        }

        if not positions_df.empty:
            latest_positions = positions_df.iloc[-1]
            summary.update({
                'current_leverage': latest_positions.abs().sum(),
                'long_positions': (latest_positions > 0).sum(),
                'short_positions': (latest_positions < 0).sum(),
                'largest_position': latest_positions.abs().max(),
                'position_concentration': latest_positions.abs().max() / latest_positions.abs().sum()
            })

        return summary

    def get_instrument_allocations(self) -> pd.Series:
        """
        Get current instrument allocations as percentages
        
        Returns:
        --------
        pd.Series
            Instrument allocations
        """
        positions_df = self.current_positions.get_portfolio_positions()

        if positions_df.empty:
            return pd.Series()

        latest_positions = positions_df.iloc[-1]
        total_exposure = latest_positions.abs().sum()

        if total_exposure > 0:
            allocations = latest_positions.abs() / total_exposure
            return allocations.sort_values(ascending=False)

        return pd.Series()

    def _calculate_returns(self, prices: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Parameters:
        -----------
        prices: Dict[str, pd.Series]
            Price data for each instrument
            
        Returns:
        --------
        pd.DataFrame
            Returns for each instrument
        """
        returns_data = {}

        for instrument_name, price_series in prices.items():
            if instrument_name in self.instruments:
                returns_data[instrument_name] = price_series.pct_change().dropna()

        returns_df = pd.DataFrame(returns_data)
        return returns_df.dropna()

    def _should_rebalance(self,
                          new_positions: PositionSeries,
                          threshold: float) -> bool:
        """
        Check if portfolio should be rebalanced
        
        Parameters:
        -----------
        new_positions: PositionSeries
            New target positions
        threshold: float
            Rebalance threshold
            
        Returns:
        --------
        bool
            Whether rebalancing is needed
        """
        if not self.current_positions:
            return True

        current_df = self.current_positions.get_portfolio_positions()
        new_df = new_positions.get_portfolio_positions()

        if current_df.empty or new_df.empty:
            return True

        # Calculate position changes
        common_index = current_df.index.intersection(new_df.index)
        if len(common_index) == 0:
            return True

        current_latest = current_df.loc[common_index[-1]]
        new_latest = new_df.loc[common_index[-1]]

        # Calculate relative change
        position_changes = (new_latest - current_latest).abs()
        max_change = position_changes.max()

        return max_change > threshold

    def update_capital(self, new_capital: float):
        """
        Update portfolio capital
        
        Parameters:
        -----------
        new_capital: float
            New portfolio capital
        """
        self.current_capital = new_capital
        self.logger.info(f"Updated portfolio capital to {new_capital:,.2f}")

    def get_risk_metrics(self, prices: Dict[str, pd.Series]) -> Dict:
        """
        Calculate portfolio risk metrics
        
        Parameters:
        -----------
        prices: Dict[str, pd.Series]
            Price data for risk calculations
            
        Returns:
        --------
        Dict
            Risk metrics
        """
        positions_df = self.current_positions.get_portfolio_positions()

        if positions_df.empty:
            return {}

        # Calculate portfolio volatilities
        volatilities = self.volatility_estimator.estimate_portfolio_volatilities(prices)

        # Calculate portfolio-level metrics
        risk_metrics = {
            'individual_volatilities': volatilities,
            'position_concentration': self._calculate_concentration(),
            'leverage_utilization': self._calculate_leverage_utilization(),
            'asset_class_exposure': self._calculate_asset_class_exposure()
        }

        return risk_metrics

    def _calculate_concentration(self) -> float:
        allocations = self.get_instrument_allocations()

        if allocations.empty:
            return 0.0

        return (allocations ** 2).sum()

    def _calculate_leverage_utilization(self) -> float:
        """
        Calculate leverage utilization
        
        Returns:
        --------
        float
            Leverage utilization (current leverage / max leverage)
        """
        positions_df = self.current_positions.get_portfolio_positions()

        if positions_df.empty:
            return 0.0

        current_leverage = positions_df.iloc[-1].abs().sum()
        return current_leverage / self.max_leverage

    def _calculate_asset_class_exposure(self) -> Dict[str, float]:
        positions_df = self.current_positions.get_portfolio_positions()

        if positions_df.empty:
            return {}

        latest_positions = positions_df.iloc[-1]
        asset_class_exposure = {}

        for instrument_name in latest_positions.index:
            if instrument_name in self.instruments:
                instrument = self.instruments[instrument_name]
                asset_class = instrument.asset_class

                if asset_class not in asset_class_exposure:
                    asset_class_exposure[asset_class] = 0.0

                asset_class_exposure[asset_class] += abs(latest_positions[instrument_name])

        return asset_class_exposure
