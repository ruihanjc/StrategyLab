"""
Risk budgeting functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from ..sysobjects.instruments import Instrument, InstrumentList
from ..sysobjects.positions import Position, PositionSeries


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