"""
Cost modeling for backtesting
Based on pysystemtrade cost handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TradingCosts:
    """
    Container for trading cost parameters
    """
    percentage_cost: float = 0.001  # 0.1% per trade
    fixed_cost: float = 0.0  # Fixed cost per trade
    slippage: float = 0.0005  # 0.05% slippage
    minimum_cost: float = 0.0  # Minimum cost per trade
    
    def __post_init__(self):
        # Validate costs are non-negative
        if self.percentage_cost < 0:
            raise ValueError("Percentage cost cannot be negative")
        if self.fixed_cost < 0:
            raise ValueError("Fixed cost cannot be negative")
        if self.slippage < 0:
            raise ValueError("Slippage cannot be negative")
        if self.minimum_cost < 0:
            raise ValueError("Minimum cost cannot be negative")


class CostCalculator:
    """
    Calculates trading costs for backtesting
    Similar to pysystemtrade cost calculations
    """
    
    def __init__(self, costs: Union[TradingCosts, Dict[str, TradingCosts]] = None):
        if costs is None:
            self.costs = TradingCosts()
        elif isinstance(costs, TradingCosts):
            self.costs = costs
        elif isinstance(costs, dict):
            # Different costs per instrument
            self.instrument_costs = costs
            self.costs = TradingCosts()  # Default
        else:
            raise ValueError("Costs must be TradingCosts or dict of TradingCosts")
    
    def calculate_trade_costs(self, 
                            trade_size: float, 
                            price: float,
                            instrument: str = None) -> float:
        """
        Calculate cost for a single trade
        
        Parameters:
        -----------
        trade_size: float
            Size of the trade (signed)
        price: float
            Price at which trade is executed
        instrument: str
            Instrument being traded (for instrument-specific costs)
            
        Returns:
        --------
        float
            Total cost of the trade
        """
        if trade_size == 0:
            return 0.0
        
        # Get costs for this instrument
        costs = self._get_costs_for_instrument(instrument)
        
        # Calculate notional value
        notional_value = abs(trade_size) * price
        
        # Calculate percentage cost
        percentage_cost = notional_value * costs.percentage_cost
        
        # Calculate slippage cost
        slippage_cost = notional_value * costs.slippage
        
        # Total cost
        total_cost = percentage_cost + slippage_cost + costs.fixed_cost
        
        # Apply minimum cost
        total_cost = max(total_cost, costs.minimum_cost)
        
        return total_cost
    
    def calculate_position_costs(self, 
                               positions: pd.Series, 
                               prices: pd.Series,
                               instrument: str = None) -> pd.Series:
        """
        Calculate costs for a position series
        
        Parameters:
        -----------
        positions: pd.Series
            Position series
        prices: pd.Series
            Price series
        instrument: str
            Instrument being traded
            
        Returns:
        --------
        pd.Series
            Cost series
        """
        # Align positions and prices
        aligned_positions, aligned_prices = positions.align(prices)
        aligned_positions = aligned_positions.ffill()
        aligned_prices = aligned_prices.ffill()
        
        # Calculate position changes (trades)
        position_changes = aligned_positions.diff().fillna(0)
        
        # Calculate costs for each trade
        costs = pd.Series(0.0, index=aligned_positions.index)
        
        for date in aligned_positions.index:
            trade_size = position_changes.loc[date]
            price = aligned_prices.loc[date]
            
            if not pd.isna(trade_size) and not pd.isna(price) and trade_size != 0:
                cost = self.calculate_trade_costs(trade_size, price, instrument)
                costs.loc[date] = cost
        
        return costs
    
    def calculate_portfolio_costs(self, 
                                positions: Dict[str, pd.Series], 
                                prices: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate costs for a portfolio of positions
        
        Parameters:
        -----------
        positions: Dict[str, pd.Series]
            Dictionary of position series by instrument
        prices: Dict[str, pd.Series]
            Dictionary of price series by instrument
            
        Returns:
        --------
        pd.DataFrame
            Cost DataFrame with instruments as columns
        """
        instrument_costs = {}
        
        for instrument in positions.keys():
            if instrument in prices:
                position_series = positions[instrument]
                price_series = prices[instrument]
                
                costs = self.calculate_position_costs(
                    position_series, price_series, instrument
                )
                instrument_costs[instrument] = costs
        
        # Create aligned DataFrame
        if instrument_costs:
            cost_df = pd.DataFrame(instrument_costs)
            cost_df = cost_df.fillna(0)
            return cost_df
        else:
            return pd.DataFrame()
    
    def _get_costs_for_instrument(self, instrument: str = None) -> TradingCosts:
        """Get costs for specific instrument"""
        if hasattr(self, 'instrument_costs') and instrument in self.instrument_costs:
            return self.instrument_costs[instrument]
        else:
            return self.costs
    
    def get_cost_breakdown(self, 
                          trade_size: float, 
                          price: float,
                          instrument: str = None) -> Dict[str, float]:
        """
        Get detailed cost breakdown for a trade
        
        Parameters:
        -----------
        trade_size: float
            Size of the trade
        price: float
            Price at which trade is executed
        instrument: str
            Instrument being traded
            
        Returns:
        --------
        Dict[str, float]
            Breakdown of costs
        """
        if trade_size == 0:
            return {
                'percentage_cost': 0.0,
                'slippage_cost': 0.0,
                'fixed_cost': 0.0,
                'total_cost': 0.0
            }
        
        costs = self._get_costs_for_instrument(instrument)
        notional_value = abs(trade_size) * price
        
        percentage_cost = notional_value * costs.percentage_cost
        slippage_cost = notional_value * costs.slippage
        fixed_cost = costs.fixed_cost
        total_cost = percentage_cost + slippage_cost + fixed_cost
        
        # Apply minimum cost
        total_cost = max(total_cost, costs.minimum_cost)
        
        return {
            'notional_value': notional_value,
            'percentage_cost': percentage_cost,
            'slippage_cost': slippage_cost,
            'fixed_cost': fixed_cost,
            'minimum_cost': costs.minimum_cost,
            'total_cost': total_cost
        }
    
    def estimate_annual_costs(self, 
                            position_series: pd.Series, 
                            price_series: pd.Series,
                            instrument: str = None) -> Dict[str, float]:
        """
        Estimate annualized costs for a position series
        
        Parameters:
        -----------
        position_series: pd.Series
            Position series
        price_series: pd.Series
            Price series
        instrument: str
            Instrument being traded
            
        Returns:
        --------
        Dict[str, float]
            Annualized cost estimates
        """
        costs = self.calculate_position_costs(position_series, price_series, instrument)
        
        # Calculate annualized metrics
        total_days = len(position_series)
        trading_days_per_year = 252
        
        if total_days == 0:
            return {
                'total_costs': 0.0,
                'annualized_costs': 0.0,
                'cost_per_trade': 0.0,
                'trades_per_year': 0.0
            }
        
        # Calculate metrics
        total_costs = costs.sum()
        annualized_costs = total_costs * (trading_days_per_year / total_days)
        
        # Trade metrics
        trades = costs[costs > 0]
        num_trades = len(trades)
        cost_per_trade = total_costs / num_trades if num_trades > 0 else 0.0
        trades_per_year = num_trades * (trading_days_per_year / total_days)
        
        # Turnover metrics
        position_changes = position_series.diff().fillna(0)
        turnover = abs(position_changes).sum()
        annualized_turnover = turnover * (trading_days_per_year / total_days)
        
        return {
            'total_costs': total_costs,
            'annualized_costs': annualized_costs,
            'cost_per_trade': cost_per_trade,
            'trades_per_year': trades_per_year,
            'turnover': turnover,
            'annualized_turnover': annualized_turnover,
            'cost_per_unit_turnover': total_costs / turnover if turnover > 0 else 0.0
        }


def create_cost_models() -> Dict[str, CostCalculator]:
    """Create common cost models for different asset classes"""
    
    # Equity costs
    equity_costs = TradingCosts(
        percentage_cost=0.001,  # 0.1% commission
        slippage=0.0005,        # 0.05% slippage
        fixed_cost=0.0,
        minimum_cost=0.0
    )
    
    # Futures costs
    futures_costs = TradingCosts(
        percentage_cost=0.0005,  # 0.05% commission
        slippage=0.0003,         # 0.03% slippage
        fixed_cost=2.0,          # $2 fixed cost per trade
        minimum_cost=2.0
    )
    
    # FX costs
    fx_costs = TradingCosts(
        percentage_cost=0.0002,  # 0.02% commission
        slippage=0.0001,         # 0.01% slippage
        fixed_cost=0.0,
        minimum_cost=0.0
    )
    
    # ETF costs
    etf_costs = TradingCosts(
        percentage_cost=0.0005,  # 0.05% commission
        slippage=0.0002,         # 0.02% slippage
        fixed_cost=0.0,
        minimum_cost=0.0
    )
    
    return {
        'equity': CostCalculator(equity_costs),
        'futures': CostCalculator(futures_costs),
        'fx': CostCalculator(fx_costs),
        'etf': CostCalculator(etf_costs)
    }


def create_sample_cost_analysis():
    """Create sample cost analysis for testing"""
    from .positions import create_sample_position
    from .prices import create_sample_price_data
    
    # Create sample data
    position = create_sample_position()
    price_data = create_sample_price_data()
    prices = price_data.adjusted_prices('close')
    
    # Create cost calculator
    calculator = CostCalculator()
    
    # Calculate costs
    costs = calculator.calculate_position_costs(position, prices)
    annual_costs = calculator.estimate_annual_costs(position, prices)
    
    return {
        'position': position,
        'prices': prices,
        'costs': costs,
        'annual_costs': annual_costs
    }