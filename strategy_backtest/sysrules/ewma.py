from strategy_backtest.sysrules.base_rules import Strategy
import pandas as pd

class EqualWeightMovingAverage(Strategy):
    """
    Equal Weight Moving Average Strategy

    This strategy uses multiple moving averages with equal weights
    to determine trading signals. Buy signals are generated when
    the majority of moving averages suggest the price is trending up,
    and sell signals when trending down.
    """

    def __init__(self, config=None, ma_periods=None, threshold=0.5):
        """
        Initialize the equal weight moving average strategy

        Parameters:
        -----------
        config: dict or str
            Configuration dictionary or strategy name for YAML
        ma_periods: list
            List of moving average periods to use
        threshold: float
            Threshold for generating signals (0.5 = majority)
        """
        # Initialize base class first
        super().__init__(config)

        # If no config provided, use direct parameters
        if config is None:
            if ma_periods is None:
                # Default MA periods if not specified
                ma_periods = [5, 10, 20, 50, 100]

            self.parameters = {
                'ma_periods': ma_periods,
                'threshold': threshold
            }

            # Default signal rules
            self.signal_rules = {
                'buy': [
                    {
                        'condition': 'price_above_ma_ratio >= threshold',
                        'lookback': 1,
                        'prior_condition': 'price_above_ma_ratio < threshold'
                    }
                ],
                'sell': [
                    {
                        'condition': 'price_below_ma_ratio >= threshold',
                        'lookback': 1,
                        'prior_condition': 'price_below_ma_ratio < threshold'
                    }
                ]
            }

    def generate_signals(self, data, price_column='close'):
        """
        Generate trading signals based on equal-weight moving averages

        Parameters:
        -----------
        data: pd.DataFrame
            Historical price data
        price_column: str
            Column name for price data

        Returns:
        --------
        pd.Series
            Series of trading signals (1=buy, -1=sell, 0=hold)
        """
        ma_periods = self.parameters['ma_periods']
        threshold = self.parameters['threshold']

        signals = pd.Series(index=data.index, data=0)

        # Calculate all moving averages
        ma_dict = {}
        for period in ma_periods:
            ma_dict[f'ma_{period}'] = data[price_column].rolling(window=period).mean()

        # Create a DataFrame with all MAs for easier analysis
        ma_df = pd.DataFrame(ma_dict)

        # Calculate how many MAs the price is above/below at each point
        above_mas = (data[price_column] > ma_df).sum(axis=1)
        below_mas = (data[price_column] < ma_df).sum(axis=1)

        # Calculate ratios
        price_above_ma_ratio = above_mas / len(ma_periods)
        price_below_ma_ratio = below_mas / len(ma_periods)

        # Add indicators to data for rule evaluation
        indicators = {
            'price_above_ma_ratio': price_above_ma_ratio,
            'price_below_ma_ratio': price_below_ma_ratio,
            'threshold': threshold
        }

        # Use signal rules from config if available
        if self.signal_rules and 'buy' in self.signal_rules and 'sell' in self.signal_rules:
            # Process buy rules
            for rule in self.signal_rules['buy']:
                # Replace variable names with actual values for evaluation
                condition = rule['condition']
                lookback = rule.get('lookback', 0)
                prior_condition = rule.get('prior_condition', None)

                # Evaluate current condition
                condition_mask = self._evaluate_condition(condition, data, indicators)

                # If there's a prior condition, evaluate it with lookback
                if prior_condition:
                    prior_mask = self._evaluate_condition(prior_condition, data, indicators)
                    if lookback > 0:
                        prior_mask = prior_mask.shift(lookback)
                    # Buy when current condition is true but prior condition was false
                    signals[condition_mask & prior_mask] = 1
                else:
                    # Simply use current condition
                    signals[condition_mask] = 1

            # Process sell rules
            for rule in self.signal_rules['sell']:
                condition = rule['condition']
                lookback = rule.get('lookback', 0)
                prior_condition = rule.get('prior_condition', None)

                condition_mask = self._evaluate_condition(condition, data, indicators)

                if prior_condition:
                    prior_mask = self._evaluate_condition(prior_condition, data, indicators)
                    if lookback > 0:
                        prior_mask = prior_mask.shift(lookback)
                    signals[condition_mask & prior_mask] = -1
                else:
                    signals[condition_mask] = -1
        else:
            # Default implementation if no rules in config
            # Buy when majority of MAs are below price
            buy_condition = price_above_ma_ratio >= threshold
            buy_signal = buy_condition & ~buy_condition.shift(1)
            signals[buy_signal] = 1

            # Sell when majority of MAs are above price
            sell_condition = price_below_ma_ratio >= threshold
            sell_signal = sell_condition & ~sell_condition.shift(1)
            signals[sell_signal] = -1

        return signals

    def _evaluate_condition(self, condition, data, indicators):
        local_vars = {}
        local_vars.update(indicators)

        # Add parameters to local vars
        for param_name, param_value in self.parameters.items():
            local_vars[param_name] = param_value

        # Add columns from data to local vars
        for column in data.columns:
            local_vars[column] = data[column]

        try:
            # Use safer eval approach with only necessary variables
            result = eval(condition, {"__builtins__": {}}, local_vars)
            return result
        except Exception as e:
            print(f"Error evaluating condition '{condition}': {str(e)}")
            return pd.Series(False, index=data.index)


# Register with StrategyFactory
if __name__ == "__main__":
    from strategies.base import StrategyFactory
    from utils.config import ConfigManager
    import yaml

    # Register the strategy
    StrategyFactory.register_strategy('EqualWeightMovingAverage', EqualWeightMovingAverage)

    # Create config
    config_manager = ConfigManager()
    config_manager.create_config('strategies/equal_weight_ma', yaml.safe_load(EQUAL_WEIGHT_MA_CONFIG))

    print("Equal Weight Moving Average Strategy registered and config created")