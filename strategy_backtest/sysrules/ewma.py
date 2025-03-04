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
                ma_periods = [16,32,64]

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

        # Make sure we handle NaN values properly
        ma_df_filled = ma_df.bfill() # Backfill NaN values

        # Calculate how many MAs the price is above/below at each point
        # First align price with MA dataframe
        price_series = data[price_column].reindex(ma_df_filled.index)

        above_mas = (price_series.values.reshape(-1, 1) > ma_df_filled.values).sum(axis=1)
        below_mas = (price_series.values.reshape(-1, 1) < ma_df_filled.values).sum(axis=1)

        above_mas = pd.Series(above_mas, index=ma_df_filled.index)
        below_mas = pd.Series(below_mas, index=ma_df_filled.index)

        # Calculate ratios (avoid division by zero if no MA periods)
        if len(ma_periods) > 0:
            price_above_ma_ratio = above_mas / len(ma_periods)
            price_below_ma_ratio = below_mas / len(ma_periods)
        else:
            price_above_ma_ratio = pd.Series(0, index=ma_df_filled.index)
            price_below_ma_ratio = pd.Series(0, index=ma_df_filled.index)

        # Add indicators to data for rule evaluation
        indicators = {
            'price_above_ma_ratio': price_above_ma_ratio,
            'price_below_ma_ratio': price_below_ma_ratio,
            'threshold': threshold
        }

        if self.signal_rules and 'buy' in self.signal_rules and 'sell' in self.signal_rules:
            for rule in self.signal_rules['buy']:
                condition = rule['condition']
                lookback = rule.get('lookback', 0)
                prior_condition = rule.get('prior_condition', None)
                condition_mask = self._evaluate_condition(condition, data, indicators)

                if prior_condition:
                    prior_mask = self._evaluate_condition(prior_condition, data, indicators)
                    if lookback > 0:
                        prior_mask = prior_mask.shift(lookback)
                    signals[condition_mask & prior_mask] = 1
                else:
                    signals[condition_mask] = 1

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
            buy_condition = price_above_ma_ratio >= threshold
            buy_signal = buy_condition & ~buy_condition.shift(1)
            signals[buy_signal] = 1

            sell_condition = price_below_ma_ratio >= threshold
            sell_signal = sell_condition & ~sell_condition.shift(1)
            signals[sell_signal] = -1

        return signals

    def _evaluate_condition(self, condition, data, indicators):
        local_vars = {}
        local_vars.update(indicators)

        for param_name, param_value in self.parameters.items():
            local_vars[param_name] = param_value

        for column in data.columns:
            local_vars[column] = data[column]

        try:
            result = eval(condition, {"__builtins__": {}}, local_vars)
            return result
        except Exception as e:
            print(f"Error evaluating condition '{condition}': {str(e)}")
            return pd.Series(False, index=data.index)

