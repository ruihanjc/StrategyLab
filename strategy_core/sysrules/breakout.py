import numpy as np
import pandas as pd


def breakout(price_data, params):
    """
    :param price: The price or other series to use (assumed Tx1)
    :type price: pd.DataFrame

    :param lookback: Lookback in days
    :type lookback: int

    :param lookback: Smooth to apply in days. Must be less than lookback! Defaults to smooth/4
    :type lookback: int

    :returns: pd.DataFrame -- unscaled, uncapped forecast

    With thanks to nemo4242 on elitetrader.com for vectorisation

    """
    if isinstance(params, dict):
        if 'params' in params and isinstance(params['params'], list):
            param_list = params['params']
            if len(param_list) >= 1:
                lookback = param_list[0]
            else:
                return 0.0
        else:
            return 0.0
    elif isinstance(params, (list, tuple)) and len(params) >= 2:
        lookback = params[0]
    else:
        return 0.0

    try:
        # Extract close price series - handle new dict format
        if isinstance(price_data, dict) and 'price_data' in price_data:
            actual_price_data = price_data['price_data']
            if hasattr(actual_price_data, 'close'):
                price_series = actual_price_data.close
            elif hasattr(actual_price_data, 'data') and 'close' in actual_price_data.data.columns:
                price_series = actual_price_data.data['close']
            elif isinstance(actual_price_data, pd.Series):
                price_series = actual_price_data
            else:
                return 0.0
        elif hasattr(price_data, 'close'):
            price_series = price_data.close
        elif hasattr(price_data, 'data') and 'close' in price_data.data.columns:
            price_series = price_data.data['close']
        elif isinstance(price_data, pd.Series):
            price_series = price_data
        else:
            return 0.0

        # Data should already be pre-sliced to the current point in time by the caller

        if price_series.empty or len(price_series) < lookback:
            return 0.0

        # Handle NaN values properly for financial data
        # Forward fill any NaN values (don't use 0!)
        price_series_clean = price_series.ffill()

        # Drop any remaining NaN values at the beginning
        if price_series_clean.iloc[0] == 0.0:
            price_series_clean = price_series_clean.iloc[1:].dropna()
        else:
            price_series_clean = price_series_clean.dropna()

        price_series_clean = price_series_clean.iloc[1:]

        # Check if we still have enough data after cleaning
        if price_series_clean.empty or len(price_series_clean) < lookback:
            return 0.0

        # Use TA-Lib for EMA calculations
        price_array = price_series_clean.values.astype(float)

        smooth = max(int(lookback / 4.0), 1)

        roll_max = price_array.rolling(
            lookback, min_periods=int(min(len(price_array), np.ceil(lookback / 2.0)))
        ).max()
        roll_min = price_array.rolling(
            lookback, min_periods=int(min(len(price_array), np.ceil(lookback / 2.0)))
        ).min()

        roll_mean = (roll_max + roll_min) / 2.0

        # gives a nice natural scaling
        output = 40.0 * ((price_array - roll_mean) / (roll_max - roll_min))
        smoothed_output = output.ewm(span=smooth, min_periods=np.ceil(smooth / 2.0)).mean()

        return smoothed_output

    except Exception as e:
        print(f"Error in ewmac calculation: {str(e)}")
        return 0.0