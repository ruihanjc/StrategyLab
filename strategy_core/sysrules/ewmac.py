

import pandas as pd


def ewmac(price_data, params):
    """
    EWMAC trading rule - Exponentially Weighted Moving Average Crossover
    
    Parameters:
    -----------
    price_data: MultiplePrices or pd.Series
        Price data for the instrument
    params: dict
        Parameters containing Lfast and Lslow
        
    Returns:
    --------
    float
        Current forecast value
    """
    # Handle different parameter formats
    if isinstance(params, dict):
        if 'params' in params and isinstance(params['params'], list):
            param_list = params['params']
            if len(param_list) >= 2:
                Lfast, Lslow = param_list[0], param_list[1]
            else:
                return 0.0
        elif 'Lfast' in params and 'Lslow' in params:
            Lfast, Lslow = params['Lfast'], params['Lslow']
        else:
            return 0.0
    elif isinstance(params, (list, tuple)) and len(params) >= 2:
        Lfast, Lslow = params[0], params[1]
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
            
        if price_series.empty or len(price_series) < max(Lfast, Lslow):
            return 0.0
        
        # Calculate EMAs
        fast_ewma = price_series.ewm(span=Lfast, min_periods=1).mean()
        slow_ewma = price_series.ewm(span=Lslow, min_periods=1).mean()
        
        # Calculate raw signal
        raw_ewmac = fast_ewma - slow_ewma
        
        # Estimate volatility for scaling (simple approach)
        returns = price_series.pct_change().dropna()
        if len(returns) < 10:
            vol = returns.std() if len(returns) > 1 else 0.01
        else:
            vol = returns.rolling(window=min(32, len(returns))).std().iloc[-1]
        
        # Avoid division by zero
        if pd.isna(vol) or vol == 0:
            vol = 0.01
            
        # Scale by volatility and return latest forecast
        scaled_forecast = raw_ewmac / (price_series * vol)
        
        # Return the latest value, scaled to reasonable forecast range
        latest_forecast = scaled_forecast.iloc[-1] * 10  # Scale factor
        
        # Cap the forecast
        return max(-20, min(20, latest_forecast))
        
    except Exception as e:
        print(f"Error in ewmac calculation: {str(e)}")
        return 0.0
