"""
Moving Average Crossover trading rule with optimization
"""

import pandas as pd
import numpy as np
import talib
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def ma_crossover(price_data, params):
    """
    Simple Moving Average Crossover trading rule
    
    Parameters:
    -----------
    price_data: MultiplePrices or pd.Series
        Price data for the instrument (should be pre-sliced to current point in time)
    params: dict
        Parameters containing 'fast' and 'slow' periods
        
    Returns:
    --------
    float
        Current forecast value
    """
    
    # Handle different parameter formats
    if isinstance(params, dict):
        if 'fast' in params and 'slow' in params:
            fast_period = params['fast']
            slow_period = params['slow']
        else:
            return 0.0
    else:
        return 0.0
    
    try:
        # Extract close price series
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
        if price_series.empty or len(price_series) < max(fast_period, slow_period):
            return 0.0
        
        # Handle NaN values properly for financial data
        # Forward fill any NaN values (don't use 0!)
        price_series_clean = price_series.ffill().dropna()
        
        # Check if we still have enough data after cleaning
        if price_series_clean.empty or len(price_series_clean) < max(fast_period, slow_period):
            return 0.0
        
        # Calculate moving averages using TA-Lib
        price_array = price_series_clean.values.astype(float)
            
        fast_ma = talib.SMA(price_array, timeperiod=fast_period)
        slow_ma = talib.SMA(price_array, timeperiod=slow_period)
        
        # Get the latest values (most recent calculation)
        if np.isnan(fast_ma[-1]) or np.isnan(slow_ma[-1]):
            return 0.0
                
        latest_fast = fast_ma[-1]
        latest_slow = slow_ma[-1]
        
        # Calculate crossover signal
        # Positive when fast > slow (bullish), negative when fast < slow (bearish)
        if latest_slow > 0:
            forecast = ((latest_fast - latest_slow) / latest_slow) * 100  # Percentage difference
        else:
            forecast = 0.0
        
        # Cap the forecast between -20 and +20
        return max(-20, min(20, forecast))
        
    except Exception as e:
        print(f"Error in ma_crossover calculation: {str(e)}")
        return 0.0


def optimize_ma_crossover(engine, target_metric='sharpe_ratio', max_evals=50):

    space = {
        "engine": engine,
        "fast": hp.quniform("fast", 3, 30, 1),
        "slow": hp.quniform("slow", 10, 100, 1),
    }

    original_params = {}
    for i, rule in enumerate(engine.strategy.trading_rules):
        if rule.get_rule().__name__ == 'ma_crossover':
            original_params[i] = rule.params.copy()
    
    def optimization_objective(params):
        try:
            fast = int(params["fast"])
            slow = int(params["slow"])

            if fast >= slow:
                return {"loss": float('inf'), "status": STATUS_OK}

            for rule in engine.strategy.trading_rules:
                if rule.get_rule().__name__ == 'ma_crossover':
                    rule.params.update({"fast": fast, "slow": slow})

            results = engine.run()
            metric_value = results.metrics.get(target_metric, 0.0)
            
            if not np.isfinite(metric_value):
                metric_value = -999.0
            
            print(f"MA: fast={fast:2d}, slow={slow:3d} -> {target_metric}={metric_value:.4f}")
            
            return {"loss": -metric_value, "status": STATUS_OK}
            
        except Exception as e:
            return {"loss": float('inf'), "status": STATUS_OK}

    print("-" * 50)
    
    trials = Trials()
    best = fmin(
        fn=optimization_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    # 恢复原始参数
    for i, rule in enumerate(engine.strategy.trading_rules):
        if i in original_params:
            rule.params.update(original_params[i])
    
    best_params = {
        'fast': int(best['fast']),
        'slow': int(best['slow'])
    }
    
    best_metric = -min(trials.losses())
    
    print("-" * 50)
    print(f"Best sharpe ratio {target_metric}: {best_metric:.4f}")
    print(f"Best param: {best_params}")
    
    return trials, best_params


def apply_ma_crossover_params(engine, best_params):
    print(f"MA Best Params: {best_params}")
    
    for rule in engine.strategy.trading_rules:
        if rule.get_rule().__name__ == 'ma_crossover':
            rule.params.update(best_params)
