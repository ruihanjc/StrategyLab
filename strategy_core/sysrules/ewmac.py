import pandas as pd
import numpy as np
import talib
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def ewmac(price_data, params):
    """
    EWMAC trading rule - Exponentially Weighted Moving Average Crossover
    
    Parameters:
    -----------
    price_data: MultiplePrices or pd.Series
        Price data for the instrument (should be pre-sliced to current point in time)
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

        # Data should already be pre-sliced to the current point in time by the caller

        if price_series.empty or len(price_series) < max(Lfast, Lslow):
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
        if price_series_clean.empty or len(price_series_clean) < Lslow:
            return 0.0

        # Use TA-Lib for EMA calculations
        price_array = price_series_clean.values.astype(float)

        # Calculate EMAs using TA-Lib
        fast_ema = talib.EMA(price_array, timeperiod=Lfast)
        slow_ema = talib.EMA(price_array, timeperiod=Lslow)

        # Get the latest values (most recent calculation)
        if np.isnan(fast_ema[-1]) or np.isnan(slow_ema[-1]):
            return 0.0

        latest_fast = fast_ema[-1]
        latest_slow = slow_ema[-1]

        # Calculate raw EWMAC signal
        raw_ewmac = latest_fast - latest_slow

        scaled_forecast = 0.0

        # Early exit conditions
        if len(price_series_clean) <= 1:
            return 0.0

        # Calculate volatility scalar
        price_changes = price_series_clean.pct_change().dropna()
        if len(price_changes) == 0:
            return 0.0

        vol_scalar = price_changes.std() * np.sqrt(252)
        denominator = latest_slow * vol_scalar

        # Calculate scaled forecast if valid denominator
        if vol_scalar > 0 and abs(denominator) > 1e-10:
            scaled_forecast = raw_ewmac / denominator

        # Apply scaling, ensure finite, and cap
        forecast = scaled_forecast * 10
        forecast = 0.0 if not np.isfinite(forecast) else forecast
        return max(-20, min(20, forecast))

    except Exception as e:
        print(f"Error in ewmac calculation: {str(e)}")
        return 0.0


def optimize_ewmac(engine, target_metric='sharpe_ratio', max_evals=50):
    space = {
        "engine": engine,
        "Lfast": hp.quniform("Lfast", 16, 32, 1),
        "Lslow": hp.quniform("Lslow", 28, 256, 1),
    }

    # 保存原始参数
    original_params = {}
    for i, rule in enumerate(engine.strategy.trading_rules):
        if rule.get_rule().__name__ == 'ewmac':
            original_params[i] = rule.params.copy()

    def optimization_objective(params):
        try:
            lfast = int(params["Lfast"])
            lslow = int(params["Lslow"])

            if lfast >= lslow:
                return {"loss": float('inf'), "status": STATUS_OK}

            for rule in engine.strategy.trading_rules:
                if rule.get_rule().__name__ == 'ewmac':
                    rule.params.update({"Lfast": lfast, "Lslow": lslow})

            results = engine.run()
            metric_value = results.metrics.get(target_metric, 0.0)

            if not np.isfinite(metric_value):
                metric_value = -999.0

            print(f"EWMAC: Lfast={lfast:2d}, Lslow={lslow:3d} -> {target_metric}={metric_value:.4f}")

            return {"loss": -metric_value, "status": STATUS_OK}

        except Exception as e:
            return {"loss": float('inf'), "status": STATUS_OK}

    print(f"Target metric: {target_metric}")
    print("-" * 50)

    trials = Trials()
    best = fmin(
        fn=optimization_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    for i, rule in enumerate(engine.strategy.trading_rules):
        if i in original_params:
            rule.params.update(original_params[i])

    best_params = {
        'Lfast': int(best['Lfast']),
        'Lslow': int(best['Lslow'])
    }

    best_metric = -min(trials.losses())

    print("-" * 50)
    print(f"Best metric achieved: {target_metric}: {best_metric:.4f}")
    print(f"Best params: {best_params}")
    
    # Debug: Show all trials to see if (16,64) was tested
    print(f"\nAll {len(trials.trials)} trials tested:")
    for i, trial in enumerate(trials.trials[:10]):  # Show first 10
        lfast = int(trial['misc']['vals']['Lfast'][0])
        lslow = int(trial['misc']['vals']['Lslow'][0])
        loss = trial['result']['loss']
        metric = -loss if loss != float('inf') else 'INVALID'
        print(f"  Trial {i+1}: Lfast={lfast:2d}, Lslow={lslow:3d} -> {metric}")
    
    if len(trials.trials) > 10:
        print(f"  ... and {len(trials.trials) - 10} more trials")

    return trials, best_params