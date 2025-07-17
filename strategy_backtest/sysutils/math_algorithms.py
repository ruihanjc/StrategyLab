"""
Mathematical algorithms for trading system calculations
Enhanced with pysystemtrade-style implementations
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def calculate_sharpe_ratio(strategy_returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sharpe Ratio for strategy returns

    Parameters:
    -----------
    strategy_returns: pd.Series
        Series of strategy returns
    risk_free_rate: float
        Annual risk-free rate (default: 0.0)
    periods_per_year: int
        Number of periods in a year (252 for daily, 52 for weekly, 12 for monthly)

    Returns:
    --------
    float
        Annualized Sharpe Ratio
    """
    # Convert annual risk-free rate to per-period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    # Calculate excess returns
    excess_returns = strategy_returns - rf_per_period

    # Calculate mean and standard deviation of excess returns
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()

    # Avoid division by zero
    if std_excess_return == 0:
        return 0

    # Calculate and annualize Sharpe Ratio
    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(periods_per_year)

    return sharpe_ratio


def robust_vol_calc(price_data: pd.Series, vol_days: int = 35) -> pd.Series:
    """
    Calculate robust volatility estimate
    Based on pysystemtrade volatility calculation
    
    Parameters:
    -----------
    price_data: pd.Series
        Price data
    vol_days: int
        Number of days for volatility calculation
        
    Returns:
    --------
    pd.Series
        Volatility series (annualized)
    """
    # Calculate returns
    returns = price_data.pct_change()
    
    # Calculate rolling volatility
    vol = returns.rolling(window=vol_days, min_periods=int(vol_days * 0.75)).std()
    
    # Annualize (assuming daily data)
    vol_annualized = vol * np.sqrt(252)
    
    # Apply floor to avoid division by zero
    vol_floor = 0.0001  # 1 basis point minimum
    vol_annualized = vol_annualized.clip(lower=vol_floor)
    
    return vol_annualized


def ewmac_calc(price_data: pd.Series, Lfast: int, Lslow: int, vol_days: int = 35) -> pd.Series:
    """
    Calculate EWMAC (Exponential Weighted Moving Average Crossover) forecast
    Based on pysystemtrade EWMAC implementation
    
    Parameters:
    -----------
    price_data: pd.Series
        Price data
    Lfast: int
        Fast moving average period
    Lslow: int
        Slow moving average period
    vol_days: int
        Volatility calculation period
        
    Returns:
    --------
    pd.Series
        EWMAC forecast
    """
    # Calculate exponential moving averages
    fast_ma = price_data.ewm(span=Lfast).mean()
    slow_ma = price_data.ewm(span=Lslow).mean()
    
    # Calculate raw signal
    raw_signal = fast_ma - slow_ma
    
    # Calculate volatility
    vol = robust_vol_calc(price_data, vol_days)
    
    # Normalize by volatility and price level
    normalized_signal = raw_signal / (price_data * vol / 100)
    
    return normalized_signal


def calculate_correlation_matrix(returns_data: pd.DataFrame, 
                               window: int = 125, 
                               min_periods: int = 20) -> pd.DataFrame:
    """
    Calculate rolling correlation matrix
    
    Parameters:
    -----------
    returns_data: pd.DataFrame
        Returns data with instruments as columns
    window: int
        Rolling window size
    min_periods: int
        Minimum periods required for calculation
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    return returns_data.rolling(window=window, min_periods=min_periods).corr()


def calculate_diversification_multiplier(correlation_matrix: pd.DataFrame,
                                       weights: pd.Series) -> float:
    """
    Calculate diversification multiplier from correlation matrix
    
    Parameters:
    -----------
    correlation_matrix: pd.DataFrame
        Correlation matrix
    weights: pd.Series
        Portfolio weights
        
    Returns:
    --------
    float
        Diversification multiplier
    """
    if correlation_matrix.empty or weights.empty:
        return 1.0
    
    # Calculate portfolio volatility
    portfolio_var = np.dot(weights.values, np.dot(correlation_matrix.values, weights.values))
    portfolio_vol = np.sqrt(portfolio_var)
    
    # Calculate individual volatility contribution
    individual_vol = np.sqrt(np.diag(correlation_matrix))
    weighted_individual_vol = np.dot(weights.values, individual_vol)
    
    # Diversification multiplier
    div_mult = weighted_individual_vol / portfolio_vol if portfolio_vol > 0 else 1.0
    
    return div_mult


def calculate_forecast_scalar(forecast: pd.Series, 
                            target_abs_forecast: float = 10.0,
                            lookback: int = 2500,
                            min_periods: int = 500) -> float:
    """
    Calculate forecast scalar for normalizing forecasts
    
    Parameters:
    -----------
    forecast: pd.Series
        Raw forecast series
    target_abs_forecast: float
        Target absolute forecast level
    lookback: int
        Maximum lookback period
    min_periods: int
        Minimum periods required
        
    Returns:
    --------
    float
        Forecast scalar
    """
    if len(forecast) < min_periods:
        return 1.0
    
    # Use recent data for calculation
    recent_forecast = forecast.dropna().tail(lookback)
    
    if len(recent_forecast) < min_periods:
        return 1.0
    
    # Calculate average absolute forecast
    avg_abs_forecast = recent_forecast.abs().mean()
    
    if avg_abs_forecast == 0:
        return 1.0
    
    # Calculate scalar
    scalar = target_abs_forecast / avg_abs_forecast
    
    # Apply reasonable bounds
    scalar = np.clip(scalar, 0.1, 10.0)
    
    return scalar


def calculate_position_from_forecast(forecast: pd.Series,
                                   volatility: pd.Series,
                                   target_risk: float = 0.25,
                                   forecast_cap: float = 20.0,
                                   instrument_weight: float = 1.0) -> pd.Series:
    """
    Calculate position from forecast
    
    Parameters:
    -----------
    forecast: pd.Series
        Forecast series
    volatility: pd.Series
        Volatility series
    target_risk: float
        Target risk level
    forecast_cap: float
        Forecast cap
    instrument_weight: float
        Instrument weight in portfolio
        
    Returns:
    --------
    pd.Series
        Position series
    """
    # Cap forecast
    capped_forecast = forecast.clip(-forecast_cap, forecast_cap)
    
    # Calculate position
    position = (capped_forecast / forecast_cap) * (target_risk / volatility) * instrument_weight
    
    return position


def calculate_kelly_position(expected_return: float,
                           volatility: float,
                           max_leverage: float = 1.0) -> float:
    """
    Calculate Kelly optimal position size
    
    Parameters:
    -----------
    expected_return: float
        Expected return
    volatility: float
        Volatility
    max_leverage: float
        Maximum leverage constraint
        
    Returns:
    --------
    float
        Optimal position size
    """
    if volatility == 0:
        return 0.0
    
    # Kelly formula
    kelly_position = expected_return / (volatility ** 2)
    
    # Apply leverage constraint
    kelly_position = np.clip(kelly_position, -max_leverage, max_leverage)
    
    return kelly_position


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown from equity curve
    
    Parameters:
    -----------
    equity_curve: pd.Series
        Equity curve
        
    Returns:
    --------
    pd.Series
        Drawdown series
    """
    if equity_curve.empty:
        return pd.Series()
    
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    return drawdown


def calculate_risk_adjusted_return(returns: pd.Series,
                                 benchmark_returns: pd.Series = None,
                                 risk_free_rate: float = 0.0) -> dict:
    """
    Calculate various risk-adjusted return metrics
    
    Parameters:
    -----------
    returns: pd.Series
        Return series
    benchmark_returns: pd.Series
        Benchmark return series (optional)
    risk_free_rate: float
        Risk-free rate
        
    Returns:
    --------
    dict
        Dictionary of risk-adjusted metrics
    """
    if returns.empty:
        return {}
    
    # Basic statistics
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    excess_return = mean_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
    
    # Calmar ratio
    equity_curve = (1 + returns).cumprod()
    max_drawdown = calculate_drawdown(equity_curve).min()
    calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    metrics = {
        'mean_return': mean_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'downside_volatility': downside_volatility
    }
    
    # Add benchmark-relative metrics if benchmark provided
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Beta
        covariance = returns.cov(benchmark_returns)
        benchmark_var = benchmark_returns.var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        
        # Alpha
        benchmark_mean = benchmark_returns.mean() * 252
        alpha = mean_return - (risk_free_rate + beta * (benchmark_mean - risk_free_rate))
        
        # Information ratio
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        metrics.update({
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        })
    
    return metrics


def calculate_trade_statistics(positions: pd.Series) -> dict:
    """
    Calculate trade statistics from position series
    
    Parameters:
    -----------
    positions: pd.Series
        Position series
        
    Returns:
    --------
    dict
        Trade statistics
    """
    if positions.empty:
        return {}
    
    # Calculate position changes (trades)
    position_changes = positions.diff().fillna(0)
    trades = position_changes[position_changes != 0]
    
    # Basic trade statistics
    num_trades = len(trades)
    avg_trade_size = trades.abs().mean() if num_trades > 0 else 0
    max_trade_size = trades.abs().max() if num_trades > 0 else 0
    
    # Turnover
    turnover = position_changes.abs().sum()
    
    # Position statistics
    avg_position = positions.abs().mean()
    max_position = positions.abs().max()
    
    return {
        'num_trades': num_trades,
        'avg_trade_size': avg_trade_size,
        'max_trade_size': max_trade_size,
        'turnover': turnover,
        'avg_position': avg_position,
        'max_position': max_position
    }


def smooth_series(series: pd.Series, window: int = 5, method: str = 'ewm') -> pd.Series:
    """
    Smooth a time series
    
    Parameters:
    -----------
    series: pd.Series
        Input series
    window: int
        Window size
    method: str
        Smoothing method ('ewm', 'rolling', 'median')
        
    Returns:
    --------
    pd.Series
        Smoothed series
    """
    if series.empty:
        return series
    
    if method == 'ewm':
        return series.ewm(span=window).mean()
    elif method == 'rolling':
        return series.rolling(window=window).mean()
    elif method == 'median':
        return series.rolling(window=window).median()
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def normalize_series(series: pd.Series, method: str = 'zscore') -> pd.Series:
    """
    Normalize a time series
    
    Parameters:
    -----------
    series: pd.Series
        Input series
    method: str
        Normalization method ('zscore', 'minmax', 'robust')
        
    Returns:
    --------
    pd.Series
        Normalized series
    """
    if series.empty:
        return series
    
    if method == 'zscore':
        return (series - series.mean()) / series.std()
    elif method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'robust':
        median = series.median()
        mad = (series - median).abs().median()
        return (series - median) / mad if mad > 0 else series - median
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_rolling_beta(returns: pd.Series, 
                         market_returns: pd.Series,
                         window: int = 252) -> pd.Series:
    """
    Calculate rolling beta
    
    Parameters:
    -----------
    returns: pd.Series
        Asset returns
    market_returns: pd.Series
        Market returns
    window: int
        Rolling window
        
    Returns:
    --------
    pd.Series
        Rolling beta
    """
    if returns.empty or market_returns.empty:
        return pd.Series()
    
    # Align series
    aligned_returns, aligned_market = returns.align(market_returns, join='inner')
    
    # Calculate rolling covariance and market variance
    rolling_cov = aligned_returns.rolling(window=window).cov(aligned_market)
    rolling_market_var = aligned_market.rolling(window=window).var()
    
    # Calculate beta
    beta = rolling_cov / rolling_market_var
    
    return beta