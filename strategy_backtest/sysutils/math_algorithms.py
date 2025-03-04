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