import pandas as pd
import numpy as np
from strategy_backtest.sysrules.proper_ewmac import ProperEWMAC, ewmac_calc_vol

# Create test data similar to real market data
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
np.random.seed(42)
# Generate more realistic price data
price_base = 100
prices = [price_base]
for i in range(1, len(dates)):
    daily_return = np.random.normal(0.0005, 0.02)  # 0.05% mean return, 2% daily vol
    new_price = prices[-1] * (1 + daily_return)
    prices.append(new_price)

price_series = pd.Series(prices, index=dates, name='close')

print('Price data sample:')
print(price_series.head())
print(f'Price range: {price_series.min():.2f} to {price_series.max():.2f}')

# Test EWMAC components step by step
print('\nTesting EWMAC components:')

# Test volatility calculation
from strategy_backtest.sysrules.proper_ewmac import robust_vol_calc
vol = robust_vol_calc(price_series, days=35)
print(f'Volatility mean: {vol.mean():.6f}')
print(f'Volatility std: {vol.std():.6f}')
print(f'Volatility range: {vol.min():.6f} to {vol.max():.6f}')

# Test EWMA calculations
fast_ewma = price_series.ewm(span=16, min_periods=2).mean()
slow_ewma = price_series.ewm(span=64, min_periods=2).mean()
raw_ewmac = fast_ewma - slow_ewma

print(f'\nFast EWMA mean: {fast_ewma.mean():.2f}')
print(f'Slow EWMA mean: {slow_ewma.mean():.2f}')
print(f'Raw EWMAC mean: {raw_ewmac.mean():.6f}')
print(f'Raw EWMAC std: {raw_ewmac.std():.6f}')

# Test final forecast
forecast_values = raw_ewmac / vol
forecast_values = forecast_values.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f'\nFinal forecast mean: {forecast_values.mean():.6f}')
print(f'Final forecast std: {forecast_values.std():.6f}')
print(f'Final forecast range: {forecast_values.min():.6f} to {forecast_values.max():.6f}')

# Test rule
ewmac_rule = ProperEWMAC(Lfast=16, Lslow=64, vol_days=35)
forecast = ewmac_rule(price_series)

print('\nRule forecast data:')
if hasattr(forecast, 'get_data'):
    data = forecast.get_data()
    print(f'Forecast mean: {data.mean():.6f}')
    print(f'Forecast std: {data.std():.6f}')
    print(f'Forecast range: {data.min():.6f} to {data.max():.6f}')
    non_zero = (data != 0).sum()
    print(f'Non-zero forecasts: {non_zero}')
    print('Sample forecasts:')
    print(data.dropna().head(10))
else:
    print('No forecast data available')