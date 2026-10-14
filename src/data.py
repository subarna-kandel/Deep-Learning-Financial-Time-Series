"""Data ingestion and preprocessing for asset prices."""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller


def download_price_series(tickers, start_date='2018-01-01', end_date=None):
    """Download daily adjusted close prices for given tickers."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            prices = data['Adj Close']
        else:
            prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data else data['Close']
    return prices.dropna()


def compute_log_returns(prices):
    """Compute continuous daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def run_adf_test(series):
    """Run Augmented Dickey-Fuller test for unit root stationarity."""
    result = adfuller(series.dropna())
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Number of Observations': result[3],
        'Critical Value (1%)': result[4]['1%'],
        'Critical Value (5%)': result[4]['5%'],
        'Critical Value (10%)': result[4]['10%'],
        'Stationary': bool(result[1] < 0.05)
    }
