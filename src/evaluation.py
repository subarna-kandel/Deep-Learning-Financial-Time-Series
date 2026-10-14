"""Model evaluation metrics, backtesting routines, and walk-forward splits."""

import numpy as np


def calculate_metrics(y_true, y_pred):
    """Compute MSE, RMSE, MAE, and R-squared for regression estimates."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else 0.0
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def backtest_strategy(predictions, realized_returns, threshold=0.0):
    """Evaluate simple long/cash quantitative rule on realized returns."""
    signals = np.where(predictions > threshold, 1.0, 0.0)
    strategy_returns = signals * realized_returns
    equity_curve = np.cumprod(1.0 + strategy_returns)
    total_return = equity_curve[-1] - 1.0
    n = len(strategy_returns)
    ann_return = (1.0 + total_return) ** (252.0 / max(n, 1)) - 1.0
    vol = np.std(strategy_returns) * np.sqrt(252.0)
    sharpe = ann_return / vol if vol > 1e-8 else 0.0
    return {
        'Total Return': total_return,
        'Annualized Return': ann_return,
        'Annualized Volatility': vol,
        'Sharpe Ratio': sharpe,
        'Equity Curve': equity_curve
    }


def create_walk_forward_folds(n_samples, train_size=500, test_size=250):
    """Generate rolling walk-forward fold indices."""
    folds = []
    start = 0
    while start + train_size + test_size <= n_samples:
        train_idx = list(range(start, start + train_size))
        test_idx = list(range(start + train_size, start + train_size + test_size))
        folds.append((train_idx, test_idx))
        start += test_size
    return folds
