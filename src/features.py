"""Feature engineering, sequence generation, and GAF image transformations."""

import numpy as np
import pandas as pd


def create_sequences(return_series, lookback=25, horizon=25):
    """Generate rolling sequence windows (X) and future target horizons (y)."""
    values = np.asarray(return_series)
    X, y = [], []
    for i in range(len(values) - lookback - horizon + 1):
        X.append(values[i:i + lookback])
        y.append(values[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y)


def chronological_split(X, y, test_size=0.20):
    """Strict chronological train/test partition without data leakage."""
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]


def create_gaf(sequence):
    """Convert 1D time series sequence into Gramian Angular Summation Field (GASF)."""
    sequence = np.asarray(sequence, dtype=np.float64)
    min_val, max_val = np.min(sequence), np.max(sequence)
    if max_val == min_val:
        scaled = np.zeros_like(sequence)
    else:
        scaled = 2 * (sequence - min_val) / (max_val - min_val) - 1
    scaled = np.clip(scaled, -1.0, 1.0)
    phi = np.arccos(scaled)
    # G_ij = cos(phi_i + phi_j)
    gaf = np.cos(phi[:, None] + phi[None, :])
    return gaf


def batch_create_gaf(X_sequences):
    """Convert batch of 1D sequence arrays into 2D GAF image matrices."""
    return np.array([create_gaf(seq) for seq in X_sequences])
