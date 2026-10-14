"""Visualization functions for model predictions, loss curves, and equity paths."""

import matplotlib.pyplot as plt
import numpy as np


def plot_equity_curves(strategy_returns_dict, title="Strategy Equity Growth"):
    """Plot cumulative equity paths for multiple strategies or models."""
    plt.figure(figsize=(10, 6))
    for name, rets in strategy_returns_dict.items():
        cum = np.cumprod(1.0 + np.asarray(rets))
        plt.plot(cum, label=name)
    plt.title(title)
    plt.xlabel("Evaluation Steps")
    plt.ylabel("Normalized Portfolio Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return plt.gcf()


def plot_gaf_sample(gaf_matrix, title="Gramian Angular Field (GASF)"):
    """Display 2D heatmap of Gramian Angular Field."""
    plt.figure(figsize=(5, 5))
    plt.imshow(gaf_matrix, cmap='rainbow', origin='lower')
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    return plt.gcf()
