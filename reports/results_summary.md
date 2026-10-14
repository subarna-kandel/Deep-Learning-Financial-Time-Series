# Research & Methodology Summary: Deep Learning for Asset Pricing

## 1. Executive Overview
This research develops a deep learning framework for multi-asset return forecasting and systematic portfolio construction. We benchmark standard feed-forward networks (MLP), recurrent architectures (LSTM), and Computer Vision representations via Gramian Angular Summation Fields (CNN-GAF) on financial time series.

## 2. Methodology & Architectures
- **Sequence Encoding**: 1D historical log-return series are transformed into 2D Gramian Angular Fields (GASF), preserving temporal dependencies and cross-step quasi-correlations in polar coordinate space.
- **Model Topology**:
  - **MLP Regressor**: Dense layers with dropout regularization.
  - **LSTM Network**: Recurrent cells capturing multi-scale memory and non-linear decay.
  - **CNN-GAF**: 2D convolutional filter banks extracting geometric patterns from matrix-encoded return histories.
- **Evaluation Framework**: Non-overlapping walk-forward optimization with purging and embargoing protocols to prevent look-ahead bias and information leakage.

## 3. Key Findings & Performance Metrics
- **Predictive Power**: LSTM networks demonstrated superior stability in directional return classification over long-horizon test sets, reducing out-of-sample RMSE relative to vanilla linear baselines.
- **Leakage Hazards**: Standard random K-Fold cross-validation exhibited dramatic out-of-sample degradation due to overlapping label leakage; strictly purged walk-forward splits restored realistic performance bounds.
- **Portfolio Impact**: Incorporating multi-output deep neural forecasts into minimum-variance and risk-parity weighting improved annualized Sharpe ratios while reducing portfolio maximum drawdown.

## 4. Limitations & Production Considerations
- Sensitivity to regime shifts and extreme market shocks (e.g. liquidity dry-ups).
- Computational overhead of continuous 2D image transformation for large asset universes.
