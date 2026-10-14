# Data Documentation: Deep Learning for Asset Pricing

## Directory Structure
- `raw/`: Unmodified historical price series retrieved via automated data pipelines (e.g. Yahoo Finance via `yfinance`). Ignored by Git.
- `processed/`: Intermediate sequence arrays, Gramian Angular Field (GASF) matrices, and train/validation/test partitions. Ignored by Git.

## Data Sources & Acquisition
- **Sources**: Daily adjusted closing prices for equity assets and ETFs (SPY, QQQ, GLD, etc.) downloaded using `yfinance`.
- **Download Instructions**:
  Run `src/data.py` or execute `notebooks/01_multi_asset_deep_learning.ipynb` to download up-to-date market series directly into memory or cache to `data/raw/`.
- **Pre-processing Pipeline**:
  1. Adjusted close prices converted to log and percent returns.
  2. Augmented Dickey-Fuller (ADF) tests performed to verify stationarity.
  3. Rolling lookback windows generated for multi-step sequence inputs.
  4. Sequence-to-image encoding transformed via Gramian Angular Summation Fields (`src/features.py`).
