# Truncated-PC Statistical Risk Models for Equity Portfolios

A quantitative finance research repository implementing the truncated principal-component (PC) statistical risk models described in:

> Kakushadze, Z. (2015). *Statistical Risk Models*. SSRN Working Paper.
> https://ssrn.com/abstract=2551233

## Overview

This project reproduces and extends the empirical backtest from the Kakushadze (2015) paper, which constructs mean-reversion equity strategies and benchmarks a family of covariance estimators across a large-cap US equity universe. The central question is how many principal components to retain when building a factor risk model from the sample covariance matrix of stock returns.

Key ideas:

- **Truncated-PC covariance**: The sample covariance matrix is decomposed via PCA. The leading *K* eigenvectors capture systematic (factor) risk; the remaining eigenvalues are collapsed into an idiosyncratic diagonal. Different values of *K* (including *K* = 0, the "sample" model) are compared.
- **Average Daily Dollar Volume (ADDV)** drives universe selection and position sizing, exactly as in the paper.
- **Mean-reversion alpha**: Overnight returns (open-to-open, adjusted for corporate actions) are used as the raw signal; positions are sized proportionally to alpha and scaled so that gross exposure equals a fixed dollar amount.
- **Performance metrics**: Return on Capital (ROC), annualised Sharpe ratio, and Cents Per Share (CPS) are reported for each model variant.

## Repository Structure

```
.
├── data/
│   ├── __init__.py
│   └── data_loader.py       # Massive REST API data fetching & caching
├── src/
│   ├── __init__.py
│   ├── universe.py          # ADDV computation & universe / rebalance scheduling
│   └── metrics.py           # ROC, Sharpe, CPS and auxiliary performance metrics
├── exp/                     # Experiment scripts and notebooks
├── tests/                   # Unit tests
├── results/                 # Output artefacts (excluded from version control)
├── .gitignore
└── README.md
```

## Data

Daily OHLCV bars are fetched from the **Massive REST API** (polygon.io-compatible). Both adjusted and unadjusted prices are required:

- **Adjusted close / open** — for return computation (corporate-action corrected).
- **Unadjusted close / volume** — for ADDV and position-bound calculation.

Set your API key before running any data fetch:

```bash
export MASSIVE_TOKEN=<your_api_key>
```

The default backtest window is **2008-08-01 → 2014-09-05**, matching the paper's sample period. Fetched data can be cached locally as an HDF5 file to avoid repeated API calls.

## Methodology

### Universe Construction

At each 21-trading-day rebalance block, the top-*N* stocks by trailing 21-day ADDV are selected. ADDV is computed from unadjusted prices so that it reflects actual market liquidity without distortion from dividend adjustments.

### Alpha Signal

The overnight return for stock *i* on day *s* is:

```
r_i(s) = O_i(s) / C_i(s-1) - 1
```

where *O* and *C* denote adjusted open and close prices. The cross-sectional mean is subtracted (market-neutralisation) and positions are set proportional to the resulting demeaned signal.

### Risk Model

The sample return covariance matrix **C** is estimated on the in-sample period. Its eigendecomposition yields eigenvectors **V** and eigenvalues **λ**. The truncated-PC model retains the top *K* eigenvectors as explicit risk factors and replaces remaining eigenvalues with their average:

```
C ≈ Σ_{a=1}^{K} λ_a v_a v_a' + ξ I
```

The resulting inverse is used to compute optimal (risk-adjusted) portfolio weights.

### Performance Metrics

| Metric | Formula |
|---|---|
| ROC | `252 × E[daily P&L] / gross_exposure` |
| Sharpe | `√252 × E[daily P&L] / σ(daily P&L)` |
| CPS | `total P&L / total shares traded` |

## Quickstart

```python
from data.data_loader import load_or_fetch_price_data, MEMBERS_FOR_BACKTEST
from src.universe import build_universe_schedule, get_trading_days

# Fetch (or load cached) price matrices
prices = load_or_fetch_price_data(
    tickers=MEMBERS_FOR_BACKTEST,
    from_date='2008-08-01',
    to_date='2014-09-05',
    cache_path='data/prices.h5',
)

# Build rebalance universe schedule
trading_days = get_trading_days(prices['adj_close'].index, '2009-01-01', '2014-09-05')
schedule = build_universe_schedule(
    unadj_close=prices['unadj_close'],
    unadj_volume=prices['unadj_volume'],
    trading_days=trading_days,
)
```

## Requirements

- Python ≥ 3.10
- `pandas`, `numpy`
- `massive` (Massive REST API Python client)
- `tables` (PyTables, for HDF5 caching via `pd.HDFStore`)

## Reference

Kakushadze, Z. (2015). Statistical Risk Models. *Wilmott*, 2015(79), 50–sixty. SSRN: https://ssrn.com/abstract=2551233
