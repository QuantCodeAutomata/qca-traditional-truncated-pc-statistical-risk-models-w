import numpy as np
import pandas as pd
from typing import Optional


def compute_roc(
    daily_pnl: pd.Series,
    gross_exposure: float = 20_000_000.0,
    annualization: int = 252,
) -> float:
    """Return on Capital = 252 * mean(daily P&L) / gross_exposure.

    Args:
        daily_pnl: Series of daily total P&L values
        gross_exposure: Portfolio gross exposure I in dollars
        annualization: Trading days per year

    Returns:
        Annualized ROC as a decimal (e.g., 0.4774 for 47.74%)
    """
    if len(daily_pnl) == 0:
        return 0.0
    return annualization * daily_pnl.mean() / gross_exposure


def compute_annualized_sharpe(
    daily_pnl: pd.Series,
    annualization: int = 252,
) -> float:
    """Annualized Sharpe ratio = sqrt(252) * mean(daily P&L) / std(daily P&L).

    Args:
        daily_pnl: Series of daily total P&L values
        annualization: Trading days per year

    Returns:
        Annualized Sharpe ratio
    """
    if len(daily_pnl) < 2:
        return 0.0
    std = daily_pnl.std(ddof=1)
    if std == 0:
        return 0.0
    return np.sqrt(annualization) * daily_pnl.mean() / std


def compute_cps(
    daily_pnl: pd.Series,
    daily_shares: pd.Series,
) -> float:
    """Cents per share = total P&L / total shares traded.

    Args:
        daily_pnl: Series of daily total P&L (dollars)
        daily_shares: Series of daily total shares traded

    Returns:
        CPS in dollars per share (paper reports in cents; multiply by 100)
    """
    total_shares = daily_shares.sum()
    if total_shares == 0:
        return 0.0
    return daily_pnl.sum() / total_shares


def summarize_performance(
    daily_pnl: pd.Series,
    daily_shares: pd.Series,
    gross_exposure: float = 20_000_000.0,
    annualization: int = 252,
    label: str = '',
) -> dict:
    """Compute all standard performance metrics.

    Args:
        daily_pnl: Series of daily P&L
        daily_shares: Series of daily shares traded
        gross_exposure: Portfolio gross exposure
        annualization: Trading days per year
        label: Optional model label for display

    Returns:
        Dict with keys: label, roc, sharpe, cps, n_days, total_pnl
    """
    roc = compute_roc(daily_pnl, gross_exposure, annualization)
    sharpe = compute_annualized_sharpe(daily_pnl, annualization)
    cps = compute_cps(daily_pnl, daily_shares)

    return {
        'label': label,
        'roc_pct': round(roc * 100, 2),
        'sharpe': round(sharpe, 2),
        'cps_dollars': round(cps, 4),
        'cps_cents': round(cps * 100, 4),
        'n_days': len(daily_pnl),
        'total_pnl': round(daily_pnl.sum(), 2),
        'mean_daily_pnl': round(daily_pnl.mean(), 2),
        'std_daily_pnl': round(daily_pnl.std(ddof=1), 2),
    }


def compute_average_correlation(
    Psi: np.ndarray,
) -> float:
    """Compute average off-diagonal correlation from a correlation matrix.

    overline(Psi) = (1/N^2) * sum_{i,j} Psi_{ij}

    Note: The paper uses (1/N^2) including diagonal entries.
    """
    N = Psi.shape[0]
    return float(Psi.sum()) / (N * N)


def compute_bound_utilization(
    holdings: np.ndarray,
    addv_bounds: np.ndarray,
    bound_fraction: float = 0.01,
) -> float:
    """Compute zeta = max_i(|H_i| / (bound_fraction * ADDV_i)).

    A value <= 1 means all positions are within bounds.
    """
    bounds = bound_fraction * addv_bounds
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.abs(holdings) / np.where(bounds > 0, bounds, np.inf)
    return float(np.max(ratios))
