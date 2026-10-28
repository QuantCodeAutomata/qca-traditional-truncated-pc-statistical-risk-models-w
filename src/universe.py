import numpy as np
import pandas as pd
from typing import Optional


def compute_addv(
    unadj_close: pd.DataFrame,
    unadj_volume: pd.DataFrame,
    lookback: int = 21,
) -> pd.DataFrame:
    """Compute 21-day Average Daily Dollar Volume for each asset.

    ADDV_i(s) = (1/21) * sum_{r=1}^{21} V_{i,s-r} * P^C_{i,s-r}
    where r goes over the preceding 21 trading days (not including day s).

    Args:
        unadj_close: DataFrame of unadjusted close prices, dates x tickers
        unadj_volume: DataFrame of unadjusted volumes, dates x tickers
        lookback: Number of prior trading days for ADDV computation (21)

    Returns:
        DataFrame of ADDV values, same shape as inputs
    """
    dollar_vol = unadj_close * unadj_volume
    # Rolling mean of prior `lookback` days (shift(1) ensures no look-ahead)
    addv = dollar_vol.shift(1).rolling(window=lookback, min_periods=lookback).mean()
    return addv


def get_trading_days(
    all_dates: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
) -> pd.DatetimeIndex:
    """Return trading days within a date range from an existing date index."""
    mask = (all_dates >= pd.Timestamp(start_date)) & (all_dates <= pd.Timestamp(end_date))
    return all_dates[mask]


def build_rebalance_schedule(
    trading_days: pd.DatetimeIndex,
    block_size: int = 21,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Build list of (block_start, block_end) trading-day pairs.

    The backtest is split into non-overlapping blocks of `block_size` trading days.
    Returns list of (start, end) inclusive date pairs for each block.
    """
    blocks = []
    n = len(trading_days)
    idx = 0
    while idx < n:
        end_idx = min(idx + block_size - 1, n - 1)
        blocks.append((trading_days[idx], trading_days[end_idx]))
        idx += block_size
    return blocks


def select_universe(
    addv: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    top_n: int = 2000,
    min_required: int = 50,
) -> list[str]:
    """Select the top `top_n` names by ADDV as of `rebalance_date`.

    Uses the ADDV row for `rebalance_date` (which is based on prior-21-day data
    with no look-ahead). Excludes names with NaN ADDV.

    Args:
        addv: DataFrame of ADDV values (dates x tickers)
        rebalance_date: The date at which to compute the universe
        top_n: Maximum universe size (2000 in the paper)
        min_required: Minimum universe size to proceed

    Returns:
        List of ticker strings in the selected universe
    """
    if rebalance_date not in addv.index:
        # Use closest prior date
        prior_dates = addv.index[addv.index <= rebalance_date]
        if len(prior_dates) == 0:
            return []
        rebalance_date = prior_dates[-1]

    row = addv.loc[rebalance_date].dropna()
    row = row[row > 0]
    sorted_tickers = row.nlargest(top_n).index.tolist()

    if len(sorted_tickers) < min_required:
        return sorted_tickers
    return sorted_tickers


def build_universe_schedule(
    unadj_close: pd.DataFrame,
    unadj_volume: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    block_size: int = 21,
    top_n: int = 2000,
) -> dict[pd.Timestamp, list[str]]:
    """Build the complete universe schedule for all rebalance blocks.

    At each rebalance date, selects the top-n names by prior-21-day ADDV.
    Returns dict mapping block_start_date -> list_of_tickers.
    """
    addv = compute_addv(unadj_close, unadj_volume, lookback=block_size)
    blocks = build_rebalance_schedule(trading_days, block_size)

    schedule = {}
    for block_start, block_end in blocks:
        tickers = select_universe(addv, block_start, top_n=top_n)
        schedule[block_start] = tickers

    return schedule
