"""Backtest engine for intraday overnight-alpha statistical risk model experiments.

Implements the 1260-trading-day block backtest from the paper, supporting both
unbounded and bounded portfolio optimization.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.risk_models import build_risk_model, build_shrinkage_model_diagonal, build_shrinkage_model_uniform_corr
from src.portfolio import compute_unbounded_holdings, compute_bounded_holdings, compute_daily_pnl
from src.metrics import summarize_performance, compute_average_correlation, compute_bound_utilization

logger = logging.getLogger(__name__)


# ── Lookback configuration ─────────────────────────────────────────────────────

ESTIMATION_LOOKBACK = 21    # Trading days of returns used for risk estimation (≈ 1 block, matching M << N)
BLOCK_SIZE = 21             # Rebalance frequency in trading days
GROSS_EXPOSURE = 20_000_000.0   # $20 million gross exposure
BOUND_FRACTION = 0.01       # Per-name liquidity bound: 0.01 * ADDV


# ── Single block: risk model + holdings ───────────────────────────────────────

def run_block(
    block_trading_days: pd.DatetimeIndex,
    universe_tickers: list[str],
    adj_close: pd.DataFrame,
    adj_open: pd.DataFrame,
    unadj_open: pd.DataFrame,
    unadj_close: pd.DataFrame,
    unadj_volume: pd.DataFrame,
    variant: str = 'M1',
    bounded: bool = False,
    K_override: Optional[int] = None,
    shrinkage_params: Optional[dict] = None,
    estimation_lookback: int = ESTIMATION_LOOKBACK,
) -> dict:
    """Run one 21-day block of the backtest.

    For a given universe and block of trading days:
    1. Build risk model from prior adjusted close-to-close returns.
    2. Each day: compute overnight alpha, solve portfolio, compute P&L.

    Args:
        block_trading_days: DatetimeIndex of trading days in this block
        universe_tickers: List of tickers for this block's universe
        adj_close: DataFrame (dates x tickers) of adjusted close prices
        adj_open: DataFrame (dates x tickers) of adjusted open prices
        unadj_open: DataFrame (dates x tickers) of unadjusted open prices
        unadj_close: DataFrame (dates x tickers) of unadjusted close prices
        unadj_volume: DataFrame (dates x tickers) of unadjusted volume
        variant: Risk model variant ('M1'-'M6', 'shrinkage_diag', 'shrinkage_uniform')
        bounded: If True, apply per-name liquidity bounds
        K_override: Fixed K (used when variant is ignored for fixed-K experiments)
        shrinkage_params: Dict with 'q' and optionally 'rho' for shrinkage models
        estimation_lookback: Number of prior trading days for risk estimation

    Returns:
        Dict with daily_pnl, daily_shares, selected_K, avg_corr, and daily_holdings
    """
    if len(universe_tickers) == 0:
        return _empty_block_result(block_trading_days)

    # ── Prepare return data ─────────────────────────────────────────────────
    first_day = block_trading_days[0]

    # All available dates up to (but not including) first_day
    all_dates = adj_close.index
    prior_dates = all_dates[all_dates < first_day]

    if len(prior_dates) < estimation_lookback + 1:
        logger.warning(
            f"Block {first_day}: insufficient prior dates ({len(prior_dates)}) "
            f"for estimation lookback {estimation_lookback}. Skipping block."
        )
        return _empty_block_result(block_trading_days)

    # Select estimation window: last `estimation_lookback+1` prior dates
    est_dates = prior_dates[-(estimation_lookback + 1):]

    # Filter to available tickers in adj_close
    avail_tickers = [t for t in universe_tickers if t in adj_close.columns]
    if len(avail_tickers) < 2:
        return _empty_block_result(block_trading_days)

    est_adj_close = adj_close.loc[est_dates, avail_tickers]

    # Fully adjusted close-to-close log returns: R_is = ln(P^AC_{i,s} / P^AC_{i,s+1})
    # Note: going from older to newer: R_is = ln(close[t] / close[t-1])
    # Using .diff(1) / forward-time convention: R_is = ln(P[t+1]/P[t]) or ln(P[t]/P[t-1])?
    # Paper: R_is = ln(P^AC_{is} / P^AC_{i,s+1}) where s+1 is prior in time
    # Treating index as time-ascending: R_t = ln(close[t] / close[t-1])
    log_prices = np.log(est_adj_close.replace(0, np.nan).ffill())
    raw_returns = log_prices.diff(1).dropna()  # shape (estimation_lookback, N_tickers)

    # Drop tickers with any NaN in estimation window
    valid_tickers = raw_returns.columns[raw_returns.notna().all()].tolist()
    if len(valid_tickers) < 2:
        return _empty_block_result(block_trading_days)

    returns_matrix = raw_returns[valid_tickers].values  # (M, N)
    M, N = returns_matrix.shape

    if M < 5 or N < 2:
        return _empty_block_result(block_trading_days)

    # ── Build risk model ────────────────────────────────────────────────────
    try:
        if variant.startswith('shrinkage_diag'):
            q = shrinkage_params.get('q', 0.3) if shrinkage_params else 0.3
            model = build_shrinkage_model_diagonal(returns_matrix, valid_tickers, q=q)
        elif variant.startswith('shrinkage_uniform'):
            q = shrinkage_params.get('q', 0.3) if shrinkage_params else 0.3
            rho = shrinkage_params.get('rho', 0.3) if shrinkage_params else 0.3
            model = build_shrinkage_model_uniform_corr(returns_matrix, valid_tickers, q=q, rho=rho)
        else:
            model = build_risk_model(
                returns_matrix,
                valid_tickers,
                variant=variant if K_override is None else 'fixed',
                K_override=K_override,
                use_gram=(M < N),
                use_woodbury=True,
                compute_inverse=not bounded,
            )
    except Exception as e:
        logger.warning(f"Block {first_day}: risk model failed ({e}). Skipping.")
        return _empty_block_result(block_trading_days)

    selected_K = model.K

    # Validate model
    if model.Gammahat is None or (model.Gammahat_inv is None and not bounded):
        return _empty_block_result(block_trading_days)

    # Average correlation
    if model.all_eigenvalues is not None:
        # Psi average correlation: (1/N^2) * sum_{ij} Psi_ij
        # For large N, reconstruct Psi from eigenvalues/vectors
        try:
            from scipy.linalg import eigh
            C, Psi_full, sigmas, Y = _quick_corr(returns_matrix)
            avg_corr = float(Psi_full.sum()) / (N * N)
        except Exception:
            avg_corr = float('nan')
    else:
        avg_corr = float('nan')

    # ── Daily loop ──────────────────────────────────────────────────────────
    daily_pnl = []
    daily_shares = []
    daily_holdings_list = []
    daily_zeta = []

    # Pre-compute ADDV for this block (for bounded optimization)
    # ADDV = mean of prior 21-day unadjusted dollar volume
    if bounded:
        addv_block = _compute_block_addv(
            unadj_close, unadj_volume, first_day, valid_tickers
        )
    else:
        addv_block = None

    for trade_date in block_trading_days:
        # Check trade_date is in adj_open (for overnight alpha)
        if trade_date not in adj_open.index or trade_date not in unadj_open.index:
            continue

        # Overnight alpha: E_is = ln(P^AO_{is} / P^AC_{i,s-1})
        # We need adjusted open on trade_date and adjusted close on prior day
        prior_trade_dates = all_dates[all_dates < trade_date]
        if len(prior_trade_dates) == 0:
            continue
        prior_date = prior_trade_dates[-1]

        if prior_date not in adj_close.index:
            continue

        adj_open_today = adj_open.loc[trade_date, valid_tickers]
        adj_close_prior = adj_close.loc[prior_date, valid_tickers]

        # Compute overnight returns
        with np.errstate(divide='ignore', invalid='ignore'):
            overnight = np.log(
                adj_open_today.values / adj_close_prior.values
            )
        overnight = np.where(
            np.isfinite(overnight),
            overnight,
            0.0
        )

        # Execution prices (unadjusted)
        unadj_open_today = unadj_open.loc[trade_date, valid_tickers].values \
            if trade_date in unadj_open.index else None
        unadj_close_today = unadj_close.loc[trade_date, valid_tickers].values \
            if trade_date in unadj_close.index else None

        if unadj_open_today is None or unadj_close_today is None:
            continue

        valid_exec = (
            np.isfinite(unadj_open_today) & (unadj_open_today > 0) &
            np.isfinite(unadj_close_today) & (unadj_close_today > 0)
        )

        if not valid_exec.any():
            continue

        # ── Portfolio optimization ──────────────────────────────────────
        try:
            if bounded and addv_block is not None:
                H = compute_bounded_holdings(
                    model.Gammahat,
                    overnight,
                    addv_block,
                    gross_exposure=GROSS_EXPOSURE,
                    bound_fraction=BOUND_FRACTION,
                )
            else:
                H = compute_unbounded_holdings(
                    model.Gammahat_inv,
                    overnight,
                    gross_exposure=GROSS_EXPOSURE,
                )
        except Exception as e:
            logger.debug(f"Portfolio opt failed {trade_date}: {e}")
            continue

        if H is None or len(H) == 0 or not np.isfinite(H).all():
            continue

        # Compute P&L
        pnl_day, shares_day = compute_daily_pnl(H, unadj_open_today, unadj_close_today)
        daily_pnl.append(pnl_day)
        daily_shares.append(shares_day)
        daily_holdings_list.append((trade_date, H))

        # Bound utilization diagnostic
        if bounded and addv_block is not None:
            zeta = compute_bound_utilization(H, addv_block, BOUND_FRACTION)
            daily_zeta.append(zeta)

    return {
        'daily_pnl': daily_pnl,
        'daily_shares': daily_shares,
        'daily_holdings': daily_holdings_list,
        'selected_K': selected_K,
        'avg_corr': avg_corr,
        'daily_zeta': daily_zeta,
        'n_assets': N,
        'M': M,
        'valid_tickers': valid_tickers,
        'first_day': first_day,
    }


def _quick_corr(returns_matrix: np.ndarray):
    """Compute C, Psi, sigmas, Y quickly (helper)."""
    M, N = returns_matrix.shape
    X = returns_matrix - returns_matrix.mean(axis=0)
    C = (X.T @ X) / M
    sigmas = np.sqrt(np.diag(C))
    sigmas = np.where(sigmas > 1e-10, sigmas, 1e-10)
    outer_sigma = np.outer(sigmas, sigmas)
    Psi = C / outer_sigma
    np.fill_diagonal(Psi, 1.0)
    Y = X / sigmas
    return C, Psi, sigmas, Y


def _compute_block_addv(
    unadj_close: pd.DataFrame,
    unadj_volume: pd.DataFrame,
    block_start: pd.Timestamp,
    tickers: list[str],
    lookback: int = 21,
) -> np.ndarray:
    """Compute ADDV for each ticker in the universe as of block_start.

    Uses prior lookback trading days of unadjusted close * volume.
    """
    avail_cols = [t for t in tickers if t in unadj_close.columns and t in unadj_volume.columns]

    prior_dates = unadj_close.index[unadj_close.index < block_start]
    if len(prior_dates) < lookback:
        return np.ones(len(tickers)) * 1e6  # fallback

    est_dates = prior_dates[-lookback:]
    dollar_vol = (unadj_close.loc[est_dates, avail_cols] *
                  unadj_volume.loc[est_dates, avail_cols]).mean(axis=0)

    # Align with full tickers list
    addv = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        if t in dollar_vol.index and np.isfinite(dollar_vol[t]) and dollar_vol[t] > 0:
            addv[i] = dollar_vol[t]
        else:
            addv[i] = 1e6  # fallback to avoid zero bounds

    return addv


def _empty_block_result(block_trading_days: pd.DatetimeIndex) -> dict:
    """Return an empty block result dict."""
    return {
        'daily_pnl': [],
        'daily_shares': [],
        'daily_holdings': [],
        'selected_K': 0,
        'avg_corr': float('nan'),
        'daily_zeta': [],
        'n_assets': 0,
        'M': 0,
        'valid_tickers': [],
        'first_day': block_trading_days[0] if len(block_trading_days) > 0 else None,
    }


# ── Full backtest ──────────────────────────────────────────────────────────────

def run_full_backtest(
    adj_close: pd.DataFrame,
    adj_open: pd.DataFrame,
    unadj_open: pd.DataFrame,
    unadj_close: pd.DataFrame,
    unadj_volume: pd.DataFrame,
    universe_schedule: dict,
    backtest_start: str,
    backtest_end: str,
    variant: str = 'M1',
    bounded: bool = False,
    K_override: Optional[int] = None,
    shrinkage_params: Optional[dict] = None,
    block_size: int = BLOCK_SIZE,
    estimation_lookback: int = ESTIMATION_LOOKBACK,
    verbose: bool = True,
) -> dict:
    """Run the full 1260-trading-day intraday backtest.

    Args:
        adj_close: (dates x tickers) adjusted close prices
        adj_open: (dates x tickers) adjusted open prices
        unadj_open: (dates x tickers) unadjusted open prices
        unadj_close: (dates x tickers) unadjusted close prices
        unadj_volume: (dates x tickers) unadjusted volume
        universe_schedule: {block_start_date: [tickers]} from build_universe_schedule
        backtest_start: Start date of backtest (e.g., '2009-01-01')
        backtest_end: End date of backtest (e.g., '2014-09-05')
        variant: Risk model variant
        bounded: Whether to apply per-name liquidity bounds
        K_override: Fixed K value (overrides variant for fixed-K experiments)
        shrinkage_params: Parameters for shrinkage models
        block_size: Number of trading days per rebalance block
        estimation_lookback: Prior days for risk model estimation
        verbose: If True, log progress

    Returns:
        Dict with:
            daily_pnl: pd.Series
            daily_shares: pd.Series
            block_K: list of selected K per block
            block_avg_corr: list of avg correlations
            block_zeta: list of daily zeta values
            summary: performance metrics dict
    """
    # Filter trading days for backtest period
    all_dates = adj_close.index
    backtest_dates = all_dates[
        (all_dates >= pd.Timestamp(backtest_start)) &
        (all_dates <= pd.Timestamp(backtest_end))
    ]

    if len(backtest_dates) == 0:
        logger.error(f"No trading days in backtest range {backtest_start} to {backtest_end}")
        return {}

    # Build block schedule from backtest dates
    from src.universe import build_rebalance_schedule
    blocks = build_rebalance_schedule(backtest_dates, block_size)

    all_pnl = {}
    all_shares = {}
    block_K = []
    block_avg_corr = []
    all_zeta = []
    block_n_assets = []

    for block_idx, (block_start, block_end) in enumerate(blocks):
        # Get universe for this block from schedule
        # Find the closest schedule entry <= block_start
        schedule_dates = sorted(universe_schedule.keys())
        matching = [d for d in schedule_dates if d <= block_start]
        if not matching:
            logger.warning(f"No universe for block {block_start}; skipping")
            continue
        universe_tickers = universe_schedule[matching[-1]]

        if not universe_tickers:
            continue

        # Block trading days
        block_days = backtest_dates[
            (backtest_dates >= block_start) & (backtest_dates <= block_end)
        ]

        if verbose and block_idx % 10 == 0:
            logger.info(
                f"Block {block_idx+1}/{len(blocks)}: {block_start.date()} "
                f"({len(universe_tickers)} tickers, {len(block_days)} days)"
            )

        result = run_block(
            block_trading_days=block_days,
            universe_tickers=universe_tickers,
            adj_close=adj_close,
            adj_open=adj_open,
            unadj_open=unadj_open,
            unadj_close=unadj_close,
            unadj_volume=unadj_volume,
            variant=variant,
            bounded=bounded,
            K_override=K_override,
            shrinkage_params=shrinkage_params,
            estimation_lookback=estimation_lookback,
        )

        # Collect results
        for i, trade_date in enumerate(block_days):
            if i < len(result['daily_pnl']):
                all_pnl[trade_date] = result['daily_pnl'][i]
                all_shares[trade_date] = result['daily_shares'][i]

        block_K.append(result['selected_K'])
        block_avg_corr.append(result['avg_corr'])
        all_zeta.extend(result['daily_zeta'])
        block_n_assets.append(result['n_assets'])

    # Build time series
    pnl_series = pd.Series(all_pnl).sort_index()
    shares_series = pd.Series(all_shares).sort_index()

    # Compute summary metrics
    summary = summarize_performance(
        pnl_series, shares_series,
        gross_exposure=GROSS_EXPOSURE,
        label=f"{variant}{'_bounded' if bounded else ''}",
    )

    return {
        'daily_pnl': pnl_series,
        'daily_shares': shares_series,
        'block_K': block_K,
        'block_avg_corr': block_avg_corr,
        'block_zeta': all_zeta,
        'block_n_assets': block_n_assets,
        'summary': summary,
        'variant': variant,
        'bounded': bounded,
    }
