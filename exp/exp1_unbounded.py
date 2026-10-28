"""Experiment 1: Traditional truncated-PC risk models, unbounded backtests.

Reproduces Table 1 from the paper: compares six K-selection variants (M1-M6)
in an unbounded intraday alpha backtest using overnight returns.

Paper reference: Kakushadze (2015), Table 1.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_or_fetch_price_data, MEMBERS_FOR_BACKTEST
from src.universe import compute_addv, build_rebalance_schedule, select_universe
from src.backtest import run_full_backtest
from src.metrics import summarize_performance

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

BACKTEST_START = '2009-01-01'
BACKTEST_END   = '2014-09-05'
DATA_START     = '2008-08-01'
DATA_END       = '2014-09-05'
VARIANTS       = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']

# Paper plausibility checks (M1-M6)
PAPER_REF = {
    'M1': {'roc_pct': 47.74, 'sharpe': 11.88, 'cps_dollars': 0.0226},
    'M2': {'roc_pct': 46.87, 'sharpe': 11.59, 'cps_dollars': 0.0223},
    'M3': {'roc_pct': 47.89, 'sharpe': 11.20, 'cps_dollars': 0.0228},
    'M4': {'roc_pct': 47.13, 'sharpe': 11.10, 'cps_dollars': 0.0224},
}


def build_universe_schedule(
    unadj_close: pd.DataFrame,
    unadj_volume: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    top_n: int = 2000,
    block_size: int = 21,
) -> dict:
    """Build universe schedule using ADDV ranking."""
    addv_df = compute_addv(unadj_close, unadj_volume)
    blocks = build_rebalance_schedule(trading_days, block_size)
    schedule = {}
    for block_start, _ in blocks:
        tickers = select_universe(addv_df, block_start, top_n=top_n)
        schedule[block_start] = tickers
    return schedule


def run_experiment_1(
    tickers: list = None,
    top_n: int = None,
    cache_path: str = None,
) -> pd.DataFrame:
    """Run Experiment 1: unbounded backtests for all six K-selection variants.

    Args:
        tickers: List of tickers to use. Defaults to MEMBERS_FOR_BACKTEST.
        top_n: Universe size cap. Defaults to len(tickers).
        cache_path: Optional HDF5 cache path for price data.

    Returns:
        DataFrame with performance metrics for each variant.
    """
    if tickers is None:
        tickers = MEMBERS_FOR_BACKTEST
    if top_n is None:
        top_n = len(tickers)

    logger.info(f"Exp1: Loading price data for {len(tickers)} tickers")
    data = load_or_fetch_price_data(tickers, DATA_START, DATA_END, cache_path)

    adj_close   = data['adj_close']
    adj_open    = data['adj_open']
    unadj_close = data['unadj_close']
    unadj_open  = data['unadj_open']
    unadj_volume = data['unadj_volume']

    # Align all frames to the same dates
    common_dates = adj_close.index.intersection(adj_open.index) \
                                  .intersection(unadj_close.index) \
                                  .intersection(unadj_open.index) \
                                  .intersection(unadj_volume.index)
    adj_close    = adj_close.loc[common_dates]
    adj_open     = adj_open.loc[common_dates]
    unadj_close  = unadj_close.loc[common_dates]
    unadj_open   = unadj_open.loc[common_dates]
    unadj_volume = unadj_volume.loc[common_dates]

    trading_days = common_dates[
        (common_dates >= pd.Timestamp(DATA_START)) &
        (common_dates <= pd.Timestamp(DATA_END))
    ]

    logger.info(f"Exp1: {len(trading_days)} trading days from {trading_days[0].date()} "
                f"to {trading_days[-1].date()}")

    # Build universe schedule
    logger.info("Exp1: Building universe schedule")
    universe_schedule = build_universe_schedule(unadj_close, unadj_volume, trading_days, top_n)

    # Run backtests for each variant
    results = []
    for variant in VARIANTS:
        logger.info(f"Exp1: Running variant {variant}")
        bt = run_full_backtest(
            adj_close=adj_close,
            adj_open=adj_open,
            unadj_open=unadj_open,
            unadj_close=unadj_close,
            unadj_volume=unadj_volume,
            universe_schedule=universe_schedule,
            backtest_start=BACKTEST_START,
            backtest_end=BACKTEST_END,
            variant=variant,
            bounded=False,
            verbose=True,
        )

        if not bt:
            logger.warning(f"Variant {variant} returned empty results")
            continue

        summary = bt['summary']
        avg_K = float(np.mean([k for k in bt['block_K'] if k > 0])) if bt['block_K'] else float('nan')
        avg_corr = float(np.nanmean(bt['block_avg_corr'])) if bt['block_avg_corr'] else float('nan')

        row = {
            'variant': variant,
            'roc_pct': summary['roc_pct'],
            'sharpe': summary['sharpe'],
            'cps_cents': summary['cps_cents'],
            'cps_dollars': summary['cps_dollars'],
            'avg_K': round(avg_K, 1),
            'avg_corr_pct': round(avg_corr * 100, 2) if np.isfinite(avg_corr) else float('nan'),
            'n_days': summary['n_days'],
            'total_pnl': summary['total_pnl'],
        }
        results.append(row)
        logger.info(f"  {variant}: ROC={summary['roc_pct']:.2f}%, SR={summary['sharpe']:.2f}, "
                    f"CPS={summary['cps_cents']:.4f}c, avgK={avg_K:.1f}")

        # Save daily P&L
        bt['daily_pnl'].to_csv(RESULTS_DIR / f'exp1_{variant}_daily_pnl.csv')

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'exp1_table1.csv', index=False)
    logger.info(f"Exp1: Results saved to {RESULTS_DIR / 'exp1_table1.csv'}")

    # ── Plot cumulative P&L ────────────────────────────────────────────────
    _plot_cumulative_pnl(df, results, RESULTS_DIR)

    return df


def _plot_cumulative_pnl(df: pd.DataFrame, results_list: list, results_dir: Path) -> None:
    """Plot cumulative P&L for all variants."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(VARIANTS)))

    for i, variant in enumerate(VARIANTS):
        pnl_file = results_dir / f'exp1_{variant}_daily_pnl.csv'
        if not pnl_file.exists():
            continue
        pnl = pd.read_csv(pnl_file, index_col=0, parse_dates=True).squeeze()
        cum_pnl = pnl.cumsum()
        ax.plot(cum_pnl.index, cum_pnl.values / 1e6, label=variant, color=colors[i])

    ax.set_title('Exp1: Cumulative P&L – Unbounded (all variants)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L ($M)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(results_dir / 'exp1_cumulative_pnl.png', dpi=150)
    plt.close(fig)
    logger.info(f"Exp1: Plot saved to {results_dir / 'exp1_cumulative_pnl.png'}")


if __name__ == '__main__':
    df = run_experiment_1()
    print('\n' + '='*60)
    print('Experiment 1 – Unbounded K-selection Comparison (Table 1)')
    print('='*60)
    print(df.to_string(index=False))
    print('\nPaper reference (M1): ROC≈47.74%, SR≈11.88, CPS≈2.26c')
