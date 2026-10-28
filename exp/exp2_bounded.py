"""Experiment 2: Traditional truncated-PC risk models, bounded backtests.

Reproduces Table 2 / Table 3 from the paper: bounded optimization with
per-name liquidity bounds |H_i| <= 0.01 * ADDV_i.

Paper reference: Kakushadze (2015), Tables 2-3.
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

# Paper tests M1-M4 in bounded table; M5/M6 as diagnostic
BOUNDED_VARIANTS = ['M1', 'M2', 'M3', 'M4']

PAPER_REF_BOUNDED = {
    'M1': {'roc_pct': 40.92, 'sharpe': 14.33, 'cps_dollars': 0.0196},
    'M2': {'roc_pct': 40.36, 'sharpe': 14.04, 'cps_dollars': 0.0194},
    'M3': {'roc_pct': 40.78, 'sharpe': 14.01, 'cps_dollars': 0.0196},
    'M4': {'roc_pct': 40.84, 'sharpe': 14.06, 'cps_dollars': 0.0196},
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


def run_experiment_2(
    tickers: list = None,
    top_n: int = None,
    cache_path: str = None,
) -> pd.DataFrame:
    """Run Experiment 2: bounded backtests for M1-M4 variants.

    Returns:
        DataFrame with performance metrics and diagnostics.
    """
    if tickers is None:
        tickers = MEMBERS_FOR_BACKTEST
    if top_n is None:
        top_n = len(tickers)

    logger.info(f"Exp2: Loading price data for {len(tickers)} tickers")
    data = load_or_fetch_price_data(tickers, DATA_START, DATA_END, cache_path)

    adj_close    = data['adj_close']
    adj_open     = data['adj_open']
    unadj_close  = data['unadj_close']
    unadj_open   = data['unadj_open']
    unadj_volume = data['unadj_volume']

    # Align dates
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

    logger.info(f"Exp2: {len(trading_days)} trading days")
    universe_schedule = build_universe_schedule(unadj_close, unadj_volume, trading_days, top_n)

    results = []
    for variant in BOUNDED_VARIANTS:
        logger.info(f"Exp2: Running bounded variant {variant}")
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
            bounded=True,
            verbose=True,
        )

        if not bt:
            continue

        summary = bt['summary']
        avg_K = float(np.mean([k for k in bt['block_K'] if k > 0])) if bt['block_K'] else float('nan')
        avg_corr = float(np.nanmean(bt['block_avg_corr'])) if bt['block_avg_corr'] else float('nan')
        avg_zeta = float(np.nanmean(bt['block_zeta'])) if bt['block_zeta'] else float('nan')

        row = {
            'variant': variant,
            'roc_pct': summary['roc_pct'],
            'sharpe': summary['sharpe'],
            'cps_cents': summary['cps_cents'],
            'cps_dollars': summary['cps_dollars'],
            'avg_K': round(avg_K, 1),
            'avg_corr_pct': round(avg_corr * 100, 2) if np.isfinite(avg_corr) else float('nan'),
            'avg_zeta': round(avg_zeta, 4),
            'n_days': summary['n_days'],
        }
        results.append(row)
        logger.info(f"  {variant}: ROC={summary['roc_pct']:.2f}%, SR={summary['sharpe']:.2f}, "
                    f"CPS={summary['cps_cents']:.4f}c, avgK={avg_K:.1f}, avgZeta={avg_zeta:.4f}")

        bt['daily_pnl'].to_csv(RESULTS_DIR / f'exp2_{variant}_daily_pnl.csv')

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'exp2_table2.csv', index=False)
    logger.info(f"Exp2: Saved to {RESULTS_DIR / 'exp2_table2.csv'}")

    # Plot
    _plot_cumulative_pnl_bounded(df, RESULTS_DIR)

    return df


def _plot_cumulative_pnl_bounded(df: pd.DataFrame, results_dir: Path) -> None:
    """Plot cumulative P&L for bounded variants."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(BOUNDED_VARIANTS)))

    for i, variant in enumerate(BOUNDED_VARIANTS):
        pnl_file = results_dir / f'exp2_{variant}_daily_pnl.csv'
        if not pnl_file.exists():
            continue
        pnl = pd.read_csv(pnl_file, index_col=0, parse_dates=True).squeeze()
        ax.plot(pnl.cumsum().values / 1e6, label=variant, color=colors[i])

    ax.set_title('Exp2: Cumulative P&L – Bounded (M1-M4)', fontsize=14)
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Cumulative P&L ($M)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(results_dir / 'exp2_cumulative_pnl.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    df = run_experiment_2()
    print('\n' + '='*60)
    print('Experiment 2 – Bounded K-selection Comparison (Table 2)')
    print('='*60)
    print(df.to_string(index=False))
    print('\nPaper ref (M1 bounded): ROC≈40.92%, SR≈14.33, CPS≈1.96c')
