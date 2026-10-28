"""
Experiment 4: Shrinkage-based risk models vs traditional truncated-PC models.

Reproduces the paper's bounded backtests comparing diagonal-target shrinkage
and uniform-correlation shrinkage against traditional truncated-PC models.

Paper reference: Kakushadze (2015), shrinkage comparison section.
"""

import logging
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

# Paper shrinkage parameter grids
Q_GRID = [1e-6, 0.3, 0.6, 0.9]
RHO_GRID = [0.1, 0.5, 0.9]

# Paper reference values for plausibility
PAPER_REF_SHRINKAGE = {
    'diag': {'roc_pct': 40.66, 'sharpe': 13.82, 'cps_cents': 1.81},
    'uniform': {'roc_pct': 40.97, 'sharpe': 14.01, 'cps_cents': 1.82},
}


def build_universe_schedule(
    unadj_close: pd.DataFrame,
    unadj_volume: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    top_n: int = 2000,
    block_size: int = 21,
) -> dict:
    addv_df = compute_addv(unadj_close, unadj_volume)
    blocks = build_rebalance_schedule(trading_days, block_size)
    return {bs: select_universe(addv_df, bs, top_n=top_n) for bs, _ in blocks}


def run_experiment_4(
    tickers: list = None,
    top_n: int = None,
    cache_path: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Experiment 4: shrinkage model comparisons.
    
    Returns:
        (diag_df, uniform_df) with performance metrics
    """
    if tickers is None:
        tickers = MEMBERS_FOR_BACKTEST
    if top_n is None:
        top_n = len(tickers)

    logger.info(f"Exp4: Loading data for {len(tickers)} tickers")
    data = load_or_fetch_price_data(tickers, DATA_START, DATA_END, cache_path)

    adj_close    = data['adj_close']
    adj_open     = data['adj_open']
    unadj_close  = data['unadj_close']
    unadj_open   = data['unadj_open']
    unadj_volume = data['unadj_volume']

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
    universe_schedule = build_universe_schedule(unadj_close, unadj_volume, trading_days, top_n)

    # ── Diagonal shrinkage ─────────────────────────────────────────────────
    diag_results = []
    for q in Q_GRID:
        logger.info(f"Exp4: Diagonal shrinkage q={q}")
        bt = run_full_backtest(
            adj_close=adj_close, adj_open=adj_open,
            unadj_open=unadj_open, unadj_close=unadj_close, unadj_volume=unadj_volume,
            universe_schedule=universe_schedule,
            backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
            variant='shrinkage_diag',
            bounded=True,
            shrinkage_params={'q': q},
            verbose=False,
        )
        if not bt:
            continue
        s = bt['summary']
        diag_results.append({
            'q': q, 'roc_pct': s['roc_pct'], 'sharpe': s['sharpe'],
            'cps_cents': s['cps_cents'], 'n_days': s['n_days'],
        })
        bt['daily_pnl'].to_csv(RESULTS_DIR / f'exp4_diag_q{q}_daily_pnl.csv')
        logger.info(f"  q={q}: ROC={s['roc_pct']:.2f}%, SR={s['sharpe']:.2f}, CPS={s['cps_cents']:.4f}c")

    diag_df = pd.DataFrame(diag_results)
    diag_df.to_csv(RESULTS_DIR / 'exp4_shrinkage_diagonal.csv', index=False)

    # ── Uniform correlation shrinkage ──────────────────────────────────────
    uniform_results = []
    for q in [0.3, 0.6, 0.9]:  # subset of grid for efficiency
        for rho in RHO_GRID:
            logger.info(f"Exp4: Uniform shrinkage q={q}, rho={rho}")
            bt = run_full_backtest(
                adj_close=adj_close, adj_open=adj_open,
                unadj_open=unadj_open, unadj_close=unadj_close, unadj_volume=unadj_volume,
                universe_schedule=universe_schedule,
                backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
                variant='shrinkage_uniform',
                bounded=True,
                shrinkage_params={'q': q, 'rho': rho},
                verbose=False,
            )
            if not bt:
                continue
            s = bt['summary']
            uniform_results.append({
                'q': q, 'rho': rho,
                'roc_pct': s['roc_pct'], 'sharpe': s['sharpe'],
                'cps_cents': s['cps_cents'], 'n_days': s['n_days'],
            })
            logger.info(f"  q={q},rho={rho}: ROC={s['roc_pct']:.2f}%, SR={s['sharpe']:.2f}")

    uniform_df = pd.DataFrame(uniform_results)
    uniform_df.to_csv(RESULTS_DIR / 'exp4_shrinkage_uniform.csv', index=False)

    # ── Comparison plot ────────────────────────────────────────────────────
    _plot_shrinkage_comparison(diag_df, uniform_df, RESULTS_DIR)

    return diag_df, uniform_df


def _plot_shrinkage_comparison(
    diag_df: pd.DataFrame,
    uniform_df: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Plot Sharpe vs q for diagonal and uniform shrinkage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Diagonal
    if not diag_df.empty:
        ax = axes[0]
        ax.plot(diag_df['q'], diag_df['sharpe'], 'o-', color='steelblue', linewidth=2)
        ax.set_title('Diagonal Shrinkage: Sharpe vs q', fontsize=12)
        ax.set_xlabel('q (shrinkage parameter)')
        ax.set_ylabel('Annualized Sharpe Ratio')
        ax.grid(True, alpha=0.3)

    # Uniform correlation
    if not uniform_df.empty:
        ax = axes[1]
        colors = plt.cm.tab10(np.linspace(0, 1, len(RHO_GRID)))
        for i, rho in enumerate(RHO_GRID):
            sub = uniform_df[uniform_df['rho'] == rho]
            if not sub.empty:
                ax.plot(sub['q'], sub['sharpe'], 'o-', label=f'rho={rho}',
                        color=colors[i], linewidth=2)
        ax.set_title('Uniform Corr Shrinkage: Sharpe vs q', fontsize=12)
        ax.set_xlabel('q (shrinkage parameter)')
        ax.set_ylabel('Annualized Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Exp4: Shrinkage Model Sensitivity', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(results_dir / 'exp4_shrinkage_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    diag_df, uniform_df = run_experiment_4()
    print('\n' + '='*60)
    print('Experiment 4 – Shrinkage Model Comparison')
    print('='*60)
    print('Diagonal shrinkage:')
    print(diag_df.to_string(index=False))
    print('\nUniform correlation shrinkage:')
    print(uniform_df.to_string(index=False))
