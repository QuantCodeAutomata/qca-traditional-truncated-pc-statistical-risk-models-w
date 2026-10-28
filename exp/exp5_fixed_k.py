"""
Experiment 5: Fixed-K sensitivity analysis for traditional statistical risk models.

Reproduces Tables 7 and 8 from the paper: performance as a function of fixed K
for both unbounded and bounded portfolio optimization.

Paper reference: Kakushadze (2015), Tables 7-8.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

BACKTEST_START = '2009-01-01'
BACKTEST_END   = '2014-09-05'
DATA_START     = '2008-08-01'
DATA_END       = '2014-09-05'

# K grid to test (paper reports K=1..M-1; focus on K=1..25 for efficiency)
K_GRID = list(range(1, 20))


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


def run_experiment_5(
    tickers: list = None,
    top_n: int = None,
    cache_path: str = None,
    k_grid: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Experiment 5: fixed-K sensitivity analysis.
    
    Returns:
        (unbounded_df, bounded_df) with performance metrics for each K.
    """
    if tickers is None:
        tickers = MEMBERS_FOR_BACKTEST
    if top_n is None:
        top_n = len(tickers)
    if k_grid is None:
        k_grid = K_GRID

    logger.info(f"Exp5: Loading data for {len(tickers)} tickers")
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

    unbounded_results = []
    bounded_results = []

    for K in k_grid:
        logger.info(f"Exp5: K={K}")
        
        # Unbounded
        bt_unb = run_full_backtest(
            adj_close=adj_close, adj_open=adj_open,
            unadj_open=unadj_open, unadj_close=unadj_close, unadj_volume=unadj_volume,
            universe_schedule=universe_schedule,
            backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
            variant='M1',  # variant irrelevant for fixed K
            bounded=False, K_override=K, verbose=False,
        )
        if bt_unb:
            s = bt_unb['summary']
            unbounded_results.append({
                'K': K, 'roc_pct': s['roc_pct'], 'sharpe': s['sharpe'],
                'cps_cents': s['cps_cents'], 'n_days': s['n_days'],
            })
            bt_unb['daily_pnl'].to_csv(RESULTS_DIR / f'exp5_unb_K{K}_daily_pnl.csv')
        
        # Bounded
        bt_bnd = run_full_backtest(
            adj_close=adj_close, adj_open=adj_open,
            unadj_open=unadj_open, unadj_close=unadj_close, unadj_volume=unadj_volume,
            universe_schedule=universe_schedule,
            backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
            variant='M1',  # variant irrelevant for fixed K
            bounded=True, K_override=K, verbose=False,
        )
        if bt_bnd:
            s = bt_bnd['summary']
            bounded_results.append({
                'K': K, 'roc_pct': s['roc_pct'], 'sharpe': s['sharpe'],
                'cps_cents': s['cps_cents'], 'n_days': s['n_days'],
            })
        
        if unbounded_results and bounded_results:
            logger.info(f"  K={K}: unb SR={unbounded_results[-1]['sharpe']:.2f}, "
                        f"bnd SR={bounded_results[-1]['sharpe']:.2f}")

    unb_df = pd.DataFrame(unbounded_results)
    bnd_df = pd.DataFrame(bounded_results)

    unb_df.to_csv(RESULTS_DIR / 'exp5_fixed_k_unbounded.csv', index=False)
    bnd_df.to_csv(RESULTS_DIR / 'exp5_fixed_k_bounded.csv', index=False)
    
    # Plot
    _plot_fixed_k_sensitivity(unb_df, bnd_df, RESULTS_DIR)
    
    return unb_df, bnd_df


def _plot_fixed_k_sensitivity(
    unb_df: pd.DataFrame,
    bnd_df: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Plot performance metrics versus fixed K."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [('roc_pct', 'ROC (%)'), ('sharpe', 'Annualized Sharpe'), ('cps_cents', 'CPS (cents)')]
    
    for ax, (metric, title) in zip(axes, metrics):
        if not unb_df.empty:
            ax.plot(unb_df['K'], unb_df[metric], 'o-', label='Unbounded', color='steelblue', linewidth=2)
        if not bnd_df.empty:
            ax.plot(bnd_df['K'], bnd_df[metric], 's-', label='Bounded', color='darkorange', linewidth=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('K (number of factors)')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Exp5: Performance vs Fixed K', fontsize=14)
    plt.tight_layout()
    fig.savefig(results_dir / 'exp5_fixed_k_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    unb_df, bnd_df = run_experiment_5()
    print('\n' + '='*60)
    print('Experiment 5 – Fixed-K Sensitivity (Tables 7-8)')
    print('='*60)
    print('Unbounded:')
    print(unb_df.to_string(index=False))
    print('\nBounded:')
    print(bnd_df.to_string(index=False))
    print('\nPaper: best unbounded at K~10, best bounded at K~9-10')
