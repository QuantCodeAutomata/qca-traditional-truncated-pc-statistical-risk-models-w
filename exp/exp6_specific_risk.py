"""
Experiment 6: Specific-risk robustness experiments.

Reproduces the paper's Tables 9 and 10: ad hoc specific variances and their
impact on out-of-sample performance.

Study A (Table 9): Fix factor count at K1=M_eff, borrow xi^2 from a varying-K model.
Study B (Table 10): Vary K but set xi_i^2 = 1 (ad hoc specific variance).

Paper reference: Kakushadze (2015), Tables 9-10, Figures 1-2.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_or_fetch_price_data, MEMBERS_FOR_BACKTEST
from src.universe import compute_addv, build_rebalance_schedule, select_universe
from src.backtest import run_block, BLOCK_SIZE, ESTIMATION_LOOKBACK
from src.risk_models import (
    compute_sample_covariance,
    eigen_decompose_correlation,
    build_traditional_model,
    select_K,
    compute_xi2,
    RiskModelResult,
)
from src.portfolio import compute_unbounded_holdings, compute_daily_pnl
from src.metrics import summarize_performance

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

BACKTEST_START = '2009-01-01'
BACKTEST_END   = '2014-09-05'
DATA_START     = '2008-08-01'
DATA_END       = '2014-09-05'
GROSS_EXPOSURE = 20_000_000.0

# K grid for Study B
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


def _woodbury_inv(D_vec: np.ndarray, F: np.ndarray, eps_reg: float = 0.0) -> np.ndarray:
    """Compute (diag(D_vec) + F @ F.T)^{-1} via Woodbury identity.

    (D + F F^T)^{-1} = D^{-1} - D^{-1} F (I_K + F^T D^{-1} F)^{-1} F^T D^{-1}

    Optional trace-based regularisation eps_reg > 0 is added to D_vec before
    inversion to guard against near-singular specific-risk diagonals.
    """
    N, K = F.shape
    D_reg = D_vec + eps_reg  # (N,)
    D_inv = 1.0 / D_reg      # (N,)
    DinvF = F * D_inv[:, None]                    # (N, K)
    M_mat = np.eye(K) + F.T @ DinvF              # (K, K)
    try:
        from scipy.linalg import solve
        M_inv = solve(M_mat, np.eye(K), assume_a='pos')
    except Exception:
        M_inv = np.linalg.pinv(M_mat)
    Gammahat_inv = np.diag(D_inv) - DinvF @ M_inv @ DinvF.T
    return Gammahat_inv


def build_adhoc_xi1_model(
    returns: np.ndarray,
    tickers: list,
    K: int,
) -> RiskModelResult:
    """Build risk model with xi_i^2 = 1 (Study B ad hoc specific variance).
    
    The model Gamma_ij = delta_ij + sum_{A=1}^K lambda^(A) V_i^(A) V_j^(A)
    uses xi_i^2 = 1 for all i, regardless of K.
    Note: diagonal of Gamma is 1 + sum_A lambda^(A) (V_i^(A))^2 != 1 in general.
    
    Args:
        returns: (M, N) return matrix
        tickers: List of tickers
        K: Number of factors
    
    Returns:
        RiskModelResult with ad hoc xi^2=1 model
    """
    M, N = returns.shape
    C, Psi, sigmas, Y = compute_sample_covariance(returns)
    eigenvalues, eigenvectors = eigen_decompose_correlation(Psi)
    
    K = max(1, min(K, M - 1, N - 1, eigenvectors.shape[1]))
    
    # Ad hoc: xi_i^2 = 1 for all i
    xi2_adhoc = np.ones(N)
    
    # Factor loadings in covariance space
    V_K = eigenvectors[:, :K]
    lam_K = eigenvalues[:K]
    F = sigmas[:, None] * V_K * np.sqrt(lam_K)[None, :]  # (N, K)
    D_vec = xi2_adhoc * (sigmas ** 2)                     # (N,)

    Gammahat = np.diag(D_vec) + F @ F.T

    # Trace-based regularisation (D_vec = sigmas^2 is already well-conditioned,
    # but defensive reg keeps condition number bounded regardless of scale)
    eps_reg = 1e-4 * np.trace(Gammahat) / N
    Gammahat_inv = _woodbury_inv(D_vec, F, eps_reg=eps_reg)
    
    return RiskModelResult(
        variant=f'adhoc_xi1_K{K}',
        K=K,
        tickers=tickers,
        V=V_K,
        lambdas=lam_K,
        xi2=xi2_adhoc,
        sigmas=sigmas,
        Gammahat=Gammahat,
        Gammahat_inv=Gammahat_inv,
        all_eigenvalues=eigenvalues,
    )


def build_borrowed_xi_model(
    returns: np.ndarray,
    tickers: list,
    K_full: int,
    K_var: int,
) -> RiskModelResult:
    """Study A: Use K_full factors but borrow xi^2 from K_var model.
    
    Factor count is K_full (maximal), but specific variances come from K_var.
    When K_var is close to M-1, xi2_borrowed → 0, making the specific-risk
    diagonal nearly singular; trace-based regularisation is applied.
    
    Args:
        returns: (M, N) return matrix
        tickers: List of tickers
        K_full: Full factor count (M-1)
        K_var: Varying K that determines xi^2
    
    Returns:
        RiskModelResult
    """
    M, N = returns.shape
    C, Psi, sigmas, Y = compute_sample_covariance(returns)
    eigenvalues, eigenvectors = eigen_decompose_correlation(Psi)
    
    K_full = max(1, min(K_full, M - 1, N - 1))
    K_var = max(1, min(K_var, M - 1, N - 1))
    
    # Borrowed xi^2 from K_var model
    xi2_borrowed = compute_xi2(eigenvalues, eigenvectors, K_var)
    xi2_borrowed = np.maximum(xi2_borrowed, 1e-8)
    
    V_K = eigenvectors[:, :K_full]
    lam_K = eigenvalues[:K_full]
    
    F = sigmas[:, None] * V_K * np.sqrt(lam_K)[None, :]  # (N, K_full)
    D_vec = xi2_borrowed * (sigmas ** 2)                   # (N,)

    Gammahat = np.diag(D_vec) + F @ F.T

    # Trace-based regularisation critical when K_var → M-1 (xi2_borrowed → 0)
    eps_reg = 1e-3 * np.trace(Gammahat) / N
    Gammahat_inv = _woodbury_inv(D_vec, F, eps_reg=eps_reg)
    
    return RiskModelResult(
        variant=f'borrowed_xi_K{K_full}_Kvar{K_var}',
        K=K_full,
        tickers=tickers,
        V=V_K,
        lambdas=lam_K,
        xi2=xi2_borrowed,
        sigmas=sigmas,
        Gammahat=Gammahat,
        Gammahat_inv=Gammahat_inv,
        all_eigenvalues=eigenvalues,
    )


def _run_custom_backtest(
    model_builder,
    adj_close, adj_open, unadj_open, unadj_close, unadj_volume,
    universe_schedule, backtest_start, backtest_end,
    estimation_lookback=ESTIMATION_LOOKBACK,
    block_size=BLOCK_SIZE,
) -> dict:
    """Run a full backtest with a custom model builder function.
    
    model_builder: callable(returns, tickers) -> RiskModelResult
    """
    all_dates = adj_close.index
    backtest_dates = all_dates[
        (all_dates >= pd.Timestamp(backtest_start)) &
        (all_dates <= pd.Timestamp(backtest_end))
    ]
    if len(backtest_dates) == 0:
        return {}

    blocks = build_rebalance_schedule(backtest_dates, block_size)
    all_pnl = {}
    all_shares = {}

    for block_start, block_end in blocks:
        schedule_dates = sorted(universe_schedule.keys())
        matching = [d for d in schedule_dates if d <= block_start]
        if not matching:
            continue
        universe_tickers = universe_schedule[matching[-1]]
        if not universe_tickers:
            continue

        block_days = backtest_dates[
            (backtest_dates >= block_start) & (backtest_dates <= block_end)
        ]

        first_day = block_start
        prior_dates = all_dates[all_dates < first_day]
        if len(prior_dates) < estimation_lookback + 1:
            continue
        est_dates = prior_dates[-(estimation_lookback + 1):]

        avail_tickers = [t for t in universe_tickers if t in adj_close.columns]
        if len(avail_tickers) < 2:
            continue

        est_adj_close = adj_close.loc[est_dates, avail_tickers]
        log_prices = np.log(est_adj_close.replace(0, np.nan).ffill())
        raw_returns = log_prices.diff(1).dropna()
        valid_tickers = raw_returns.columns[raw_returns.notna().all()].tolist()
        if len(valid_tickers) < 2:
            continue

        returns_matrix = raw_returns[valid_tickers].values
        M, N = returns_matrix.shape
        if M < 5 or N < 2:
            continue

        try:
            model = model_builder(returns_matrix, valid_tickers)
        except Exception as e:
            logger.debug(f"Model build failed {block_start}: {e}")
            continue

        if model is None or model.Gammahat_inv is None:
            continue

        for trade_date in block_days:
            if trade_date not in adj_open.index:
                continue
            prior_trade = all_dates[all_dates < trade_date]
            if len(prior_trade) == 0:
                continue
            prior_date = prior_trade[-1]
            if prior_date not in adj_close.index:
                continue

            adj_open_today = adj_open.loc[trade_date, valid_tickers]
            adj_close_prior = adj_close.loc[prior_date, valid_tickers]
            with np.errstate(divide='ignore', invalid='ignore'):
                overnight = np.log(adj_open_today.values / adj_close_prior.values)
            overnight = np.where(np.isfinite(overnight), overnight, 0.0)

            if trade_date not in unadj_open.index or trade_date not in unadj_close.index:
                continue
            po = unadj_open.loc[trade_date, valid_tickers].values
            pc = unadj_close.loc[trade_date, valid_tickers].values

            try:
                H = compute_unbounded_holdings(model.Gammahat_inv, overnight, GROSS_EXPOSURE)
            except Exception:
                continue

            if H is None or not np.isfinite(H).all():
                continue

            pnl, shares = compute_daily_pnl(H, po, pc)
            all_pnl[trade_date] = pnl
            all_shares[trade_date] = shares

    pnl_series = pd.Series(all_pnl).sort_index()
    shares_series = pd.Series(all_shares).sort_index()
    summary = summarize_performance(pnl_series, shares_series, GROSS_EXPOSURE)
    return {'daily_pnl': pnl_series, 'daily_shares': shares_series, 'summary': summary}


def run_experiment_6(
    tickers: list = None,
    top_n: int = None,
    cache_path: str = None,
    k_grid: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Experiment 6: specific-risk robustness studies.
    
    Returns:
        (study_a_df, study_b_df)
    """
    if tickers is None:
        tickers = MEMBERS_FOR_BACKTEST
    if top_n is None:
        top_n = len(tickers)
    if k_grid is None:
        k_grid = K_GRID

    logger.info(f"Exp6: Loading data for {len(tickers)} tickers")
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

    base_kwargs = dict(
        adj_close=adj_close, adj_open=adj_open,
        unadj_open=unadj_open, unadj_close=unadj_close, unadj_volume=unadj_volume,
        universe_schedule=universe_schedule,
        backtest_start=BACKTEST_START, backtest_end=BACKTEST_END,
    )

    # ── Study A: K_full=M-1, varying xi^2 from K_var ─────────────────────
    logger.info("Exp6: Study A – borrowed specific risks")
    study_a_results = []
    for K_var in k_grid:
        logger.info(f"  K_var={K_var}")

        def make_builder(kv):
            def builder(returns, tickers_list):
                M_eff = returns.shape[0] - 1
                K_full = max(1, min(M_eff, returns.shape[1] - 1))
                return build_borrowed_xi_model(returns, tickers_list, K_full, kv)
            return builder

        bt = _run_custom_backtest(make_builder(K_var), **base_kwargs)
        if not bt:
            continue
        s = bt['summary']
        study_a_results.append({
            'K_var': K_var,
            'roc_pct': s['roc_pct'],
            'sharpe': s['sharpe'],
            'cps_cents': s['cps_cents'],
        })
        bt['daily_pnl'].to_csv(RESULTS_DIR / f'exp6_studyA_Kvar{K_var}_daily_pnl.csv')
        logger.info(f"    K_var={K_var}: SR={s['sharpe']:.2f}")

    study_a_df = pd.DataFrame(study_a_results)
    study_a_df.to_csv(RESULTS_DIR / 'exp6_study_a.csv', index=False)

    # ── Study B: varying K, xi_i^2 = 1 ────────────────────────────────────
    logger.info("Exp6: Study B – ad hoc xi_i^2 = 1")
    study_b_results = []
    for K in k_grid:
        logger.info(f"  K={K}")

        def make_builder_b(k):
            def builder(returns, tickers_list):
                return build_adhoc_xi1_model(returns, tickers_list, k)
            return builder

        bt = _run_custom_backtest(make_builder_b(K), **base_kwargs)
        if not bt:
            continue
        s = bt['summary']
        study_b_results.append({
            'K': K,
            'roc_pct': s['roc_pct'],
            'sharpe': s['sharpe'],
            'cps_cents': s['cps_cents'],
        })
        bt['daily_pnl'].to_csv(RESULTS_DIR / f'exp6_studyB_K{K}_daily_pnl.csv')
        logger.info(f"    K={K}: SR={s['sharpe']:.2f}")

    study_b_df = pd.DataFrame(study_b_results)
    study_b_df.to_csv(RESULTS_DIR / 'exp6_study_b.csv', index=False)

    # ── Plots (Figures 1 and 2) ────────────────────────────────────────────
    _plot_specific_risk_figures(study_a_df, study_b_df, RESULTS_DIR)

    return study_a_df, study_b_df


def _plot_specific_risk_figures(
    study_a_df: pd.DataFrame,
    study_b_df: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Generate Figures 1 and 2: Sharpe vs K for Studies A and B."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Figure 1: Study A
    if not study_a_df.empty:
        ax = axes[0]
        ax.plot(study_a_df['K_var'], study_a_df['sharpe'], 'o-',
                color='steelblue', linewidth=2, label='Borrowed xi2(K_var)')
        ax.set_title('Fig1 (Exp6 Study A): Sharpe vs K_var\n(K_full=M-1, xi² borrowed)', fontsize=11)
        ax.set_xlabel('K_var (source of xi²)')
        ax.set_ylabel('Annualized Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Figure 2: Study B
    if not study_b_df.empty:
        ax = axes[1]
        ax.plot(study_b_df['K'], study_b_df['sharpe'], 's-',
                color='darkorange', linewidth=2, label='Ad hoc xi²=1')
        ax.set_title('Fig2 (Exp6 Study B): Sharpe vs K\n(xi²=1 fixed)', fontsize=11)
        ax.set_xlabel('K (number of factors)')
        ax.set_ylabel('Annualized Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle('Exp6: Specific-Risk Robustness (Figures 1-2)', fontsize=13)
    plt.tight_layout()
    fig.savefig(results_dir / 'exp6_specific_risk_figures.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    a_df, b_df = run_experiment_6()
    print('\n' + '='*60)
    print('Experiment 6 – Specific Risk Robustness')
    print('='*60)
    print('Study A (Borrowed xi²):')
    print(a_df.to_string(index=False))
    print('\nStudy B (Ad hoc xi²=1):')
    print(b_df.to_string(index=False))
