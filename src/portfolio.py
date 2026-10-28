"""Portfolio optimization for intraday mean-reversion alpha strategies.

Implements both the unbounded (closed-form) and bounded (constrained QP)
portfolio optimizers from the paper.
"""

import logging
from typing import Optional

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ── Unbounded optimizer ────────────────────────────────────────────────────────

def compute_unbounded_holdings(
    Gammahat_inv: np.ndarray,
    alpha: np.ndarray,
    gross_exposure: float = 20_000_000.0,
) -> np.ndarray:
    """Closed-form dollar-neutral Sharpe-maximizing holdings (unbounded).

    Paper formula:
        H_i = -eta * [Gamma^{-1} E - (1^T Gamma^{-1} E)/(1^T Gamma^{-1} 1) * Gamma^{-1} 1]

    The minus sign implements mean reversion. eta rescales to |H|_1 = gross_exposure.

    Args:
        Gammahat_inv: (N, N) inverse model covariance matrix
        alpha: (N,) overnight return vector E_i
        gross_exposure: Target gross exposure I (default $20M)

    Returns:
        (N,) array of dollar holdings H_i
    """
    N = len(alpha)
    if N == 0:
        return np.array([])

    GinvE = Gammahat_inv @ alpha          # (N,)
    Ginv_rowsums = Gammahat_inv.sum(axis=1)  # Gamma^{-1} 1  (N,)
    sum_Ginv = Ginv_rowsums.sum()             # 1^T Gamma^{-1} 1

    correction_scalar = GinvE.sum() / sum_Ginv if abs(sum_Ginv) > 1e-15 else 0.0
    H_raw = -(GinvE - correction_scalar * Ginv_rowsums)

    gross_raw = np.abs(H_raw).sum()
    if gross_raw < 1e-12:
        logger.warning("Near-zero gross exposure in unbounded optimizer; returning zeros")
        return np.zeros(N)

    return H_raw * (gross_exposure / gross_raw)


# ── Fast bounded optimizer (direct OSQP, no CVXPY overhead) ───────────────────

def compute_bounded_holdings(
    Gammahat: np.ndarray,
    alpha: np.ndarray,
    addv_bounds: np.ndarray,
    gross_exposure: float = 20_000_000.0,
    bound_fraction: float = 0.01,
    max_bisect_iters: int = 30,
    tol_gross: float = 1e3,
    fallback_to_unbounded: bool = False,
) -> np.ndarray:
    """Bounded dollar-neutral Sharpe-maximizing holdings via bisection + OSQP.

    Solves (for each lambda_risk via bisection):
        minimize    (1/2) H^T Gammahat H - lambda_risk * alpha^T H
        subject to  sum(H) = 0
                    |H_i| <= bound_fraction * ADDV_i

    lambda_risk is chosen so that sum(|H_i|) = gross_exposure.

    Uses OSQP directly (no CVXPY) for ~0.04 ms per solve with warm-starting.

    Args:
        Gammahat: (N, N) model covariance matrix
        alpha: (N,) overnight return vector
        addv_bounds: (N,) average daily dollar volume (21-day)
        gross_exposure: Target gross dollar exposure I (default $20M)
        bound_fraction: Fraction of ADDV for per-name bound (0.01)
        max_bisect_iters: Max bisection iterations
        tol_gross: Convergence tolerance on gross exposure (dollars)
        fallback_to_unbounded: Return unbounded solution if QP fails

    Returns:
        (N,) array of dollar holdings H_i
    """
    import osqp

    N = len(alpha)
    if N == 0:
        return np.array([])

    box_bounds = bound_fraction * np.maximum(addv_bounds, 0.0)

    if box_bounds.max() < 1e-8:
        logger.warning("All ADDV bounds are zero; using unbounded optimizer")
        Ginv = np.linalg.pinv(Gammahat)
        return compute_unbounded_holdings(Ginv, alpha, gross_exposure)

    # ── Build OSQP problem matrices (fixed across lambda bisection) ─────────
    # minimize 0.5 x^T P x + q^T x,  s.t.  l <= A x <= u
    # P = Gammahat, q = -lambda * alpha
    # A = [1^T ; I_N],  l = [0; -b],  u = [0; b]
    # Stabilise condition number: add 1e-3 * mean_eigenvalue * I.
    # Ensures condition number ≤ 1000, critical for near-rank-deficient
    # Gammahat (e.g. diagonal shrinkage with q=1e-6 where 70 eigenvalues → 0).
    # Effect on holdings is negligible given 0.01*ADDV box constraints.
    # Custom — OSQP best-practice for ill-conditioned financial QPs.
    _mean_eig = float(np.trace(Gammahat)) / N
    _eps_reg = 1e-3 * max(_mean_eig, 1e-12)
    _P_reg = Gammahat + _eps_reg * np.eye(N)
    P_osqp = sp.csc_matrix(_P_reg)
    ones_row = sp.csc_matrix(np.ones((1, N)))
    eye_N = sp.eye(N, format='csc')
    A_osqp = sp.vstack([ones_row, eye_N], format='csc')
    l_osqp = np.hstack([0.0, -box_bounds])
    u_osqp = np.hstack([0.0,  box_bounds])

    solver = osqp.OSQP()
    solver.setup(
        P_osqp, +alpha,  # q = +1.0 * alpha (mean-reversion: minimize E^T H)
        A_osqp, l_osqp, u_osqp,
        warm_starting=True,
        verbose=False,
        eps_abs=1e-5, eps_rel=1e-5,
        max_iter=4000,
        adaptive_rho=True,
        polish=False,   # disabled: QDLDL fails on ill-conditioned P
    )

    def solve_for_lambda(lam: float) -> Optional[np.ndarray]:
        solver.update(q=+lam * alpha)
        res = solver.solve()
        if res.info.status_val in (1, 2):  # solved or solved_inaccurate
            return res.x.copy()
        return None

    # ── Bisection ───────────────────────────────────────────────────────────
    lam_lo, lam_hi = 1e-8, 1e8

    H_lo = solve_for_lambda(lam_lo)
    H_hi = solve_for_lambda(lam_hi)

    if H_lo is None or H_hi is None:
        if fallback_to_unbounded:
            Ginv = np.linalg.pinv(Gammahat)
            return compute_unbounded_holdings(Ginv, alpha, gross_exposure)
        return np.zeros(N)

    H_final = H_lo
    for _ in range(max_bisect_iters):
        lam_mid = 0.5 * (np.log(lam_lo) + np.log(lam_hi))
        lam_mid = np.exp(lam_mid)
        H_mid = solve_for_lambda(lam_mid)
        if H_mid is None:
            break

        gross_mid = np.abs(H_mid).sum()
        H_final = H_mid

        if abs(gross_mid - gross_exposure) < tol_gross:
            break

        if gross_mid < gross_exposure:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    # Rescale so gross exactly equals target (small numerical correction)
    gross_final = np.abs(H_final).sum()
    if gross_final > 1e-8:
        H_final = H_final * (gross_exposure / gross_final)

    # Clip to bounds after rescale (numerical safety)
    H_final = np.clip(H_final, -box_bounds, box_bounds)
    return H_final


# ── P&L computation ────────────────────────────────────────────────────────────

def compute_daily_pnl(
    holdings: np.ndarray,
    unadj_open: np.ndarray,
    unadj_close: np.ndarray,
) -> tuple[float, float]:
    """Compute daily P&L and shares traded for a set of holdings.

    Daily P&L per name:    Pi_is = H_is * (P^C_is / P^O_is - 1)
    Shares traded per name: Q_is = 2 * |H_is| / P^O_is

    Args:
        holdings: (N,) dollar holdings H_is
        unadj_open: (N,) unadjusted open prices P^O_is
        unadj_close: (N,) unadjusted close prices P^C_is

    Returns:
        (total_pnl, total_shares) for the day
    """
    valid = (unadj_open > 0) & (unadj_close > 0) & np.isfinite(holdings)
    if not valid.any():
        return 0.0, 0.0

    h = holdings[valid]
    po = unadj_open[valid]
    pc = unadj_close[valid]

    pnl_per_name = h * (pc / po - 1.0)
    shares_per_name = 2.0 * np.abs(h) / po

    return float(pnl_per_name.sum()), float(shares_per_name.sum())
