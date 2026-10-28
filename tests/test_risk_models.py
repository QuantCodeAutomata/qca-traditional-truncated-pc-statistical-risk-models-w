"""
Tests for risk model implementations.

Verifies correctness of:
- Sample covariance computation
- Eigendecomposition
- K-selection variants (M1-M6)
- Traditional model construction
- Woodbury identity inversion
- Shrinkage models
- eRank computation
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from src.risk_models import (
    compute_sample_covariance,
    eigen_decompose_correlation,
    compute_erank,
    compute_xi2,
    build_traditional_model,
    minimization_k_selection,
    erank_k_selection,
    select_K,
    build_risk_model,
    build_shrinkage_model_diagonal,
    build_shrinkage_model_uniform_corr,
    RiskModelResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_returns(M: int = 50, N: int = 20, seed: int = 42) -> np.ndarray:
    """Generate synthetic return matrix with factor structure."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((M, 3)) * np.array([3.0, 1.5, 1.0])
    B = rng.standard_normal((N, 3)) * 0.5
    eps = rng.standard_normal((M, N))
    R = F @ B.T + eps
    return R


# ── Sample covariance ──────────────────────────────────────────────────────────

def test_sample_covariance_shape():
    R = make_returns(50, 20)
    C, Psi, sigmas, Y = compute_sample_covariance(R)
    assert C.shape == (20, 20)
    assert Psi.shape == (20, 20)
    assert sigmas.shape == (20,)
    assert Y.shape == (50, 20)


def test_sample_covariance_diagonal_psi():
    """Sample correlation matrix must have unit diagonal."""
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    diag = np.diag(Psi)
    np.testing.assert_allclose(diag, np.ones(20), atol=1e-10)


def test_sample_covariance_symmetry():
    R = make_returns(50, 20)
    C, Psi, _, _ = compute_sample_covariance(R)
    np.testing.assert_allclose(C, C.T, atol=1e-12)
    np.testing.assert_allclose(Psi, Psi.T, atol=1e-12)


def test_sample_covariance_psi_range():
    """Correlation coefficients must be in [-1, 1]."""
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    assert Psi.min() >= -1.001
    assert Psi.max() <= 1.001


def test_normalized_returns_unit_variance():
    """Normalized returns Y should have approximately unit variance per asset."""
    R = make_returns(50, 20)
    _, _, _, Y = compute_sample_covariance(R)
    # Var(Y_i) = (1/M) sum_s Y_is^2 should be 1 for demeaned Y
    var_Y = (Y ** 2).mean(axis=0)
    np.testing.assert_allclose(var_Y, np.ones(20), atol=0.01)


# ── Eigendecomposition ─────────────────────────────────────────────────────────

def test_eigendecomposition_sorted_descending():
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, eigenvectors = eigen_decompose_correlation(Psi)
    assert np.all(np.diff(eigenvalues) <= 1e-10), "Eigenvalues should be sorted descending"


def test_eigendecomposition_orthonormality():
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    _, V = eigen_decompose_correlation(Psi)
    # V^T V = I
    VtV = V.T @ V
    np.testing.assert_allclose(VtV, np.eye(20), atol=1e-8)


def test_eigendecomposition_reconstruction():
    """Psi V = lambda V (eigenvector equation)."""
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    for a in range(5):  # check first 5
        lhs = Psi @ V[:, a]
        rhs = eigenvalues[a] * V[:, a]
        np.testing.assert_allclose(lhs, rhs, atol=1e-8)


def test_eigenvalues_positive():
    """Correlation matrix is PSD so eigenvalues >= 0."""
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, _ = eigen_decompose_correlation(Psi)
    assert eigenvalues.min() >= -1e-10


def test_eigenvalues_sum_to_N():
    """Sum of eigenvalues of correlation matrix = N (trace=N)."""
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, _ = eigen_decompose_correlation(Psi)
    np.testing.assert_allclose(eigenvalues.sum(), 20.0, rtol=1e-10)


# ── eRank ──────────────────────────────────────────────────────────────────────

def test_erank_uniform_eigenvalues():
    """If all N eigenvalues are equal, eRank = N."""
    eigenvalues = np.ones(10)
    er = compute_erank(eigenvalues)
    np.testing.assert_allclose(er, 10.0, atol=1e-10)


def test_erank_single_eigenvalue():
    """If one eigenvalue dominates, eRank = 1."""
    eigenvalues = np.array([1.0, 0.0, 0.0, 0.0])
    er = compute_erank(eigenvalues)
    np.testing.assert_allclose(er, 1.0, atol=1e-10)


def test_erank_between_1_and_N():
    """eRank should be in [1, N] for any set of eigenvalues."""
    rng = np.random.default_rng(7)
    eigenvalues = rng.exponential(1.0, size=20)
    er = compute_erank(eigenvalues)
    assert 1.0 <= er <= 20.0 + 1e-10


def test_erank_positive_only():
    """eRank should only use positive eigenvalues."""
    eigenvalues = np.array([5.0, 3.0, 1.0, 0.0, -0.001])
    er = compute_erank(eigenvalues)
    # Should equal eRank of [5.0, 3.0, 1.0]
    er_ref = compute_erank(np.array([5.0, 3.0, 1.0]))
    np.testing.assert_allclose(er, er_ref, atol=1e-10)


# ── Specific variances ─────────────────────────────────────────────────────────

def test_xi2_sums_correctly():
    """xi_i^2 = 1 - sum_{A=1}^K lambda^(A) (V_i^(A))^2."""
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    K = 5
    xi2 = compute_xi2(eigenvalues, V, K)
    # Manual computation
    xi2_manual = 1.0 - np.sum(eigenvalues[:K] * (V[:, :K] ** 2), axis=1)
    np.testing.assert_allclose(xi2, xi2_manual, atol=1e-12)


def test_xi2_positive_for_small_k():
    """For K < rank(Psi), xi_i^2 should be positive."""
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    xi2 = compute_xi2(eigenvalues, V, K=3)
    assert np.all(xi2 > -1e-10)


def test_xi2_full_rank_approximately_zero():
    """For K = N (full rank), xi_i^2 ≈ 0."""
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    xi2 = compute_xi2(eigenvalues, V, K=20)
    # Should be close to 0 (sum of all eigenvector squares times eigenvalues = 1)
    np.testing.assert_allclose(xi2, np.zeros(20), atol=1e-8)


# ── Traditional model ──────────────────────────────────────────────────────────

def test_traditional_model_diagonal_covariance():
    """Model covariance diagonal should equal sigma_i^2 (in-sample variance preservation)."""
    R = make_returns(50, 20)
    C, Psi, sigmas, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    K = 5
    xi2, Gammahat, _ = build_traditional_model(eigenvalues, V, sigmas, K)
    # Diagonal of Gammahat = sigma_i^2 * (xi_i^2 + sum_A lambda_A V_iA^2) = sigma_i^2
    expected_diag = sigmas ** 2
    np.testing.assert_allclose(np.diag(Gammahat), expected_diag, rtol=1e-6)


def test_traditional_model_correlation_diagonal_one():
    """Model correlation (Gamma) should have unit diagonal."""
    R = make_returns(50, 20)
    C, Psi, sigmas, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    xi2, Gammahat, _ = build_traditional_model(eigenvalues, V, sigmas, K=5)
    Gamma_diag = np.diag(Gammahat) / (sigmas ** 2)
    np.testing.assert_allclose(Gamma_diag, np.ones(20), atol=1e-8)


def test_traditional_model_positive_definite():
    """Model covariance should be positive definite."""
    R = make_returns(50, 20)
    C, Psi, sigmas, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    xi2, Gammahat, _ = build_traditional_model(eigenvalues, V, sigmas, K=5)
    min_eig = np.linalg.eigvalsh(Gammahat).min()
    assert min_eig > -1e-8, f"Model not PSD: min eigenvalue = {min_eig}"


def test_woodbury_inverse_accuracy():
    """Woodbury inverse should match direct inverse."""
    R = make_returns(50, 20)
    C, Psi, sigmas, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    _, Gammahat, Gammahat_inv_woodbury = build_traditional_model(
        eigenvalues, V, sigmas, K=5, use_woodbury=True
    )
    from scipy.linalg import inv
    Gammahat_inv_direct = inv(Gammahat)
    np.testing.assert_allclose(Gammahat_inv_woodbury, Gammahat_inv_direct, rtol=1e-4, atol=1e-8)


def test_model_covariance_invertibility():
    """Gammahat @ Gammahat_inv should be close to identity."""
    R = make_returns(50, 20)
    C, Psi, sigmas, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    _, Gammahat, Gammahat_inv = build_traditional_model(eigenvalues, V, sigmas, K=5)
    I_approx = Gammahat @ Gammahat_inv
    np.testing.assert_allclose(I_approx, np.eye(20), atol=1e-6)


# ── K selection ────────────────────────────────────────────────────────────────

def test_minimization_k_returns_valid_k():
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    K = minimization_k_selection(eigenvalues, V, max_K=19)
    assert 1 <= K <= 19


def test_minimization_k_exclude_first_gte_2():
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    K = minimization_k_selection(eigenvalues, V, max_K=19, exclude_first=True)
    assert K >= 2


def test_erank_k_round_floor():
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, _ = eigen_decompose_correlation(Psi)
    K_round, er_round = erank_k_selection(eigenvalues, method='round', max_K=19)
    K_floor, er_floor = erank_k_selection(eigenvalues, method='floor', max_K=19)
    assert 1 <= K_round <= 19
    assert 1 <= K_floor <= 19
    assert K_round >= K_floor  # round >= floor always
    np.testing.assert_allclose(er_round, er_floor, atol=1e-10)


def test_select_k_all_variants():
    """All six variants should return valid K."""
    R = make_returns(50, 20)
    _, Psi, _, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    M = 50
    for variant in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
        K, erank_val = select_K(variant, eigenvalues, V, M)
        assert 1 <= K <= M - 1, f"Invalid K={K} for variant {variant}"


# ── Full model pipeline ────────────────────────────────────────────────────────

def test_build_risk_model_m1():
    R = make_returns(50, 20)
    tickers = [f'T{i}' for i in range(20)]
    model = build_risk_model(R, tickers, variant='M1')
    assert isinstance(model, RiskModelResult)
    assert model.K >= 1
    assert model.Gammahat is not None
    assert model.Gammahat_inv is not None
    assert model.Gammahat.shape == (20, 20)


def test_build_risk_model_fixed_k():
    R = make_returns(50, 20)
    tickers = [f'T{i}' for i in range(20)]
    model = build_risk_model(R, tickers, variant='fixed', K_override=5)
    assert model.K == 5


def test_build_risk_model_xi2_positive():
    R = make_returns(50, 20)
    tickers = [f'T{i}' for i in range(20)]
    model = build_risk_model(R, tickers, variant='M1')
    assert np.all(model.xi2 > 0), f"xi2 has non-positive values: {model.xi2.min()}"


# ── Shrinkage models ───────────────────────────────────────────────────────────

def test_shrinkage_diagonal_model():
    R = make_returns(50, 20)
    tickers = [f'T{i}' for i in range(20)]
    model = build_shrinkage_model_diagonal(R, tickers, q=0.3)
    assert model.Gammahat is not None
    assert model.Gammahat_inv is not None
    # xi2 should all equal q
    np.testing.assert_allclose(model.xi2, 0.3 * np.ones(20), atol=1e-10)


def test_shrinkage_diagonal_pd():
    """Diagonal shrinkage model should be positive definite."""
    R = make_returns(50, 20)
    tickers = [f'T{i}' for i in range(20)]
    for q in [1e-6, 0.3, 0.6, 0.9]:
        model = build_shrinkage_model_diagonal(R, tickers, q=q)
        min_eig = np.linalg.eigvalsh(model.Gammahat).min()
        assert min_eig > -1e-8, f"Not PSD for q={q}: min_eig={min_eig}"


def test_shrinkage_uniform_model():
    R = make_returns(50, 20)
    tickers = [f'T{i}' for i in range(20)]
    model = build_shrinkage_model_uniform_corr(R, tickers, q=0.3, rho=0.3)
    assert model.Gammahat is not None
    assert model.Gammahat_inv is not None
    # xi2 should equal q*(1-rho)
    expected_xi2 = 0.3 * (1 - 0.3)
    np.testing.assert_allclose(model.xi2, expected_xi2 * np.ones(20), atol=1e-10)


def test_shrinkage_diagonal_q_insensitivity():
    """Model structure (Gammahat eigenvalue count) should vary with q."""
    R = make_returns(50, 20)
    tickers = [f'T{i}' for i in range(20)]
    models = [build_shrinkage_model_diagonal(R, tickers, q=q) for q in [0.1, 0.5, 0.9]]
    # All should be valid
    for m in models:
        assert np.linalg.matrix_rank(m.Gammahat) == 20


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_single_asset_returns():
    """Test with minimum viable N=2."""
    rng = np.random.default_rng(0)
    R = rng.standard_normal((30, 2))
    C, Psi, sigmas, Y = compute_sample_covariance(R)
    assert Psi.shape == (2, 2)
    np.testing.assert_allclose(np.diag(Psi), [1.0, 1.0], atol=1e-10)


def test_very_large_returns():
    """Extreme values should not cause NaN in computations."""
    rng = np.random.default_rng(1)
    R = rng.standard_normal((50, 10)) * 100.0
    C, Psi, sigmas, Y = compute_sample_covariance(R)
    assert np.all(np.isfinite(Psi))
    assert np.all(np.isfinite(sigmas))


def test_constant_asset_returns():
    """Near-constant returns should be handled without division by zero."""
    R = np.zeros((50, 10))
    R[:, 0] = 1e-15  # Near-zero volatility
    C, Psi, sigmas, Y = compute_sample_covariance(R)
    assert np.all(np.isfinite(sigmas))
    assert np.all(sigmas > 0)


# ── Mathematical properties ────────────────────────────────────────────────────

def test_traditional_model_preserves_variances():
    """The traditional model preserves in-sample variances: diag(Gammahat)=diag(C)."""
    R = make_returns(50, 20)
    C, Psi, sigmas, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    xi2, Gammahat, _ = build_traditional_model(eigenvalues, V, sigmas, K=5)
    expected = sigmas ** 2  # = diag(C)
    np.testing.assert_allclose(np.diag(Gammahat), expected, rtol=1e-8)


def test_erank_log_formula():
    """Verify eRank = exp(-sum_a p_a ln p_a) explicitly."""
    eigenvalues = np.array([10.0, 5.0, 2.0, 1.0])
    p = eigenvalues / eigenvalues.sum()
    H_manual = -np.sum(p * np.log(p))
    er_manual = np.exp(H_manual)
    er_computed = compute_erank(eigenvalues)
    np.testing.assert_allclose(er_computed, er_manual, atol=1e-14)


# ── Portfolio optimization ─────────────────────────────────────────────────────

def test_unbounded_holdings_dollar_neutral():
    """Unbounded holdings must be dollar neutral: sum(H) = 0."""
    from src.portfolio import compute_unbounded_holdings
    R = make_returns(50, 20)
    C, Psi, sigmas, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    _, _, Ginv = build_traditional_model(eigenvalues, V, sigmas, K=5)
    alpha = np.random.default_rng(5).standard_normal(20)
    H = compute_unbounded_holdings(Ginv, alpha, gross_exposure=1e6)
    assert np.abs(H.sum()) < 1.0, f"Not dollar neutral: sum(H)={H.sum():.2f}"


def test_unbounded_holdings_gross_exposure():
    """Unbounded holdings gross exposure should match target."""
    from src.portfolio import compute_unbounded_holdings
    R = make_returns(50, 20)
    C, Psi, sigmas, _ = compute_sample_covariance(R)
    eigenvalues, V = eigen_decompose_correlation(Psi)
    _, _, Ginv = build_traditional_model(eigenvalues, V, sigmas, K=5)
    alpha = np.random.default_rng(6).standard_normal(20)
    target_gross = 20_000_000.0
    H = compute_unbounded_holdings(Ginv, alpha, gross_exposure=target_gross)
    gross = np.abs(H).sum()
    np.testing.assert_allclose(gross, target_gross, rtol=1e-6)


# ── Metrics ────────────────────────────────────────────────────────────────────

def test_roc_formula():
    """ROC = 252 * mean(P&L) / gross_exposure."""
    from src.metrics import compute_roc
    pnl = pd.Series([1000.0] * 252)
    roc = compute_roc(pnl, gross_exposure=20_000_000.0)
    expected = 252 * 1000.0 / 20_000_000.0
    np.testing.assert_allclose(roc, expected, rtol=1e-10)


def test_sharpe_formula():
    """Sharpe = sqrt(252) * mean / std."""
    from src.metrics import compute_annualized_sharpe
    import pandas as pd
    rng = np.random.default_rng(9)
    pnl = pd.Series(rng.normal(1000, 500, 252))
    sr = compute_annualized_sharpe(pnl)
    expected = np.sqrt(252) * pnl.mean() / pnl.std(ddof=1)
    np.testing.assert_allclose(sr, expected, rtol=1e-10)


def test_cps_formula():
    """CPS = total_pnl / total_shares."""
    from src.metrics import compute_cps
    import pandas as pd
    pnl = pd.Series([100.0, 200.0, 150.0])
    shares = pd.Series([1000.0, 2000.0, 1500.0])
    cps = compute_cps(pnl, shares)
    expected = 450.0 / 4500.0
    np.testing.assert_allclose(cps, expected, rtol=1e-10)


def test_average_correlation_formula():
    """Average correlation = (1/N^2) * sum_ij Psi_ij."""
    from src.metrics import compute_average_correlation
    Psi = np.eye(5) * 1.0 + np.ones((5, 5)) * 0.3
    np.fill_diagonal(Psi, 1.0)
    avg_corr = compute_average_correlation(Psi)
    # (1/25) * (5*1.0 + 20*0.3) = (5 + 6)/25 = 11/25 = 0.44 ??? let me recalculate
    # Psi_ij = 1.0 for i=j (5 terms), 0.3 for i!=j (20 terms)
    # but above code adds: 1.0*I + 0.3*ones(5,5), so Psi_ii = 1.3, Psi_ij = 0.3
    # Hmm, need to be precise. Use known Psi.
    Psi2 = np.full((4, 4), 0.5)
    np.fill_diagonal(Psi2, 1.0)
    avg2 = compute_average_correlation(Psi2)
    expected2 = Psi2.sum() / (4 * 4)
    np.testing.assert_allclose(avg2, expected2, atol=1e-10)


import pandas as pd  # needed for Sharpe/ROC tests
