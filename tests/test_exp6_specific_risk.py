"""Tests for Experiment 6 specific-risk model builders and Woodbury inverse.

Validates:
  - _woodbury_inv identity property for well-conditioned cases
  - build_adhoc_xi1_model: xi2 = 1, holdings finite for all K
  - build_borrowed_xi_model: holdings finite for all K_var including K_var = M-1
  - Near-singular boundary (K_var = M-1) does not crash or produce NaN holdings
"""

import numpy as np
import pytest

from exp.exp6_specific_risk import (
    _woodbury_inv,
    build_adhoc_xi1_model,
    build_borrowed_xi_model,
)
from src.portfolio import compute_unbounded_holdings


@pytest.fixture
def synthetic_returns():
    """N=25, M=10 returns matrix, sigma ≈ 1% daily."""
    rng = np.random.default_rng(0)
    N, M = 25, 10
    returns = rng.standard_normal((M, N)) * 0.01
    tickers = [f"T{i}" for i in range(N)]
    return returns, tickers, N, M


# ── _woodbury_inv ──────────────────────────────────────────────────────────────

class TestWoodburyInv:
    def test_identity_well_conditioned(self):
        """(D + FF^T)(D + FF^T)^{-1} ≈ I for well-conditioned D."""
        rng = np.random.default_rng(1)
        N, K = 20, 4
        D_vec = rng.uniform(0.5, 2.0, N)          # well-conditioned diagonal
        F = rng.standard_normal((N, K)) * 0.3
        Gammahat = np.diag(D_vec) + F @ F.T
        Ginv = _woodbury_inv(D_vec, F, eps_reg=0.0)
        err = np.max(np.abs(Gammahat @ Ginv - np.eye(N)))
        assert err < 1e-8, f"Identity error too large: {err:.2e}"

    def test_eps_reg_stabilises_near_singular(self):
        """With eps_reg, near-singular D_vec does not blow up the inverse."""
        rng = np.random.default_rng(2)
        N, K = 20, 10
        D_vec = np.full(N, 1e-10)      # near-zero specific risk
        F = rng.standard_normal((N, K)) * 0.1
        Gammahat = np.diag(D_vec) + F @ F.T
        eps_reg = 1e-3 * np.trace(Gammahat) / N
        Ginv = _woodbury_inv(D_vec, F, eps_reg=eps_reg)
        assert np.isfinite(Ginv).all(), "Woodbury produced NaN/Inf"

    def test_output_shape(self):
        rng = np.random.default_rng(3)
        N, K = 15, 3
        D_vec = rng.uniform(0.1, 1.0, N)
        F = rng.standard_normal((N, K)) * 0.2
        Ginv = _woodbury_inv(D_vec, F)
        assert Ginv.shape == (N, N)

    def test_symmetry(self):
        """(D + FF^T)^{-1} must be symmetric."""
        rng = np.random.default_rng(4)
        N, K = 18, 5
        D_vec = rng.uniform(0.2, 1.5, N)
        F = rng.standard_normal((N, K)) * 0.25
        Ginv = _woodbury_inv(D_vec, F)
        assert np.allclose(Ginv, Ginv.T, atol=1e-10), "Woodbury result not symmetric"


# ── build_adhoc_xi1_model ──────────────────────────────────────────────────────

class TestAdhocXi1Model:
    def test_xi2_equals_one(self, synthetic_returns):
        returns, tickers, N, M = synthetic_returns
        for K in [1, 3, 5, M - 1]:
            res = build_adhoc_xi1_model(returns, tickers, K)
            assert np.allclose(res.xi2, 1.0), f"xi2 should equal 1 for K={K}"

    def test_K_capped_at_M_minus_1(self, synthetic_returns):
        returns, tickers, N, M = synthetic_returns
        res = build_adhoc_xi1_model(returns, tickers, K=100)
        assert res.K <= M - 1, "K not capped at M-1"

    def test_holdings_finite(self, synthetic_returns):
        """Holdings must be finite for all K in [1, M-1]."""
        returns, tickers, N, M = synthetic_returns
        rng = np.random.default_rng(5)
        alpha = rng.standard_normal(N) * 0.002
        for K in [1, 3, M - 1]:
            res = build_adhoc_xi1_model(returns, tickers, K)
            H = compute_unbounded_holdings(res.Gammahat_inv, alpha, gross_exposure=20e6)
            assert np.isfinite(H).all(), f"Holdings not finite for K={K}"
            assert abs(np.sum(np.abs(H)) - 20e6) < 1.0, "Gross exposure not matched"

    def test_gammahat_shape(self, synthetic_returns):
        returns, tickers, N, M = synthetic_returns
        res = build_adhoc_xi1_model(returns, tickers, K=4)
        assert res.Gammahat.shape == (N, N)
        assert res.Gammahat_inv.shape == (N, N)


# ── build_borrowed_xi_model ────────────────────────────────────────────────────

class TestBorrowedXiModel:
    def test_K_full_is_M_minus_1(self, synthetic_returns):
        returns, tickers, N, M = synthetic_returns
        res = build_borrowed_xi_model(returns, tickers, K_full=100, K_var=3)
        assert res.K == M - 1, "K_full should be capped at M-1"

    def test_holdings_finite_all_K_var(self, synthetic_returns):
        """Holdings must be finite for all K_var including the degenerate K_var=M-1."""
        returns, tickers, N, M = synthetic_returns
        rng = np.random.default_rng(6)
        alpha = rng.standard_normal(N) * 0.002
        for K_var in [1, 3, M - 2, M - 1]:
            res = build_borrowed_xi_model(returns, tickers, K_full=M - 1, K_var=K_var)
            H = compute_unbounded_holdings(res.Gammahat_inv, alpha, gross_exposure=20e6)
            assert np.isfinite(H).all(), f"Holdings not finite for K_var={K_var}"

    def test_xi2_positive(self, synthetic_returns):
        """Borrowed xi2 must remain positive (floor clipped at 1e-8)."""
        returns, tickers, N, M = synthetic_returns
        for K_var in [1, 5, M - 1]:
            res = build_borrowed_xi_model(returns, tickers, K_full=M - 1, K_var=K_var)
            assert (res.xi2 >= 0).all(), "xi2 has negative values"

    def test_gammahat_shape_and_finite(self, synthetic_returns):
        returns, tickers, N, M = synthetic_returns
        res = build_borrowed_xi_model(returns, tickers, K_full=M - 1, K_var=3)
        assert res.Gammahat.shape == (N, N)
        assert np.isfinite(res.Gammahat).all()
        assert np.isfinite(res.Gammahat_inv).all()
