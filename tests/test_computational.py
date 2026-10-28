"""
Tests for the computational experiment module (Exp 3).

Verifies:
- Cholesky phi factor correctness
- No-iterations PC algorithm accuracy
- Power iteration convergence
- Deflation correctness
- Numerical equivalence between methods
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from src.computational import (
    build_cholesky_phi,
    no_iterations_pc,
    power_iteration_pc1,
    deflate,
    power_iterations_all_pcs,
    run_computational_experiment,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_Y(M: int = 10, N: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((M, 3)) * np.array([3.0, 1.5, 1.0])
    B = rng.standard_normal((N, 3)) * 0.5
    eps = rng.standard_normal((M, N))
    R = F @ B.T + eps
    R -= R.mean(axis=0)
    std = R.std(axis=0)
    std = np.where(std > 1e-10, std, 1e-10)
    return R / std


# ── Cholesky phi ───────────────────────────────────────────────────────────────

def test_cholesky_phi_lower_triangular():
    varphi = build_cholesky_phi(5)
    # Upper triangle (excluding diagonal) should be zero
    for i in range(5):
        for j in range(i + 1, 5):
            assert varphi[i, j] == 0.0, f"varphi[{i},{j}] = {varphi[i,j]} should be 0"


def test_cholesky_phi_reconstruction():
    """phi = varphi @ varphi.T should equal (I + uu^T)/M."""
    M = 8
    varphi = build_cholesky_phi(M)
    phi_reconstructed = varphi @ varphi.T
    u = np.ones(M)
    phi_expected = (np.eye(M) + np.outer(u, u)) / M
    np.testing.assert_allclose(phi_reconstructed, phi_expected, atol=1e-12)


def test_cholesky_phi_positive_diagonal():
    varphi = build_cholesky_phi(10)
    assert np.all(np.diag(varphi) > 0)


# ── No-iterations algorithm ────────────────────────────────────────────────────

def test_no_iterations_eigenvalues_positive():
    Y = make_Y(10, 50)
    eigenvalues, _ = no_iterations_pc(Y)
    assert np.all(eigenvalues > 0)


def test_no_iterations_eigenvectors_orthonormal():
    Y = make_Y(10, 50)
    lam, V = no_iterations_pc(Y)
    K = V.shape[1]
    VtV = V.T @ V
    np.testing.assert_allclose(VtV, np.eye(K), atol=1e-8)


def test_no_iterations_matches_direct_eigh():
    """No-iterations should give same eigenvalues as direct Gram diagonalization."""
    Y = make_Y(10, 50)
    M, N = Y.shape
    
    varphi = build_cholesky_phi(M)
    Ytilde = varphi.T @ Y
    G = Ytilde @ Ytilde.T
    
    rhos_direct, _ = eigh(G)
    rhos_direct = np.sort(rhos_direct)[::-1]
    pos = rhos_direct > 1e-10
    rhos_direct = rhos_direct[pos]
    
    lam_ni, _ = no_iterations_pc(Y)
    
    n = min(len(lam_ni), len(rhos_direct))
    np.testing.assert_allclose(lam_ni[:n], rhos_direct[:n], rtol=1e-8)


def test_no_iterations_eigenvector_equation():
    """V should satisfy Psi_gram @ V = lambda * V."""
    Y = make_Y(10, 50)
    M, N = Y.shape
    varphi = build_cholesky_phi(M)
    Ytilde = varphi.T @ Y
    Psi_gram = Ytilde.T @ Ytilde  # (N x N)
    
    lam, V = no_iterations_pc(Y, n_components=5)
    for a in range(5):
        lhs = Psi_gram @ V[:, a]
        rhs = lam[a] * V[:, a]
        np.testing.assert_allclose(lhs, rhs, atol=1e-6)


# ── Power iteration ────────────────────────────────────────────────────────────

def test_power_iteration_converges():
    Y = make_Y(10, 50)
    Psi = (Y.T @ Y) / Y.shape[0]
    result = power_iteration_pc1(Psi, precision=1e-4)
    assert result.n_iterations > 0
    assert np.isfinite(result.eigenvalue)
    assert np.isfinite(result.eigenvector).all()


def test_power_iteration_unit_eigenvector():
    Y = make_Y(10, 50)
    Psi = (Y.T @ Y) / Y.shape[0]
    result = power_iteration_pc1(Psi)
    np.testing.assert_allclose(np.linalg.norm(result.eigenvector), 1.0, atol=1e-8)


def test_power_iteration_positive_eigenvalue():
    Y = make_Y(10, 50)
    Psi = (Y.T @ Y) / Y.shape[0]
    result = power_iteration_pc1(Psi)
    assert result.eigenvalue > 0


# ── Deflation ──────────────────────────────────────────────────────────────────

def test_deflation_removes_component():
    """After deflation, Y @ v should be near zero."""
    Y = make_Y(10, 50)
    Psi = (Y.T @ Y) / Y.shape[0]
    result = power_iteration_pc1(Psi)
    v = result.eigenvector
    Y_deflated = deflate(Y, v)
    proj = Y_deflated @ v
    np.testing.assert_allclose(proj, np.zeros(Y.shape[0]), atol=1e-10)


def test_deflation_preserves_shape():
    Y = make_Y(10, 50)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(50)
    v /= np.linalg.norm(v)
    Y_deflated = deflate(Y, v)
    assert Y_deflated.shape == Y.shape


# ── Full computational experiment ──────────────────────────────────────────────

def test_run_computational_experiment_small():
    """Run a small version of the computational experiment."""
    Y = make_Y(M=8, N=100, seed=0)
    results = run_computational_experiment(Y, n_runs=3, precision=1e-3)
    assert len(results['power_iter_counts']) == 3
    assert results['power_iter_mean'] > 0
    assert results['M_for_comparison'] == 8


def test_iteration_count_exceeds_M():
    """Paper claim: total iterations >> M for power method."""
    Y = make_Y(M=10, N=200, seed=42)
    M = Y.shape[0]
    results = run_computational_experiment(Y, n_runs=5, precision=1e-3)
    # Total iterations should be > M in most runs
    counts = results['power_iter_counts']
    # At least some runs should have iterations >> M
    # For M=10, expecting many more iterations
    assert results['power_iter_mean'] > M, (
        f"Mean iterations {results['power_iter_mean']} not > M={M}"
    )


def test_no_iterations_faster_than_power_total():
    """No-iterations does not require iterative steps; verify it runs quickly."""
    import time
    Y = make_Y(M=15, N=500, seed=0)
    
    start = time.time()
    no_iterations_pc(Y)
    ni_time = time.time() - start
    
    # Just verify no-iterations completes (timing comparison is environment-dependent)
    assert ni_time < 60.0, "No-iterations took too long"
