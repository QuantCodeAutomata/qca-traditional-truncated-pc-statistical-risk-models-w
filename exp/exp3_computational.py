"""
Experiment 3: No-iterations PC algorithm vs power iterations for M << N.

Reproduces Table 4 from the paper: N=2339, M=19, M+1=20 observations, 10 runs,
convergence precision 1e-3.

Paper reference: Kakushadze (2015), Table 4.
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

from src.computational import (
    no_iterations_pc,
    power_iterations_all_pcs,
    run_computational_experiment,
    build_cholesky_phi,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Paper Table 4 configuration
N_TABLE4 = 2339
M_TABLE4 = 19  # M (not M+1)
N_RUNS = 10
PRECISION = 1e-3


def generate_synthetic_returns(
    N: int,
    M: int,
    random_seed: int = 42,
    n_factors: int = 10,
) -> np.ndarray:
    """Generate realistic synthetic normalized demeaned returns for testing.
    
    Creates a factor model return matrix with N assets and M observations.
    Y has approximately the structure of equity cross-sections:
    - A few large eigenvalues (market + sector factors)
    - Many small eigenvalues
    
    Args:
        N: Number of assets (e.g., 2339)
        M: Number of time periods (e.g., 19)
        random_seed: Random seed
        n_factors: Number of synthetic common factors
    
    Returns:
        (M, N) normalized demeaned return matrix Y
    """
    rng = np.random.default_rng(random_seed)
    
    # Factor structure
    F = rng.standard_normal((M, n_factors)) * np.array([5, 3, 2, 1.5, 1.2, 1.1, 1.05, 1.02, 1.01, 1.0][:n_factors])
    B = rng.standard_normal((N, n_factors)) / np.sqrt(n_factors)  # factor loadings
    
    # Idiosyncratic returns
    eps = rng.standard_normal((M, N))
    
    # Combined returns (M x N)
    R = (F @ B.T) + eps
    
    # Demean each asset
    R = R - R.mean(axis=0, keepdims=True)
    
    # Normalize to unit variance (to get Y)
    std = R.std(axis=0)
    std = np.where(std > 1e-10, std, 1e-10)
    Y = R / std
    
    return Y


def validate_no_iterations(
    Y: np.ndarray,
    rtol: float = 1e-3,
) -> dict:
    """Validate no-iterations PCs match direct eigendecomposition.
    
    Computes both methods and compares eigenvalues and eigenvectors.
    
    Args:
        Y: (M, N) normalized demeaned returns
        rtol: Relative tolerance for eigenvalue comparison
    
    Returns:
        Dict with validation results
    """
    from scipy.linalg import eigh
    M, N = Y.shape
    
    # Direct eigendecomposition of Psi = (1/M) * Y.T @ Y (simplified for comparison)
    # Using the Gram-based Psi for accuracy
    varphi = build_cholesky_phi(M)
    Ytilde = varphi.T @ Y  # (M x N)
    Psi_direct = Ytilde.T @ Ytilde  # (N x N) correlation matrix via Gram path
    
    # Direct eigh on M x M Gram
    G = Ytilde @ Ytilde.T  # (M x M)
    rhos_gram, U_gram = eigh(G)
    idx = np.argsort(rhos_gram)[::-1]
    rhos_gram = rhos_gram[idx]
    U_gram = U_gram[:, idx]
    pos = rhos_gram > 1e-10
    rhos_gram = rhos_gram[pos]
    U_gram = U_gram[:, pos]
    
    # No-iterations method
    ni_lam, ni_V = no_iterations_pc(Y)
    
    n_check = min(5, len(ni_lam), len(rhos_gram))
    
    # Compare eigenvalues
    lam_match = np.allclose(ni_lam[:n_check], rhos_gram[:n_check], rtol=rtol)
    
    # Compare eigenvectors (up to sign)
    vec_residuals = []
    for a in range(n_check):
        v_ni = ni_V[:, a]
        v_direct = Ytilde.T @ U_gram[:, a] / np.sqrt(rhos_gram[a])
        v_direct /= np.linalg.norm(v_direct)
        # Residual (up to sign)
        res = min(
            np.linalg.norm(v_ni - v_direct),
            np.linalg.norm(v_ni + v_direct)
        )
        vec_residuals.append(res)
    
    return {
        'ni_eigenvalues': ni_lam[:n_check],
        'gram_eigenvalues': rhos_gram[:n_check],
        'eigenvalue_match': lam_match,
        'eigenvector_residuals': vec_residuals,
        'mean_vec_residual': float(np.mean(vec_residuals)) if vec_residuals else float('nan'),
    }


def run_experiment_3(
    N: int = N_TABLE4,
    M: int = M_TABLE4,
    n_runs: int = N_RUNS,
    precision: float = PRECISION,
    random_seed: int = 42,
) -> dict:
    """Run Experiment 3: computational comparison.
    
    Args:
        N: Number of assets (paper: 2339)
        M: Number of observations (paper: 19)
        n_runs: Number of power-iteration runs (paper: 10)
        precision: Convergence precision (paper: 1e-3)
        random_seed: Random seed
    
    Returns:
        Dict with experiment results
    """
    logger.info(f"Exp3: N={N}, M={M}, {n_runs} runs, precision={precision}")
    
    # Generate synthetic returns matching paper dimensions
    Y = generate_synthetic_returns(N, M, random_seed=random_seed)
    logger.info(f"Exp3: Generated Y with shape {Y.shape}")
    
    # Validate no-iterations algorithm
    logger.info("Exp3: Validating no-iterations algorithm")
    validation = validate_no_iterations(Y)
    logger.info(f"Exp3: Eigenvalue match: {validation['eigenvalue_match']}")
    logger.info(f"Exp3: Mean eigenvector residual: {validation['mean_vec_residual']:.6f}")
    
    # Run full computational experiment
    logger.info("Exp3: Running full computational experiment")
    exp_results = run_computational_experiment(
        Y, n_runs=n_runs, precision=precision, random_seed=random_seed
    )
    
    mean_iter = exp_results['power_iter_mean']
    median_iter = exp_results['power_iter_median']
    
    logger.info(f"Exp3: Power iterations - mean={mean_iter:.1f}, median={median_iter:.1f}")
    logger.info(f"Exp3: M={M} for comparison (should be << total iterations)")
    logger.info(f"Exp3: Paper expects ~5037 total iterations (vs M=19)")
    
    # Save results
    results_df = pd.DataFrame({
        'run': range(1, n_runs + 1),
        'total_iterations': exp_results['power_iter_counts'],
    })
    results_df.to_csv(RESULTS_DIR / 'exp3_power_iter_counts.csv', index=False)
    
    # Summary
    summary = {
        'N': N,
        'M': M,
        'n_runs': n_runs,
        'precision': precision,
        'power_iter_mean': mean_iter,
        'power_iter_median': median_iter,
        'power_iter_std': exp_results['power_iter_std'],
        'power_iter_min': float(min(exp_results['power_iter_counts'])),
        'power_iter_max': float(max(exp_results['power_iter_counts'])),
        'M_comparison': M,
        'ratio_iters_to_M': mean_iter / M,
        'eigenvalue_match': validation['eigenvalue_match'],
        'mean_eigvec_residual': validation['mean_vec_residual'],
        'ni_eigenvalues_top5': validation['ni_eigenvalues'].tolist(),
    }
    
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / 'exp3_summary.csv', index=False)
    
    # Plot iteration count distribution
    _plot_iteration_histogram(exp_results['power_iter_counts'], M, RESULTS_DIR)
    
    return {**summary, 'results_df': results_df, 'validation': validation}


def _plot_iteration_histogram(counts: list, M: int, results_dir: Path) -> None:
    """Plot histogram of power-iteration counts across runs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(counts) + 1), counts, color='steelblue', alpha=0.8)
    ax.axhline(y=M, color='red', linestyle='--', linewidth=2, label=f'M={M}')
    ax.axhline(y=np.mean(counts), color='orange', linestyle='-',
               linewidth=2, label=f'Mean={np.mean(counts):.0f}')
    ax.set_title(f'Exp3: Total Power Iterations per Run (N={N_TABLE4}, M={M})', fontsize=13)
    ax.set_xlabel('Run')
    ax.set_ylabel('Total Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(results_dir / 'exp3_iteration_histogram.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    results = run_experiment_3()
    print('\n' + '='*60)
    print('Experiment 3 – Computational Comparison (Table 4)')
    print('='*60)
    print(f"N={results['N']}, M={results['M']}, {results['n_runs']} runs")
    print(f"Power iterations: mean={results['power_iter_mean']:.1f}, "
          f"median={results['power_iter_median']:.1f}")
    print(f"M for reference: {results['M']} (ratio: {results['ratio_iters_to_M']:.0f}x)")
    print(f"Paper expects ~5037 total iterations (vs M=19)")
    print(f"No-iterations eigenvalue match: {results['eigenvalue_match']}")
