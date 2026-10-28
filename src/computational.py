import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class PowerIterResult:
    """Results from power iteration extraction."""
    eigenvalue: float
    eigenvector: np.ndarray
    n_iterations: int
    residual: float


def build_cholesky_phi(
    M: int,
) -> np.ndarray:
    """Build the lower-triangular Cholesky factor varphi of phi.
    
    phi_ss' = (delta_ss' + u_s u_s') / M  where u_s = 1
    phi = varphi @ varphi.T (Cholesky decomposition)
    
    Paper formulas (1-indexed):
        varphi_ss = sqrt((s+1) / (s*M))  for s=1..M
        varphi_s>s' = 1 / sqrt(s'*(s'+1)*M)  for s > s'
        varphi_s<s' = 0
    
    Args:
        M: Number of observations
    
    Returns:
        (M, M) lower-triangular Cholesky factor (0-indexed)
    """
    varphi = np.zeros((M, M))
    for s in range(M):  # 0-indexed
        s1 = s + 1  # 1-indexed
        varphi[s, s] = np.sqrt((s1 + 1.0) / (s1 * M))
        for sp in range(s):
            sp1 = sp + 1
            varphi[s, sp] = 1.0 / np.sqrt(sp1 * (sp1 + 1.0) * M)
    return varphi


def no_iterations_pc(
    Y: np.ndarray,
    n_components: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """No-iterations PC algorithm via M x M Gram matrix.
    
    For Y of shape (M, N) with M << N:
    1. Build Cholesky factor varphi of phi
    2. Form Ytilde = varphi.T @ Y  (M x N)
    3. Gram matrix G = Ytilde @ Ytilde.T  (M x M)
    4. Eigen-decompose G: G U^(a) = rho^(a) U^(a)
    5. Recover V_i^(a) = (1/sqrt(rho^(a))) sum_s Ytilde_is U_s^(a)
    
    Args:
        Y: (M, N) normalized demeaned returns
        n_components: Number of components to return (default: M)
    
    Returns:
        (eigenvalues, eigenvectors) sorted descending
        eigenvalues: (n_components,) or (M,)
        eigenvectors: (N, n_components) or (N, M)
    """
    M, N = Y.shape
    if n_components is None:
        n_components = M
    
    varphi = build_cholesky_phi(M)
    Ytilde = varphi.T @ Y  # (M x N)
    G = Ytilde @ Ytilde.T  # (M x M)
    
    from scipy.linalg import eigh
    rhos, U = eigh(G)  # ascending
    
    # Sort descending
    idx = np.argsort(rhos)[::-1]
    rhos = rhos[idx]
    U = U[:, idx]
    
    # Keep positive eigenvalues
    pos = rhos > 1e-10
    rhos_pos = rhos[pos]
    U_pos = U[:, pos]
    
    n_keep = min(n_components, len(rhos_pos))
    rhos_pos = rhos_pos[:n_keep]
    U_pos = U_pos[:, :n_keep]
    
    # Recover N-dimensional eigenvectors
    V = Ytilde.T @ U_pos  # (N x n_keep)
    for a in range(n_keep):
        norm = np.sqrt(rhos_pos[a])
        if norm > 1e-12:
            V[:, a] /= norm
    
    return rhos_pos, V


def power_iteration_pc1(
    Psi: np.ndarray,
    init_vector: Optional[np.ndarray] = None,
    precision: float = 1e-3,
    max_iter: int = 100000,
) -> PowerIterResult:
    """Extract first principal component of Psi via power iteration.
    
    For first PC: initialize with uniform vector u/sqrt(N), then iterate
    V_{r+1} = Psi @ V_r / ||Psi @ V_r|| until convergence.
    
    Convergence criterion: ||V_{r+1} - V_r|| < precision (up to sign).
    
    Args:
        Psi: (N, N) correlation matrix
        init_vector: Initial vector. If None, uses uniform vector.
        precision: Convergence tolerance (1e-3 per paper)
        max_iter: Maximum iterations
    
    Returns:
        PowerIterResult with eigenvalue, eigenvector, and iteration count
    """
    N = Psi.shape[0]
    
    if init_vector is None:
        v = np.ones(N) / np.sqrt(N)  # uniform vector
    else:
        v = init_vector / np.linalg.norm(init_vector)
    
    for i in range(max_iter):
        v_new = Psi @ v
        lam = np.linalg.norm(v_new)
        if lam < 1e-12:
            break
        v_new /= lam
        
        # Check convergence: ||v_new - v|| or ||v_new + v|| (sign ambiguity)
        diff = min(
            np.linalg.norm(v_new - v),
            np.linalg.norm(v_new + v)
        )
        v = v_new
        
        if diff < precision:
            residual = diff
            eigenvalue = float(v.T @ Psi @ v)
            return PowerIterResult(eigenvalue, v, i + 1, residual)
    
    eigenvalue = float(v.T @ Psi @ v)
    return PowerIterResult(eigenvalue, v, max_iter, float('inf'))


def deflate(
    Y: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Deflate return matrix by removing a PC component.
    
    Y_is^(a+1) = Y_is^(a) - V_i^(a) * sum_j V_j^(a) Y_js^(a)
    
    In matrix form: Y_new = Y - (Y @ v) [:,newaxis] @ v[newaxis,:]
    Wait: Y is (M x N), v is (N,)
    Y_is^new = Y_is - V_i * sum_j V_j Y_js
    = Y_is - V_i * (Y @ v)_s  [where (Y@v)_s = sum_j V_j Y_js]
    
    Actually: Y_is^new = Y_is - V_i^(a) * (sum_j V_j^(a) * Y_js^(a))
    Y_new[:,i] = Y[:,i] - v[i] * (Y @ v)  [shape (M,)]
    So: Y_new = Y - (Y @ v)[:, np.newaxis] * v[np.newaxis, :]
    
    Args:
        Y: (M, N) return matrix
        v: (N,) eigenvector to deflate
    
    Returns:
        (M, N) deflated return matrix
    """
    proj = Y @ v  # (M,)
    return Y - np.outer(proj, v)


def power_iterations_all_pcs(
    Y: np.ndarray,
    n_components: int,
    precision: float = 1e-3,
    max_iter: int = 100000,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract multiple PCs via power iterations with deflation.
    
    First PC: initialize with uniform vector.
    Higher PCs: initialize with random N-vector of mean 0, unit variance, normalized.
    Uses successive deflation of the return matrix.
    
    Args:
        Y: (M, N) normalized demeaned return matrix
        n_components: Number of PCs to extract
        precision: Convergence tolerance (1e-3 per paper)
        max_iter: Maximum iterations per component
        random_seed: Random seed for reproducibility
    
    Returns:
        (eigenvalues, eigenvectors, total_iterations)
        eigenvalues: (n_components,) array
        eigenvectors: (N, n_components) array
        total_iterations: Total iteration count across all components
    """
    M, N = Y.shape
    rng = np.random.default_rng(random_seed)
    
    eigenvalues = np.zeros(n_components)
    eigenvectors = np.zeros((N, n_components))
    total_iters = 0
    
    Y_current = Y.copy()
    
    for a in range(n_components):
        # Compute Psi^(a) = (1/M) sum_s Y_is^(a) Y_js^(a)
        Psi_a = (Y_current.T @ Y_current) / M
        
        # Initialize vector
        if a == 0:
            init_v = np.ones(N) / np.sqrt(N)  # uniform
        else:
            raw = rng.standard_normal(N)
            raw = raw - raw.mean()  # zero mean
            raw /= np.linalg.norm(raw)  # unit norm
            init_v = raw
        
        result = power_iteration_pc1(Psi_a, init_v, precision, max_iter)
        eigenvalues[a] = result.eigenvalue
        eigenvectors[:, a] = result.eigenvector
        total_iters += result.n_iterations
        
        # Deflate Y
        Y_current = deflate(Y_current, result.eigenvector)
    
    return eigenvalues, eigenvectors, total_iters


def run_computational_experiment(
    Y: np.ndarray,
    n_runs: int = 10,
    precision: float = 1e-3,
    n_components: Optional[int] = None,
    random_seed: int = 42,
) -> dict:
    """Run Table 4 computational experiment: no-iterations vs power iterations.
    
    Args:
        Y: (M, N) normalized demeaned return matrix (M<<N setup)
        n_runs: Number of independent runs for power iterations
        precision: Convergence precision
        n_components: Number of PCs to extract (default: M)
        random_seed: Base random seed
    
    Returns:
        Dict with results from both methods
    """
    M, N = Y.shape
    if n_components is None:
        n_components = M
    
    # No-iterations method
    ni_eigenvalues, ni_eigenvectors = no_iterations_pc(Y, n_components)
    
    # Validate: check Psi @ V = lambda * V
    Psi_direct = (Y.T @ Y) / M  # simplified correlation (no demeaning by u)
    residuals_ni = []
    for a in range(min(5, len(ni_eigenvalues))):
        v = ni_eigenvectors[:, a]
        lam = ni_eigenvalues[a]
        res = np.linalg.norm(Psi_direct @ v - lam * v)
        residuals_ni.append(res)
    
    # Power iterations method (multiple runs)
    power_iter_counts = []
    for run in range(n_runs):
        _, _, total_iters = power_iterations_all_pcs(
            Y, n_components, precision=precision,
            random_seed=random_seed + run
        )
        power_iter_counts.append(total_iters)
    
    return {
        'N': N,
        'M': M,
        'n_components': n_components,
        'ni_eigenvalues': ni_eigenvalues,
        'ni_residuals': residuals_ni,
        'power_iter_counts': power_iter_counts,
        'power_iter_mean': float(np.mean(power_iter_counts)),
        'power_iter_median': float(np.median(power_iter_counts)),
        'power_iter_std': float(np.std(power_iter_counts)),
        'M_for_comparison': M,
    }
