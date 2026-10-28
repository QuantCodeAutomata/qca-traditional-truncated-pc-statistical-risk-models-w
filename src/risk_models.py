import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class RiskModelResult:
    """Container for a computed statistical risk model."""
    # K-selection variant name
    variant: str
    # Selected number of factors
    K: int
    # Universe tickers
    tickers: list
    # Factor loadings matrix (N x K) in normalized demeaned return space
    V: np.ndarray  # shape (N, K)
    # Retained eigenvalues (K,)
    lambdas: np.ndarray  # shape (K,)
    # Specific variances in correlation space (N,)
    xi2: np.ndarray  # shape (N,)
    # Sample standard deviations (N,)
    sigmas: np.ndarray  # shape (N,)
    # Model covariance matrix Gammahat (N x N)
    Gammahat: Optional[np.ndarray] = None
    # Model covariance inverse Gammahat_inv (N x N)
    Gammahat_inv: Optional[np.ndarray] = None
    # eRank of Psi
    erank: Optional[float] = None
    # All eigenvalues of Psi (descending)
    all_eigenvalues: Optional[np.ndarray] = None


def compute_sample_covariance(
    returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sample covariance, correlation, demeaned returns, and sigmas.
    
    Paper definitions:
        X_is = R_is - mean(R_i)  [demeaned returns]
        C_ij = (1/M) sum_s X_is X_js  [sample covariance]
        sigma_i = sqrt(C_ii)  [sample volatilities]
        Y_is = X_is / sigma_i  [normalized demeaned returns]
        Psi_ij = C_ij / (sigma_i * sigma_j)  [sample correlation]
    
    Args:
        returns: Array of shape (M, N) where M = number of observations,
                 N = number of assets
    
    Returns:
        Tuple (C, Psi, sigmas, Y) where:
            C: (N, N) sample covariance
            Psi: (N, N) sample correlation
            sigmas: (N,) sample standard deviations
            Y: (M, N) normalized demeaned returns (Y_is = X_is / sigma_i)
    """
    M, N = returns.shape
    # Demean each asset
    X = returns - returns.mean(axis=0, keepdims=True)  # (M, N)
    # Sample covariance (1/M normalization)
    C = (X.T @ X) / M  # (N, N)
    # Sample volatilities
    sigmas = np.sqrt(np.diag(C))  # (N,)
    # Handle near-zero volatilities
    sigmas = np.where(sigmas > 1e-10, sigmas, 1e-10)
    # Sample correlation
    outer_sigma = np.outer(sigmas, sigmas)
    Psi = C / outer_sigma  # (N, N)
    # Clip for numerical stability
    np.fill_diagonal(Psi, 1.0)
    # Normalized demeaned returns
    Y = X / sigmas[np.newaxis, :]  # (M, N)
    return C, Psi, sigmas, Y


def eigen_decompose_correlation(
    Psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose sample correlation matrix in descending eigenvalue order.
    
    Uses scipy.linalg.eigh for symmetric matrices.
    
    Args:
        Psi: (N, N) symmetric positive semidefinite correlation matrix
    
    Returns:
        (eigenvalues, eigenvectors) where eigenvalues are sorted descending.
        eigenvalues: (N,) array
        eigenvectors: (N, N) array, column a is V^(a)
    """
    from scipy.linalg import eigh
    eigenvalues, eigenvectors = eigh(Psi)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def eigen_decompose_gram(
    Y: np.ndarray,
    phi: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Efficient no-iterations PC algorithm using M x M Gram matrix.
    
    When M << N, diagonalizing the M x M Gram matrix is O(M^3 + M^2*N)
    vs O(N^3) for direct eigendecomposition.
    
    Paper: Psi_ij = sum_{s,s'} Y_is phi_ss' Y_js'
    with phi_ss' = (delta_ss' + u_s u_s') / M, u_s = 1.
    
    Using Cholesky: phi = varphi varphi^T where
        varphi_ss = sqrt((s+1)/(s*M))  (1-indexed, s=1..M)
        varphi_s>s' = 1 / sqrt(s'*(s'+1)*M)  for s > s'
        varphi_s<s' = 0
    
    Then Psi_ij = sum_{s=1}^{M} Ytilde_is Ytilde_js
    where Ytilde = Y @ varphi  (using 0-indexed arrays)
    
    Args:
        Y: (M, N) normalized demeaned returns (0-indexed, rows are time)
        phi: Optional precomputed (M, M) phi matrix; if None, computed from
             the Cholesky formula
    
    Returns:
        (eigenvalues, eigenvectors) of Psi, sorted descending.
        eigenvalues: (M,) positive eigenvalues
        eigenvectors: (N, M) array, column a is the a-th PC of Psi
    """
    from scipy.linalg import eigh, cholesky
    M, N = Y.shape
    
    # Build Cholesky factor varphi (lower triangular, 0-indexed)
    varphi = np.zeros((M, M))
    for s in range(M):  # 0-indexed s = 0..M-1 corresponds to paper's 1..M
        s1 = s + 1  # 1-indexed
        # Diagonal: varphi_ss = sqrt((s+1)/(s*M)) but for s=1: sqrt(2/M)
        if s1 == 1:
            varphi[s, s] = np.sqrt(2.0 / M)  # s'=s=1: sqrt((1+1)/(1*M))
        else:
            varphi[s, s] = np.sqrt((s1 + 1.0) / (s1 * M))
        # Below diagonal: varphi_{s,s'} = 1/sqrt(s'*(s'+1)*M) for s > s'
        for sp in range(s):  # sp is 0-indexed s'
            sp1 = sp + 1  # 1-indexed s'
            varphi[s, sp] = 1.0 / np.sqrt(sp1 * (sp1 + 1.0) * M)
    
    # Ytilde = Y @ varphi  (M x N).T @ (M x M) -> (N x M)
    # Actually Ytilde_is = sum_{s'} Y_{s'i} varphi_{s s'} ... let me re-check
    # Paper: Ytilde = Y varphi where Y is (M x N), so Ytilde is (M x N)
    # but varphi is (M x M) lower triangular
    # Psi_ij = sum_s Ytilde_is Ytilde_js, so Ytilde is (M x N): rows=time, cols=assets
    Ytilde = varphi.T @ Y  # (M x M) @ (M x N) = (M x N); varphi.T is upper triangular
    # Wait: Ytilde_is = sum_{s'} varphi_{s's} Y_{s'i} ... 
    # Let's just use: Ytilde = varphi.T @ Y to get (M x N) Ytilde
    # Then G_ss' = sum_i Ytilde_is Ytilde_is' = (Ytilde @ Ytilde.T)[s,s']
    # Psi_ij = (Ytilde.T @ Ytilde)[i,j] = sum_s Ytilde_si Ytilde_sj
    
    # Gram matrix G (M x M)
    G = Ytilde @ Ytilde.T  # (M x M)
    
    # Eigendecompose G
    rhos, U = eigh(G)  # rhos ascending, U[:,a] is eigenvector
    # Sort descending, keep positive
    idx = np.argsort(rhos)[::-1]
    rhos = rhos[idx]
    U = U[:, idx]
    
    # Keep only positive eigenvalues
    pos_mask = rhos > 1e-10
    rhos = rhos[pos_mask]
    U = U[:, pos_mask]
    
    # Recover N-dimensional eigenvectors: V_i^(a) = (1/sqrt(rho^a)) sum_s Ytilde_si U_s^(a)
    # V = (1/sqrt(rho)) * Ytilde.T @ U  -> shape (N x K)
    V = Ytilde.T @ U  # (N x K)
    for a in range(len(rhos)):
        norm = np.sqrt(rhos[a])
        if norm > 1e-12:
            V[:, a] /= norm
    
    return rhos, V


def compute_erank(
    eigenvalues: np.ndarray,
) -> float:
    """Compute effective rank (eRank) of a matrix from its eigenvalues.
    
    eRank(Psi) = exp(H) where H = -sum_a p_a ln(p_a)
    p_a = lambda^(a) / sum_b lambda^(b)  (using positive eigenvalues only)
    
    Args:
        eigenvalues: Array of eigenvalues (all or just positive ones)
    
    Returns:
        Effective rank as a float
    """
    pos = eigenvalues[eigenvalues > 0]
    if len(pos) == 0:
        return 1.0
    p = pos / pos.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def compute_xi2(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    K: int,
) -> np.ndarray:
    """Compute specific variances xi_i^2 = 1 - sum_{A=1}^K lambda^(A) (V_i^(A))^2.
    
    Args:
        eigenvalues: (N,) or (K,) eigenvalues in descending order
        eigenvectors: (N, N) or (N, K) eigenvectors, column a is V^(a)
        K: Number of factors to retain
    
    Returns:
        (N,) array of specific variances xi_i^2
    """
    N = eigenvectors.shape[0]
    xi2 = np.ones(N)
    for a in range(min(K, eigenvectors.shape[1])):
        xi2 -= eigenvalues[a] * (eigenvectors[:, a] ** 2)
    return xi2


def build_traditional_model(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    sigmas: np.ndarray,
    K: int,
    use_woodbury: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the traditional truncated-PC risk model.
    
    Correlation space:
        Gamma_ij = xi_i^2 delta_ij + sum_{A=1}^K lambda^(A) V_i^(A) V_j^(A)
    where xi_i^2 = 1 - sum_{A=1}^K lambda^(A) (V_i^(A))^2
    
    Covariance:
        Gammahat_ij = sigma_i * sigma_j * Gamma_ij
    
    Covariance inverse via Woodbury:
        (D + VLV^T)^{-1} = D^{-1} - D^{-1} V (L^{-1} + V^T D^{-1} V)^{-1} V^T D^{-1}
    where D = diag(xi_i^2 * sigma_i^2) and L = diag(lambda^(A))
    
    Args:
        eigenvalues: (N,) eigenvalues of Psi, descending
        eigenvectors: (N, N) eigenvectors of Psi
        sigmas: (N,) sample volatilities
        K: Number of factors to retain
        use_woodbury: If True, use Woodbury identity for inverse
    
    Returns:
        (xi2, Gammahat, Gammahat_inv) where:
            xi2: (N,) specific variances in correlation space
            Gammahat: (N, N) model covariance
            Gammahat_inv: (N, N) inverse model covariance
    """
    N = eigenvectors.shape[0]
    K = min(K, eigenvectors.shape[1], N - 1)
    K = max(K, 1)
    
    # Compute specific variances in correlation space
    xi2 = compute_xi2(eigenvalues, eigenvectors, K)
    
    # Clip xi2 to be positive (numerical safety)
    xi2 = np.maximum(xi2, 1e-8)
    
    # Factor loadings in correlation space: (N, K)
    V_K = eigenvectors[:, :K]  # (N, K)
    lam_K = eigenvalues[:K]    # (K,)
    
    # Factor loadings in covariance units: Vtilde_i^(A) = sigma_i * V_i^(A)
    # D_cov = diag(xi_i^2 * sigma_i^2) = diag(xi_i^2) * diag(sigma_i^2)
    xi2_cov = xi2 * (sigmas ** 2)  # specific variances in covariance units
    
    # Build model covariance via Woodbury identity
    # Gammahat = D + F F^T where F_i^(A) = sigma_i * sqrt(lambda^(A)) * V_i^(A)
    F = sigmas[:, np.newaxis] * V_K * np.sqrt(lam_K)[np.newaxis, :]  # (N, K)
    D_vec = xi2_cov  # (N,)
    D_inv = 1.0 / D_vec  # (N,)
    
    # Build Gammahat explicitly (for small N this is fine)
    Gammahat = np.diag(D_vec) + F @ F.T  # (N, N)
    
    if use_woodbury:
        # Woodbury: (D + FF^T)^{-1} = D^{-1} - D^{-1} F (I + F^T D^{-1} F)^{-1} F^T D^{-1}
        D_inv_vec = D_inv  # (N,)
        FtDinvF = F.T * D_inv_vec[np.newaxis, :] @ F  # Wait: (K x N) @ (N x K) = (K x K)
        FtDinvF = (F * D_inv_vec[:, np.newaxis]).T @ F  # (K, K)
        M_mat = np.eye(K) + FtDinvF  # (K, K)
        from scipy.linalg import solve, inv
        try:
            M_inv = inv(M_mat)  # (K, K)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M_mat)
        
        # DinvF = D^{-1} F  (N, K)
        DinvF = F * D_inv_vec[:, np.newaxis]
        # Gammahat_inv = diag(D_inv) - DinvF @ M_inv @ DinvF^T
        Gammahat_inv = np.diag(D_inv_vec) - DinvF @ M_inv @ DinvF.T
    else:
        from scipy.linalg import inv
        try:
            Gammahat_inv = inv(Gammahat)
        except np.linalg.LinAlgError:
            Gammahat_inv = np.linalg.pinv(Gammahat)
    
    return xi2, Gammahat, Gammahat_inv


def minimization_k_selection(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    max_K: Optional[int] = None,
    exclude_first: bool = False,
) -> int:
    """Select K using the minimization algorithm (Appendix A of the paper).
    
    Objective: g(K) = sqrt(min_i xi_i^2) + sqrt(max_i xi_i^2)
    Select K = argmin_K |g(K) - 1|, using upward-iteration stopping rule:
    iterate K upward, stop at first K where objective WORSENS.
    
    If exclude_first=True, start from K=2 (excluding first PC).
    
    Args:
        eigenvalues: (N,) eigenvalues of Psi, descending order
        eigenvectors: (N, N) eigenvectors of Psi
        max_K: Maximum K to consider (defaults to N-1)
        exclude_first: If True, start from K=2 and report K (already includes PC1 conceptually)
    
    Returns:
        Selected K value
    """
    N = eigenvectors.shape[0]
    if max_K is None:
        max_K = N - 1
    max_K = min(max_K, eigenvectors.shape[1])
    
    start_K = 2 if exclude_first else 1
    
    best_K = start_K
    best_obj = np.inf
    
    for K in range(start_K, max_K + 1):
        xi2 = compute_xi2(eigenvalues, eigenvectors, K)
        xi2_pos = xi2[xi2 > 0]
        if len(xi2_pos) == 0:
            break
        g_K = np.sqrt(xi2_pos.min()) + np.sqrt(xi2_pos.max())
        obj = abs(g_K - 1.0)
        
        if K == start_K:
            best_K = K
            best_obj = obj
        else:
            if obj < best_obj:
                best_K = K
                best_obj = obj
            else:
                # Objective worsens: stop and use previous K
                break
    
    return best_K


def erank_k_selection(
    eigenvalues: np.ndarray,
    psi_matrix: Optional[np.ndarray] = None,
    method: str = 'round',
    max_K: Optional[int] = None,
    exclude_first_psi: bool = False,
    eigenvectors: Optional[np.ndarray] = None,
) -> tuple[int, float]:
    """Select K using effective rank (eRank) method.
    
    K = Round(eRank(Psi)) or Floor(eRank(Psi)), capped at max_K.
    
    If exclude_first_psi=True:
        Compute Psi'_ij = sum_{a=2}^{M} V_i^(a) lambda^(a) V_j^(a)
        (excludes the first PC from the reconstruction)
        Then K' = eRank_method(Psi'), K = K' + 1
    
    Args:
        eigenvalues: (N,) eigenvalues of Psi in descending order
        psi_matrix: (N, N) sample correlation matrix (for exclude_first variant)
        method: 'round' or 'floor'
        max_K: Cap on K
        exclude_first_psi: If True, use Psi' = Psi - lambda1 V1 V1^T
        eigenvectors: (N, N) eigenvectors, required if exclude_first_psi=True
    
    Returns:
        (K, erank_value)
    """
    N = len(eigenvalues)
    if max_K is None:
        max_K = N - 1
    
    if exclude_first_psi and eigenvectors is not None:
        # Compute eigenvalues of Psi' = Psi - lambda1 V1 V1^T
        # Eigenvalues of Psi' are {0, lambda2, lambda3, ..., lambdaN}
        # (since Psi' = sum_{a>=2} lambda_a V_a V_a^T)
        psi_prime_eigenvalues = eigenvalues[1:]  # skip first eigenvalue
        er = compute_erank(psi_prime_eigenvalues)
        if method == 'round':
            Kprime = int(np.round(er))
        else:  # floor
            Kprime = int(np.floor(er))
        K = Kprime + 1
    else:
        er = compute_erank(eigenvalues)
        if method == 'round':
            K = int(np.round(er))
        else:  # floor
            K = int(np.floor(er))
    
    K = max(1, min(K, max_K))
    return K, er


def select_K(
    variant: str,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    M: int,
) -> tuple[int, Optional[float]]:
    """Select K for a given model variant.
    
    Variants:
        'M1': minimization
        'M2': minimization, exclude-first PC
        'M3': Round(eRank(Psi))
        'M4': floor(eRank(Psi))
        'M5': Round(eRank(Psi')) + 1
        'M6': floor(eRank(Psi')) + 1
    
    Args:
        variant: One of 'M1'..'M6'
        eigenvalues: (N,) eigenvalues of Psi, descending
        eigenvectors: (N, N) eigenvectors of Psi
        M: Number of return observations (K capped at M-1)
    
    Returns:
        (K, erank_value_or_None)
    """
    max_K = M - 1
    
    if variant == 'M1':
        K = minimization_k_selection(eigenvalues, eigenvectors, max_K=max_K)
        return K, None
    
    elif variant == 'M2':
        K = minimization_k_selection(eigenvalues, eigenvectors, max_K=max_K, exclude_first=True)
        return K, None
    
    elif variant == 'M3':
        K, er = erank_k_selection(eigenvalues, method='round', max_K=max_K)
        return K, er
    
    elif variant == 'M4':
        K, er = erank_k_selection(eigenvalues, method='floor', max_K=max_K)
        return K, er
    
    elif variant == 'M5':
        K, er = erank_k_selection(
            eigenvalues, method='round', max_K=max_K,
            exclude_first_psi=True, eigenvectors=eigenvectors
        )
        return K, er
    
    elif variant == 'M6':
        K, er = erank_k_selection(
            eigenvalues, method='floor', max_K=max_K,
            exclude_first_psi=True, eigenvectors=eigenvectors
        )
        return K, er
    
    else:
        raise ValueError(f'Unknown variant: {variant}')


def build_risk_model(
    returns: np.ndarray,
    tickers: list,
    variant: str = 'M1',
    K_override: Optional[int] = None,
    use_gram: bool = False,
    use_woodbury: bool = True,
    compute_inverse: bool = True,
) -> RiskModelResult:
    """Build a complete traditional statistical risk model.
    
    Args:
        returns: (M, N) array of fully adjusted close-to-close log returns
        tickers: List of N ticker strings
        variant: K-selection variant ('M1'-'M6', or 'fixed' with K_override)
        K_override: Fixed K value (used when variant='fixed')
        use_gram: If True, use Gram matrix approach for eigendecomposition
        use_woodbury: If True, use Woodbury identity for covariance inverse
        compute_inverse: If True, compute model covariance inverse
    
    Returns:
        RiskModelResult with all components
    """
    M, N = returns.shape
    
    # Step 1: Sample statistics
    C, Psi, sigmas, Y = compute_sample_covariance(returns)
    
    # Step 2: Eigendecomposition
    if use_gram and M < N:
        lam, V = eigen_decompose_gram(Y)
        # Pad with zeros to match N-dim if needed
        K_max = len(lam)
        lam_full = np.zeros(N)
        lam_full[:K_max] = lam
        V_full = np.zeros((N, N))
        V_full[:, :min(K_max, N)] = V[:, :min(K_max, N)]
        eigenvalues, eigenvectors = lam_full, V_full
    else:
        eigenvalues, eigenvectors = eigen_decompose_correlation(Psi)
    
    # Step 3: K selection
    if K_override is not None:
        K = min(K_override, M - 1, N - 1)
        K = max(K, 1)
        erank_val = compute_erank(eigenvalues)
    elif variant == 'fixed':
        raise ValueError('K_override must be provided when variant=fixed')
    else:
        K, erank_val = select_K(variant, eigenvalues, eigenvectors, M)
    
    # Validate K
    if K < 1:
        K = 1
    if K >= M:
        K = M - 1
    if K >= N:
        K = N - 1
    
    # Step 4: Build model
    xi2, Gammahat, Gammahat_inv = build_traditional_model(
        eigenvalues, eigenvectors, sigmas, K,
        use_woodbury=use_woodbury and compute_inverse
    )
    
    if not compute_inverse:
        Gammahat_inv = None
    
    return RiskModelResult(
        variant=variant,
        K=K,
        tickers=tickers,
        V=eigenvectors[:, :K],
        lambdas=eigenvalues[:K],
        xi2=xi2,
        sigmas=sigmas,
        Gammahat=Gammahat,
        Gammahat_inv=Gammahat_inv,
        erank=erank_val,
        all_eigenvalues=eigenvalues,
    )


def build_shrinkage_model_diagonal(
    returns: np.ndarray,
    tickers: list,
    q: float = 0.3,
    use_woodbury: bool = True,
) -> RiskModelResult:
    """Build diagonal-target shrinkage risk model.
    
    The model deforms the factor representation by scaling by (1-q)
    and setting specific variances to xi_i^2 = q in correlation space.
    
    Model: phitilde_ss' = (1-q) * phi_ss'  (factor weights scaled by 1-q)
          xi_i^2 = q  (uniform specific variances)
    
    In covariance units: Gammahat_ij = sigma_i * sigma_j * Gamma_ij
    where Gamma_ij = q*delta_ij + (1-q) * Psi_ij
    
    Args:
        returns: (M, N) array of log returns
        tickers: List of N tickers
        q: Shrinkage parameter in (0, 1)
    
    Returns:
        RiskModelResult with the shrinkage model
    """
    M, N = returns.shape
    C, Psi, sigmas, Y = compute_sample_covariance(returns)
    
    # Gamma = q*I + (1-q)*Psi in correlation space
    Gamma = q * np.eye(N) + (1.0 - q) * Psi
    # Scale to covariance: Gammahat_ij = sigma_i * sigma_j * Gamma_ij
    outer_sigma = np.outer(sigmas, sigmas)
    Gammahat = Gamma * outer_sigma
    
    from scipy.linalg import inv
    try:
        Gammahat_inv = inv(Gammahat)
    except Exception:
        Gammahat_inv = np.linalg.pinv(Gammahat)
    
    xi2 = np.full(N, q)
    eigenvalues, eigenvectors = eigen_decompose_correlation(Psi)
    
    return RiskModelResult(
        variant=f'shrinkage_diag_q{q}',
        K=N,
        tickers=tickers,
        V=eigenvectors,
        lambdas=eigenvalues,
        xi2=xi2,
        sigmas=sigmas,
        Gammahat=Gammahat,
        Gammahat_inv=Gammahat_inv,
        all_eigenvalues=eigenvalues,
    )


def build_shrinkage_model_uniform_corr(
    returns: np.ndarray,
    tickers: list,
    q: float = 0.3,
    rho: float = 0.3,
    use_woodbury: bool = True,
) -> RiskModelResult:
    """Build uniform-correlation / intercept-factor shrinkage model.
    
    Model in correlation space:
        Gamma_ij = xi_i^2 * delta_ij + q*rho * nu_i * nu_j + (1-q) * Psi_ij
    where xi_i^2 = q*(1-rho), nu_i = 1 for all i, and |rho| < 1.
    
    Args:
        returns: (M, N) array of log returns
        tickers: List of N tickers
        q: Shrinkage parameter
        rho: Target uniform correlation
    
    Returns:
        RiskModelResult with the shrinkage model
    """
    M, N = returns.shape
    C, Psi, sigmas, Y = compute_sample_covariance(returns)
    
    nu = np.ones(N)  # intercept factor
    # Gamma in correlation space
    Gamma = q * (1 - rho) * np.eye(N) + q * rho * np.outer(nu, nu) + (1.0 - q) * Psi
    # Scale to covariance
    outer_sigma = np.outer(sigmas, sigmas)
    Gammahat = Gamma * outer_sigma
    
    # Ensure positive definite with small jitter if needed
    min_eig = np.linalg.eigvalsh(Gammahat).min()
    if min_eig < 1e-10:
        Gammahat += (abs(min_eig) + 1e-8) * np.eye(N)
    
    from scipy.linalg import inv
    try:
        Gammahat_inv = inv(Gammahat)
    except Exception:
        Gammahat_inv = np.linalg.pinv(Gammahat)
    
    xi2 = np.full(N, q * (1 - rho))
    eigenvalues, eigenvectors = eigen_decompose_correlation(Psi)
    
    return RiskModelResult(
        variant=f'shrinkage_uniform_q{q}_rho{rho}',
        K=N,
        tickers=tickers,
        V=eigenvectors,
        lambdas=eigenvalues,
        xi2=xi2,
        sigmas=sigmas,
        Gammahat=Gammahat,
        Gammahat_inv=Gammahat_inv,
        all_eigenvalues=eigenvalues,
    )
