# AGENTS.md — Repository Knowledge Base

## Project Overview
Replication of **Kakushadze (2015) "Statistical Risk Models"** (SSRN) across 6 experiments.

- **N = 90 tickers** (S&P-100 proxy), **I = $20 M gross exposure**, **bound_fraction = 0.01**
- **1429 trading-day** backtest window (2008-08-01 → 2014-09-05), **21-trading-day rebalance blocks**
- Data fetched via **Massive REST API** (`MASSIVE_TOKEN` env var); cached to `data/full_cache.pkl` and `data/smoke_cache.pkl` (both gitignored via `*.pkl`)

---

## Repository Structure

```
data/           data_loader.py — Massive API fetch + pkl caching
src/            Core library modules (no side effects, pure functions)
  risk_models.py    PCA, K-selection (minimization, eRank), shrinkage models
  portfolio.py      Unbounded closed-form + bounded cvxpy/OSQP solver
  backtest.py       21-day block engine, daily P&L, shares traded
  universe.py       ADDV ranking, 21-day universe freeze
  metrics.py        ROC, annualized Sharpe, CPS
  computational.py  No-iterations Gram-matrix algo + power-iterations benchmark
exp/            Runnable experiment scripts (import from src/)
tests/          pytest (71 tests, all passing)
results/        Deliverable CSVs, PNGs, logs, RESULTS.md
```

---

## Key Implementation Patterns

### Risk Model (src/risk_models.py)
- **Traditional truncated-PC model**: `Gamma_ij = xi_i^2 δ_ij + Σ_{A=1}^K λ^(A) V_i^(A) V_j^(A)`
  where `xi_i^2 = 1 - Σ_{A=1}^K λ^(A) (V_i^(A))^2`
- Convert correlation → covariance: `Gammahat_ij = σ_i σ_j Γ_ij`
- **Woodbury identity** for fast inversion: `(D + VΛV^T)^{-1} = D^{-1} - D^{-1}V(I + V^T D^{-1} V)^{-1} V^T D^{-1}`
- Six K-selection variants (M1–M6): minimization (`g(K) = sqrt(min ξ²)+sqrt(max ξ²)`), exclude-first, eRank (Round/floor), eRank on Ψ'

### Portfolio (src/portfolio.py)
- **Unbounded**: closed-form `H_i = -η [Γ^{-1} E - Γ^{-1} 1 (1^T Γ^{-1} E)/(1^T Γ^{-1} 1)]`, rescaled to `|H|_1 = I`
- **Bounded**: cvxpy QP — `minimize 0.5 H^T Γ̂ H - λ E^T H` s.t. `sum(H)=0`, `|H_i|≤0.01 A_i`; outer bisection loop on λ to meet gross-exposure target
- Solver: OSQP (via cvxpy)

### Computational (src/computational.py)
- **No-iterations algo**: Cholesky of φ → Ỹ = Yφ → Gram matrix G (M×M) → eigh(G) → V^(a) = (1/√ρ^(a)) Ỹ U^(a)
- **Power iterations**: uniform init for PC1, random unit-norm init for PC2+, successive deflation; convergence criterion `‖V_{r+1} − V_r‖ < 1e-3`
- Expected total iterations ≈ 5037 for N=2339, M=19 (paper Table 4)

---

## Data Notes
- **Adjusted prices** used for returns and signal; **unadjusted** for execution P&L and ADDV
- Missing prices/volumes: exclude names, never impute zeros
- Survivorship bias intentionally preserved (universe as of 2014-09-06)
- Cache pkl files are gitignored; re-run `data/data_loader.py` to regenerate

---

## Results Summary (from results/RESULTS.md)
| Exp | Setting | Best variant | SR | ROC |
|-----|---------|-------------|------|-----|
| 1 | Unbounded dynamic-K | M3 Round(eRank) | 6.69 | ~40% |
| 2 | Bounded dynamic-K | M1 minimization | ~6.5 | ~33% |
| 3 | Computational | No-iters vs power | power iters >> M | — |
| 4 | Shrinkage | Uniform ρ=0.9 | 6.68 | ~40% |
| 5 | Fixed-K unbounded | K=9 | 6.55 | — |
| 5 | Fixed-K bounded | K=8 | 6.47 | — |
| 6 | Specific-risk robustness | M3 adaptive | 6.69 | — |

---

## GitHub
**Remote**: `https://github.com/QuantCodeAutomata/qca-traditional-truncated-pc-statistical-risk-models-w`

---

## Common Gotchas
1. `git init` needed — no pre-existing repo; org endpoint returns 404 for user accounts; use `/user/repos`
2. `.pkl` cache files must stay gitignored; do NOT commit raw Massive API responses
3. cvxpy OSQP needs `cp.Problem.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)`
4. K must be capped at `M-1` to preserve positive-definiteness of the specific-risk diagonal
5. eRank uses **only positive eigenvalues** (λ > 0) for the entropy calculation
6. Unbounded portfolio: the **minus sign** in `H_i = -η [...]` is intentional (mean reversion)
7. For exclude-first variants (M5/M6), iterate K from 2 upward (skip PC1) per Appendix A
