"""Forward Fokker-Planck solver and Eulerian primal cost (eq. 11-12, p.10).

Rather than a blind grid search over feedback functions (alpha(t,x),
beta(t,x)) -- intractable at PDE-grid resolution and not how the paper
constructs the optimiser -- the primal optimal law is recovered from the HJB
dual optimum via Remark 3.1 (p.31) and the optimality conditions (32)-(33):
the Eulerian minimiser is the law of the feedback SDE (74) with
(alpha*, beta*) = grad_{(a,b)} H(t,x, grad_x phi_p^{lambda*}, (1/2) lap
phi_p^{lambda*}). This module (i) solves the forward Fokker-Planck equation

    d_t rho + d_x(alpha rho) - (1/2) d_x^2(beta rho) = 0,  rho_0 = delta_{x0}  (eq. 12)

with these optimal feedback coefficients, by finite-volume discretisation
(upwind advective flux, central diffusive flux, implicit Euler in time,
zero-flux/reflecting boundaries so that mass is conserved exactly), and (ii)
evaluates the Eulerian primal cost

    integral_0^T integral F(t,x,alpha,beta) rho_t(dx) dt
    = integral_0^T integral [a.alpha + b.beta - H(t,x,a,b)] rho_t(dx) dt       (Young's equality
                                                                                 at the Legendre pair)

using the Fenchel equality F(alpha,beta) = a.alpha + b.beta - H(a,b), which
holds exactly at (alpha,beta) = grad H(a,b) since F = H*.

Custom implementation (paper eq. 11-12, 32-33); no listed quantitative
library provides a Fokker-Planck PDE solver.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.feedback_sde import FeedbackField
from src.hjb_solver import Grid


def _flux_matrix(alpha: np.ndarray, beta: np.ndarray, dx: float, n: int) -> sp.csc_matrix:
    """Sparse divergence-of-flux operator M such that (M rho)_j = RHS_j of d_t rho = RHS(rho)."""
    rows, cols, vals = [], [], []
    alpha_half = 0.5 * (alpha[:-1] + alpha[1:])  # size n-1, at faces j+1/2, j=0..n-2
    for j in range(n - 1):
        a_half = alpha_half[j]
        # Advective upwind coefficients on (rho_j, rho_{j+1})
        if a_half >= 0:
            f_j, f_jp1 = a_half, 0.0
        else:
            f_j, f_jp1 = 0.0, a_half
        # Diffusive central coefficients: G_{j+1/2} = (beta_{j+1} rho_{j+1} - beta_j rho_j) / (2 dx)
        g_j = -beta[j] / (2 * dx)
        g_jp1 = beta[j + 1] / (2 * dx)
        # Total face flux J_{j+1/2} = F - G
        j_coef_j = f_j - g_j
        j_coef_jp1 = f_jp1 - g_jp1
        # RHS_j -= J_{j+1/2}/dx ; RHS_{j+1} += J_{j+1/2}/dx
        rows += [j, j, j + 1, j + 1]
        cols += [j, j + 1, j, j + 1]
        vals += [-j_coef_j / dx, -j_coef_jp1 / dx, j_coef_j / dx, j_coef_jp1 / dx]
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))


def solve_forward_fp(field: FeedbackField, x0: float, t_end: float, grid: Grid,
                      n_t_per_unit_time: int) -> tuple[np.ndarray, np.ndarray]:
    """Implicit finite-volume forward solve of the Fokker-Planck equation (12).

    Returns (t_levels, rho_hist) with rho_hist of shape (n_t+1, n_x); rho_hist[n, j]
    is a density such that sum(rho_hist[n]) * dx == 1 (discrete delta at t=0).
    """
    n_t = max(4, int(round(n_t_per_unit_time * t_end)))
    dt = t_end / n_t
    dx = grid.dx
    n = grid.n_x
    t_levels = np.linspace(0.0, t_end, n_t + 1)

    rho = np.zeros(n)
    j0 = int(np.argmin(np.abs(grid.x - x0)))
    rho[j0] = 1.0 / dx
    rho_hist = np.empty((n_t + 1, n))
    rho_hist[0] = rho

    identity = sp.identity(n, format="csc")
    for k in range(n_t):
        t_next = t_levels[k + 1]
        alpha, beta = field(t_next, grid.x)
        matrix = _flux_matrix(alpha, beta, dx, n)
        system = identity - dt * matrix
        rho = spla.spsolve(system.tocsc(), rho)
        rho_hist[k + 1] = rho
    return t_levels, rho_hist


def primal_cost(field: FeedbackField, t_levels: np.ndarray, rho_hist: np.ndarray, grid: Grid) -> float:
    """integral_0^T integral F(t,x,alpha,beta) rho_t(dx) dt, via F(alpha,beta) = h*(beta) (eq. 11).

    Since the feedback pair always satisfies alpha = -beta/2 (the effective
    domain of F), the running cost reduces to the one-dimensional conjugate
    h*(beta) computed in src.hamiltonian.running_cost_f; no explicit (a,b) is
    required at this stage.
    """
    from src.hamiltonian import running_cost_f

    dt = t_levels[1] - t_levels[0]
    dx = grid.dx
    total = 0.0
    for n, t in enumerate(t_levels):
        _, beta = field(t, grid.x)
        cost_density = running_cost_f(beta)
        total += np.sum(cost_density * rho_hist[n]) * dx * dt
    return float(total)


def price_from_density(payoffs, global_index: int, rho_at_maturity: np.ndarray, grid: Grid) -> float:
    """integral G_i(x) rho_tau_p(dx), the model price of payoff i implied by the primal law."""
    g = payoffs.payoff_value(global_index, grid.x)
    return float(np.sum(g * rho_at_maturity) * grid.dx)
