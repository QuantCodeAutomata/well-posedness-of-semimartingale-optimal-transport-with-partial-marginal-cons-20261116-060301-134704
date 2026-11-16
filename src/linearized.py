"""Linearised pricing equations (75)-(76), p.29-30: gradient of Phi_0.

For each payoff index i = i(p, m), the linearised cascade

    d_t u_{q,i} + L_t^lambda u_{q,i} = 0,   L_t^lambda v = alpha^lambda . grad v + (beta^lambda/2) lap v,
    u_{q,i}(tau_q, .) = u_{q+1,i}(tau_q, .) + 1_{q=p} G_i,   u_{P+1,i} = 0,

is solved backward with the feedback coefficients (alpha^lambda, beta^lambda)
already computed and stored while solving the nonlinear cascade (22). This is
a fully linear parabolic PDE (coefficients fixed by the already-solved
nonlinear solution), so a single implicit Euler step per pseudo-time level
suffices -- no Picard sweeps are required, unlike the nonlinear cascade in
src/hjb_solver.py. Custom implementation (paper eq. 75); reuses the same
tridiagonal assembly as the nonlinear solver's linearisation step.

The gradient identity (eq. 76, p.30) is

    d Phi_0 / d lambda_i (lambda) = integral u_{1,i}^lambda(0, x) mu_0(dx) = E^{P^lambda}[G_i(X_tau_p)],

which for mu_0 = delta_{x0} reduces to u_{1,i}^lambda(0, x0).
"""

from __future__ import annotations

import numpy as np

from src.hjb_solver import CascadeResult, Grid, _assemble_and_solve
from src.payoffs import PayoffFamily


def _solve_linear_interval(terminal_profile: np.ndarray, interval, grid: Grid) -> np.ndarray:
    """Backward implicit solve of the linear PDE on one interval, reusing stored (alpha,beta)."""
    n_t = interval.alpha.shape[0] - 1
    ds = interval.t_levels[1] - interval.t_levels[0]
    u = terminal_profile.copy()
    zero_source = np.zeros(grid.n_x)
    for n in range(n_t - 1, -1, -1):
        u = _assemble_and_solve(u, interval.alpha[n], interval.beta[n], zero_source, ds, grid.dx)
    return u


def gradient_phi0(lam: np.ndarray, payoffs: PayoffFamily, cascade: CascadeResult, x0: float) -> np.ndarray:
    """Full gradient vector grad Phi_0(lambda) in R^{MP} via the linearised equations (75)-(76)."""
    grid = cascade.grid
    grad = np.zeros(payoffs.MP)
    for p_target in range(payoffs.P):
        for i in payoffs.flat_index[p_target]:
            carry = np.zeros(grid.n_x)
            for q in range(p_target, -1, -1):
                interval = cascade.intervals[q]
                terminal = carry + (payoffs.payoff_value(i, grid.x) if q == p_target else 0.0)
                carry = _solve_linear_interval(terminal, interval, grid)
            grad[i] = float(np.interp(x0, grid.x, carry))
    return grad
