"""HJB cascade solver (eq. 22, p.14) and dual value map Phi_0 (eq. 27, p.16).

The cascade is solved backward in real time t, interval by interval, from
p = P down to p = 1:

    d_t phi_p + H(t, x, grad_x phi_p, (1/2) grad_x^2 phi_p) = 0   on (tau_{p-1}, tau_p) x R,
    phi_p(tau_p, .) = phi_{p+1}(tau_p, .) + g_p^lambda,            phi_{P+1}(tau_P, .) = 0.

The paper's analysis is carried out on the full R^d (Section 1.4). Here the
state space is truncated to [-L, L] with Neumann (zero-gradient) boundary
conditions, a numerical necessity given the bounded uniformly continuous
payoffs; this truncation is recorded as a deviation in results/RESULTS.md.

Since H has no explicit (t, x) dependence for the log-martingale example, the
PDE is solved with the pseudo-time substitution s = tau_p - t, so that
d_s phi = H(grad phi, (1/2) lap phi). Each real time step is an implicit,
Picard-linearised ("policy iteration") step: the Hamiltonian is linearised
around the current spatial profile, giving a tridiagonal linear system solved
with scipy.sparse.linalg.spsolve. Custom implementation (paper eq. 22); no
general nonlinear-HJB finite-difference solver is available in the listed
quantitative libraries (skfolio, pyportfolioopt, vectorbt, arch, ... do not
cover PDE solvers), so this is implemented from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.config import GRID_HALF_WIDTH, GRID_N_T_PER_UNIT_TIME, GRID_N_X, PICARD_SWEEPS
from src.hamiltonian import hamiltonian, hamiltonian_gradient
from src.payoffs import PayoffFamily


@dataclass
class Grid:
    """Truncated spatial grid [-L, L] with GRID_N_X points."""

    L: float = GRID_HALF_WIDTH
    n_x: int = GRID_N_X

    def __post_init__(self) -> None:
        self.x = np.linspace(-self.L, self.L, self.n_x)
        self.dx = self.x[1] - self.x[0]


def _first_derivative(phi: np.ndarray, dx: float) -> np.ndarray:
    """Central difference grad_x phi, one-sided (Neumann-consistent) at the boundary."""
    a = np.empty_like(phi)
    a[1:-1] = (phi[2:] - phi[:-2]) / (2 * dx)
    a[0] = (phi[1] - phi[0]) / dx
    a[-1] = (phi[-1] - phi[-2]) / dx
    return a


def _second_derivative(phi: np.ndarray, dx: float) -> np.ndarray:
    """Central difference grad_x^2 phi, zero at the (Neumann) boundary nodes."""
    b = np.zeros_like(phi)
    b[1:-1] = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dx ** 2
    return b


def _assemble_and_solve(phi_old: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                         source: np.ndarray, ds: float, dx: float) -> np.ndarray:
    """One implicit step of  phi_new - ds*(alpha*D1 + beta/2*D2) phi_new = phi_old + ds*source,

    with Neumann boundary rows phi_new[0] = phi_new[1] and phi_new[-1] = phi_new[-2].
    """
    n = phi_old.shape[0]
    lower = ds * alpha / (2 * dx) - ds * beta / (2 * dx ** 2)
    diag = 1.0 + ds * beta / dx ** 2
    upper = -ds * alpha / (2 * dx) - ds * beta / (2 * dx ** 2)
    rhs = phi_old + ds * source

    main = np.empty(n)
    lo = np.zeros(n - 1)
    up = np.zeros(n - 1)
    main[1:-1] = diag[1:-1]
    lo[:-1] = lower[1:-1]
    up[1:] = upper[1:-1]
    # Neumann boundary: phi[0] - phi[1] = 0 ; phi[-1] - phi[-2] = 0
    main[0] = 1.0
    up[0] = -1.0
    rhs[0] = 0.0
    main[-1] = 1.0
    lo[-1] = -1.0
    rhs[-1] = 0.0

    mat = sp.diags([lo, main, up], offsets=[-1, 0, 1], format="csc")
    return spla.spsolve(mat, rhs)


@dataclass
class IntervalSolution:
    """Stored pseudo-time history of one maturity interval, used for the linearised PDEs."""

    t_levels: np.ndarray  # real times, ascending, shape (n_t+1,): t_levels[0]=tau_{p-1}, [-1]=tau_p
    phi: np.ndarray  # shape (n_t+1, n_x), phi[0] = phi_p(tau_{p-1}, .), phi[-1] = terminal data
    alpha: np.ndarray  # shape (n_t+1, n_x), feedback drift at the corresponding time level
    beta: np.ndarray  # shape (n_t+1, n_x), feedback covariance at the corresponding time level


@dataclass
class CascadeResult:
    grid: Grid
    intervals: list[IntervalSolution] = field(default_factory=list)  # intervals[0] is p=1, ..., intervals[-1] is p=P

    def phi0_value(self, x0: float) -> float:
        """Phi_0(lambda) = phi_1^lambda(0, x0), eq. (27), mu_0 = delta_{x0}."""
        first = self.intervals[0]
        return float(np.interp(x0, self.grid.x, first.phi[0]))


def solve_interval(terminal_profile: np.ndarray, t0: float, t1: float, grid: Grid) -> IntervalSolution:
    """Backward induction on (t0, t1): solve d_t phi + H(grad phi, lap phi /2) = 0, phi(t1,.)=terminal."""
    n_t = max(4, int(round(GRID_N_T_PER_UNIT_TIME * (t1 - t0))))
    ds = (t1 - t0) / n_t
    dx = grid.dx

    phi_hist = np.empty((n_t + 1, grid.n_x))
    alpha_hist = np.empty((n_t + 1, grid.n_x))
    beta_hist = np.empty((n_t + 1, grid.n_x))
    t_levels = np.linspace(t0, t1, n_t + 1)

    phi_hist[-1] = terminal_profile
    a0 = _first_derivative(terminal_profile, dx)
    b0 = 0.5 * _second_derivative(terminal_profile, dx)
    alpha_hist[-1], beta_hist[-1] = hamiltonian_gradient(a0, b0)

    phi_current = terminal_profile.copy()
    for n in range(n_t - 1, -1, -1):
        phi_guess = phi_current.copy()
        for _ in range(PICARD_SWEEPS):
            a_g = _first_derivative(phi_guess, dx)
            b_g = 0.5 * _second_derivative(phi_guess, dx)
            alpha_g, beta_g = hamiltonian_gradient(a_g, b_g)
            source = hamiltonian(a_g, b_g) - alpha_g * a_g - beta_g * b_g
            phi_guess = _assemble_and_solve(phi_current, alpha_g, beta_g, source, ds, dx)
        phi_current = phi_guess
        phi_hist[n] = phi_current
        a_f = _first_derivative(phi_current, dx)
        b_f = 0.5 * _second_derivative(phi_current, dx)
        alpha_hist[n], beta_hist[n] = hamiltonian_gradient(a_f, b_f)

    return IntervalSolution(t_levels=t_levels, phi=phi_hist, alpha=alpha_hist, beta=beta_hist)


def solve_cascade(lam: np.ndarray, payoffs: PayoffFamily, grid: Grid | None = None) -> CascadeResult:
    """Backward induction of the HJB cascade (22) for a given multiplier lambda in R^{MP}."""
    if lam.shape[0] != payoffs.MP:
        raise ValueError(f"lambda has length {lam.shape[0]}, expected MP={payoffs.MP}")
    grid = grid or Grid()
    taus = (0.0,) + payoffs.maturities
    result = CascadeResult(grid=grid)
    phi_next_at_tau_p = np.zeros(grid.n_x)  # phi_{P+1}(tau_P, .) = 0
    intervals_reversed: list[IntervalSolution] = []
    for p in range(payoffs.P - 1, -1, -1):
        terminal = phi_next_at_tau_p + payoffs.g_lambda(p, lam, grid.x)
        interval_sol = solve_interval(terminal, taus[p], taus[p + 1], grid)
        intervals_reversed.append(interval_sol)
        phi_next_at_tau_p = interval_sol.phi[0]
    result.intervals = list(reversed(intervals_reversed))
    return result


def phi0(lam: np.ndarray, payoffs: PayoffFamily, x0: float, grid: Grid | None = None) -> float:
    """Phi_0(lambda) = phi_1^lambda(0, x0), eq. (27)."""
    return solve_cascade(lam, payoffs, grid).phi0_value(x0)
