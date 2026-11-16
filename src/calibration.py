"""Finite-dimensional HJB dual optimisation (Section 3.1, p.29-31).

Solves  V(pi) = max_{lambda} { lambda . pi - Phi_0(lambda) }  (eq. 28, 30)
by gradient-based optimisation, using the analytic gradient

    grad_lambda [ lambda . pi - Phi_0(lambda) ] = pi - grad Phi_0(lambda)   (p.31)

computed through the linearised pricing equations (src/linearized.py). At a
stationary point, grad Phi_0(lambda*) = pi (eq. 77), equivalent to the
calibration constraints E^{P^{lambda*}}[G_i(X_tau_p)] = pi_i (eq. 78).

scipy.optimize.minimize (L-BFGS-B) is the quantitative-library tool used for
this unconstrained concave maximisation; the objective and its gradient
(custom implementation, paper eq. 27/76) are supplied analytically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.config import CALIBRATION_MAX_ITER, CALIBRATION_TOL
from src.hjb_solver import CascadeResult, Grid, solve_cascade
from src.linearized import gradient_phi0
from src.payoffs import PayoffFamily


@dataclass
class CalibrationResult:
    lam_star: np.ndarray
    phi0_star: float
    grad_phi0_star: np.ndarray
    value: float  # V(pi) = lambda* . pi - Phi_0(lambda*)
    n_iter: int
    converged: bool
    cascade: CascadeResult


def dual_objective_and_gradient(lam: np.ndarray, pi: np.ndarray, payoffs: PayoffFamily,
                                 x0: float, grid: Grid) -> tuple[float, np.ndarray, CascadeResult]:
    """f(lambda) = Phi_0(lambda) - lambda.pi  (minimised); grad f = grad Phi_0(lambda) - pi."""
    cascade = solve_cascade(lam, payoffs, grid)
    phi0_val = cascade.phi0_value(x0)
    grad = gradient_phi0(lam, payoffs, cascade, x0)
    f = phi0_val - lam @ pi
    g = grad - pi
    return f, g, cascade


def calibrate(pi: np.ndarray, payoffs: PayoffFamily, x0: float, grid: Grid | None = None,
              lam0: np.ndarray | None = None) -> CalibrationResult:
    """Solve V(pi) = max_lambda { lambda.pi - Phi_0(lambda) } via L-BFGS-B (eq. 28)."""
    if pi.shape[0] != payoffs.MP:
        raise ValueError(f"pi has length {pi.shape[0]}, expected MP={payoffs.MP}")
    grid = grid or Grid()
    lam0 = np.zeros(payoffs.MP) if lam0 is None else lam0
    last_cascade: list[CascadeResult] = [None]  # type: ignore[list-item]

    def objective(lam: np.ndarray) -> tuple[float, np.ndarray]:
        f, g, cascade = dual_objective_and_gradient(lam, pi, payoffs, x0, grid)
        last_cascade[0] = cascade
        return f, g

    opt = minimize(objective, lam0, jac=True, method="L-BFGS-B",
                    options={"maxiter": CALIBRATION_MAX_ITER, "gtol": CALIBRATION_TOL})
    lam_star = opt.x
    cascade = last_cascade[0] if last_cascade[0] is not None else solve_cascade(lam_star, payoffs, grid)
    phi0_star = cascade.phi0_value(x0)
    grad_star = gradient_phi0(lam_star, payoffs, cascade, x0)
    value = float(lam_star @ pi - phi0_star)
    return CalibrationResult(
        lam_star=lam_star,
        phi0_star=phi0_star,
        grad_phi0_star=grad_star,
        value=value,
        n_iter=opt.nit,
        converged=bool(opt.success),
        cascade=cascade,
    )
