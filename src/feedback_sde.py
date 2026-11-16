"""Feedback SDE dX_t = alpha^lambda(t,X_t) dt + sigma^lambda(t,X_t) dW_t  (eq. 74, p.29).

The feedback drift and diffusion coefficients (alpha^lambda, beta^lambda) are
obtained from the HJB cascade solution via the Hamiltonian gradient (eq. 86,
p.35): (alpha, beta) = grad_{(a,b)} H(t, x, grad_x phi_p^lambda, (1/2) lap phi_p^lambda).
They are stored on the cascade's (time level, spatial grid) and are
interpolated in between with scipy.interpolate.RegularGridInterpolator, as
suggested by the implementation notes of exp_5. In dimension d=1,
sigma^lambda = sqrt(beta^lambda) (real since beta^lambda >= KAPPA_MIN > 0).

Custom implementation (paper eq. 74, 86); Euler-Maruyama simulation uses
numpy only (no higher-level SDE library covers feedback coefficients obtained
from a PDE grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.hjb_solver import CascadeResult


@dataclass
class FeedbackField:
    """A callable (t, x) -> (alpha, beta) built by interpolating one cascade's stored fields."""

    cascade: CascadeResult

    def __post_init__(self) -> None:
        x = self.cascade.grid.x
        t_all = []
        alpha_all = []
        beta_all = []
        for interval in self.cascade.intervals:
            # Drop the last level of each interval except the very last to avoid duplicate t.
            t_all.append(interval.t_levels[:-1])
            alpha_all.append(interval.alpha[:-1])
            beta_all.append(interval.beta[:-1])
        t_all.append(self.cascade.intervals[-1].t_levels[-1:])
        alpha_all.append(self.cascade.intervals[-1].alpha[-1:])
        beta_all.append(self.cascade.intervals[-1].beta[-1:])
        t_grid = np.concatenate(t_all)
        alpha_grid = np.concatenate(alpha_all, axis=0)
        beta_grid = np.concatenate(beta_all, axis=0)
        self._alpha_interp = RegularGridInterpolator((t_grid, x), alpha_grid,
                                                       bounds_error=False, fill_value=None)
        self._beta_interp = RegularGridInterpolator((t_grid, x), beta_grid,
                                                      bounds_error=False, fill_value=None)
        self.t_min, self.t_max = t_grid[0], t_grid[-1]
        self.x_min, self.x_max = x[0], x[-1]

    def __call__(self, t: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_clip = np.clip(x, self.x_min, self.x_max)
        t_clip = np.clip(t, self.t_min, self.t_max)
        pts = np.stack([np.full_like(x_clip, t_clip), x_clip], axis=-1)
        alpha = self._alpha_interp(pts)
        beta = self._beta_interp(pts)
        return alpha, beta


def simulate_paths(field: FeedbackField, x0: float, maturities: tuple[float, ...], n_paths: int,
                    n_steps_per_unit_time: int, rng: np.random.Generator,
                    brownian_increments: np.ndarray | None = None) -> dict[float, np.ndarray]:
    """Euler-Maruyama simulation of (74); returns {maturity: X_tau_p array of shape (n_paths,)}.

    If ``brownian_increments`` is given (shape (n_total_steps, n_paths)) it is
    used instead of fresh draws, allowing two feedback fields to be simulated
    against the same Brownian path (common-noise coupling, eq. 89, p.36).
    """
    t_end = maturities[-1]
    n_steps = max(1, int(round(n_steps_per_unit_time * t_end)))
    dt = t_end / n_steps
    x = np.full(n_paths, x0, dtype=float)
    results: dict[float, np.ndarray] = {}
    maturity_set = set(maturities)
    sqrt_dt = np.sqrt(dt)
    if brownian_increments is None:
        brownian_increments = rng.normal(size=(n_steps, n_paths))
    t = 0.0
    for n in range(n_steps):
        alpha, beta = field(t, x)
        sigma = np.sqrt(np.maximum(beta, 1e-12))
        x = x + alpha * dt + sigma * sqrt_dt * brownian_increments[n]
        t += dt
        for tau in maturity_set:
            if abs(t - tau) < dt / 2:
                results[tau] = x.copy()
    return results
