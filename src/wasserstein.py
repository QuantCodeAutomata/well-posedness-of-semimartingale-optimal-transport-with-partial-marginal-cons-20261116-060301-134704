"""Adapted Wasserstein distance AW2 bound via common-noise coupling (Section 3.3, p.36-39).

Definition 3.2 (eq. 87, p.36) defines AW2 as an infimum over bicausal
couplings. This module does not search over couplings: it constructs the
specific bicausal coupling of Definition 3.1 obtained by driving both
feedback SDEs (74) with the *same* Brownian motion and the same deterministic
initial condition X_0 = x0 (eq. 89, p.36). Because both solutions are strong
solutions adapted to the common Brownian filtration, the coupling is
bicausal: the future of each process, given the present of both, depends
only on future Brownian increments that are independent of the other
process's filtration (Section 3.3). This coupling gives the upper bound

    AW2^2(P^{*,pi_n}, P^{*,pi}) <=
        E[int_0^T ||sigma^{lambda_n}(t,X_t^n) - sigma^lambda(t,X_t)||^2 dt]
        + E[(int_0^T |alpha^{lambda_n}(t,X_t^n) - alpha^lambda(t,X_t)| dt)^2]      (eq. 97, p.39)

estimated here by a Monte Carlo average over simulated path pairs, together
with the trajectory bound E[sup_t |X_t^n - X_t|^2] (eq. 96, p.39).

Custom implementation (paper eq. 87-97); no listed library computes the
adapted/bicausal Wasserstein distance for continuous-time feedback SDEs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.feedback_sde import FeedbackField


@dataclass
class AW2Estimate:
    aw2_upper_bound_sq: float  # Monte Carlo estimate of the right-hand side of (97)
    trajectory_sup_sq: float  # Monte Carlo estimate of E[sup_t |X^n_t - X_t|^2], eq. (96)
    n_paths: int


def common_noise_aw2_upper_bound(field_n: FeedbackField, field: FeedbackField, x0: float, t_end: float,
                                  n_steps_per_unit_time: int, n_paths: int,
                                  rng: np.random.Generator) -> AW2Estimate:
    """Simulate X^n and X driven by the same Brownian path (eq. 89) and estimate (96)-(97)."""
    n_steps = max(1, int(round(n_steps_per_unit_time * t_end)))
    dt = t_end / n_steps
    sqrt_dt = np.sqrt(dt)
    increments = rng.normal(size=(n_steps, n_paths))

    x_n = np.full(n_paths, x0, dtype=float)
    x = np.full(n_paths, x0, dtype=float)
    sup_diff_sq = np.zeros(n_paths)
    diffusion_integral = np.zeros(n_paths)  # integral ||sigma_n - sigma||^2 dt
    drift_integral = np.zeros(n_paths)  # integral |alpha_n - alpha| dt

    t = 0.0
    for n in range(n_steps):
        alpha_n, beta_n = field_n(t, x_n)
        alpha, beta = field(t, x)
        sigma_n = np.sqrt(np.maximum(beta_n, 1e-12))
        sigma = np.sqrt(np.maximum(beta, 1e-12))

        diffusion_integral += (sigma_n - sigma) ** 2 * dt
        drift_integral += np.abs(alpha_n - alpha) * dt

        dw = sqrt_dt * increments[n]
        x_n = x_n + alpha_n * dt + sigma_n * dw
        x = x + alpha * dt + sigma * dw
        t += dt
        sup_diff_sq = np.maximum(sup_diff_sq, (x_n - x) ** 2)

    aw2_sq_paths = diffusion_integral + drift_integral ** 2
    return AW2Estimate(
        aw2_upper_bound_sq=float(np.mean(aw2_sq_paths)),
        trajectory_sup_sq=float(np.mean(sup_diff_sq)),
        n_paths=n_paths,
    )
