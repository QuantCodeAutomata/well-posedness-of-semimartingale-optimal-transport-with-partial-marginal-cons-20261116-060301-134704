"""Local strong convexity of Phi_0 and uniqueness/Lipschitz stability of lambda*(pi).

Theorem 3.1 (p.33): Phi_0 is locally strongly convex, i.e. for every compact
convex K there is m_K > 0 such that for lambda_1, lambda_2 in K, theta in [0,1],

    Phi_0((1-theta)l1 + theta l2) <= (1-theta)Phi_0(l1) + theta Phi_0(l2)
                                      - (m_K/2) theta(1-theta) |l2-l1|^2.       (80)

Corollary 3.5 / Proposition 3.6 (p.34): the admissible price set P has nonempty
interior, and on int(P) the dual optimiser lambda*(pi) is unique and locally
Lipschitz in pi, with V in C^1(int(P)) and grad V = lambda*.

Custom implementation of the paper's verification procedure (sampling the
convexity defect, re-running the dual optimiser from several starting points,
and perturbing pi); scipy.optimize.minimize (src/calibration.py) supplies the
dual optimiser itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.calibration import calibrate
from src.hjb_solver import Grid, solve_cascade
from src.payoffs import PayoffFamily


def convexity_defect_modulus(payoffs: PayoffFamily, x0: float, grid: Grid, box_half_width: float,
                              n_samples: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Monte Carlo estimate of m_K = inf over sampled (l1,l2,theta) of the normalised defect in (80)."""
    defects = np.empty(n_samples)
    for s in range(n_samples):
        l1 = rng.uniform(-box_half_width, box_half_width, size=payoffs.MP)
        l2 = rng.uniform(-box_half_width, box_half_width, size=payoffs.MP)
        theta = rng.uniform(0.05, 0.95)
        diff_sq = float(np.sum((l2 - l1) ** 2))
        if diff_sq < 1e-10:
            defects[s] = np.inf
            continue
        phi_mix = solve_cascade((1 - theta) * l1 + theta * l2, payoffs, grid).phi0_value(x0)
        phi_1 = solve_cascade(l1, payoffs, grid).phi0_value(x0)
        phi_2 = solve_cascade(l2, payoffs, grid).phi0_value(x0)
        convex_combo = (1 - theta) * phi_1 + theta * phi_2
        defect = convex_combo - phi_mix
        defects[s] = 2 * defect / (theta * (1 - theta) * diff_sq)
    return float(np.min(defects)), defects


@dataclass
class UniquenessCheck:
    lam_stars: np.ndarray  # shape (n_starts, MP)
    max_pairwise_distance: float
    unique: bool


def check_uniqueness(pi: np.ndarray, payoffs: PayoffFamily, x0: float, grid: Grid,
                      n_starts: int, rng: np.random.Generator, tol: float = 1e-3) -> UniquenessCheck:
    """Run the dual optimiser from several starting points; all should converge to the same lambda*."""
    stars = []
    for _ in range(n_starts):
        lam0 = rng.normal(scale=0.3, size=payoffs.MP)
        result = calibrate(pi, payoffs, x0, grid, lam0=lam0)
        stars.append(result.lam_star)
    stars = np.array(stars)
    dists = [np.linalg.norm(stars[i] - stars[j]) for i in range(len(stars)) for j in range(i + 1, len(stars))]
    max_dist = float(np.max(dists)) if dists else 0.0
    return UniquenessCheck(lam_stars=stars, max_pairwise_distance=max_dist, unique=max_dist < tol)


@dataclass
class LipschitzCheck:
    pi_distances: np.ndarray
    lambda_distances: np.ndarray
    ratios: np.ndarray
    estimated_constant: float


def estimate_lipschitz_constant(pi0: np.ndarray, lam0_star: np.ndarray, payoffs: PayoffFamily, x0: float,
                                 grid: Grid, perturbation_scale: float, n_directions: int,
                                 rng: np.random.Generator) -> LipschitzCheck:
    """Perturb pi0 in n_directions random directions and estimate the local Lipschitz constant of lambda*(.)."""
    pi_d = np.empty(n_directions)
    lam_d = np.empty(n_directions)
    for d in range(n_directions):
        direction = rng.normal(size=payoffs.MP)
        direction /= np.linalg.norm(direction)
        pi_pert = pi0 + perturbation_scale * direction
        result = calibrate(pi_pert, payoffs, x0, grid, lam0=lam0_star)
        pi_d[d] = float(np.linalg.norm(pi_pert - pi0))
        lam_d[d] = float(np.linalg.norm(result.lam_star - lam0_star))
    ratios = lam_d / pi_d
    return LipschitzCheck(pi_distances=pi_d, lambda_distances=lam_d, ratios=ratios,
                           estimated_constant=float(np.max(ratios)))


def value_function_gradient_check(pi0: np.ndarray, lam0_star: np.ndarray, payoffs: PayoffFamily, x0: float,
                                   grid: Grid, fd_step: float) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference grad V(pi0) compared with lambda*(pi0): both should equal lambda* (eq. 31).

    Uses a central difference: the local curvature of V = Phi_0^* scales as 1/m_K
    (Theorem 3.1), which can be large when the strong-convexity modulus m_K is
    small, so a one-sided difference carries an O(h) truncation bias of order
    h * ||Hess V||; the centered scheme reduces this to O(h^2).
    """
    mp = payoffs.MP
    value_at = lambda pi: calibrate(pi, payoffs, x0, grid, lam0=lam0_star).value
    grad_v = np.zeros(mp)
    for i in range(mp):
        pi_plus, pi_minus = pi0.copy(), pi0.copy()
        pi_plus[i] += fd_step
        pi_minus[i] -= fd_step
        grad_v[i] = (value_at(pi_plus) - value_at(pi_minus)) / (2.0 * fd_step)
    return grad_v, lam0_star
