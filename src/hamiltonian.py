"""Running cost F and Hamiltonian H = F* for the log-martingale constraint.

Remark 3.3 (p.32) states that, in dimension one, for a running cost
independent of (t,x), the log-martingale constraint alpha = -beta/2 produces
Hamiltonians of the form

    H(a, b) = h(b - a/2)                                            (Remark 3.3)

for a smooth convex function h. (The paraphrase supplied for this experiment
wrote "h(b - a^2/2)"; this is checked against Remark 3.3 of the paper itself,
which states the Hamiltonian explicitly as h(b - a/2). The paper form is used
here; see "Deviations from the paper" in results/RESULTS.md.) This also
follows directly from the Legendre transform: with F(alpha, beta) =
1_{alpha=-beta/2} f(beta),

    F*(a, b) = sup_beta [ a*(-beta/2) + b*beta - f(beta) ]
             = sup_beta [ (b - a/2)*beta - f(beta) ] = f*(b - a/2).

Custom implementation (not given explicitly in the paper): we choose
h'(z) = KAPPA_MIN + (KAPPA_MAX - KAPPA_MIN) * sigmoid(z / KAPPA_SCALE), i.e.

    h(z) = KAPPA_MIN * z
           + (KAPPA_MAX - KAPPA_MIN) * KAPPA_SCALE * log(1 + exp(z / KAPPA_SCALE)).

This h is smooth, strictly convex (h'' > 0 everywhere), with globally Lipschitz
Hessian, and h'(z) ranges over the open interval (KAPPA_MIN, KAPPA_MAX). Since
beta^lambda = dH/db = h'(z), this gives the bounded effective domain and the
uniform ellipticity beta >= KAPPA_MIN > 0 required by Assumption 3 (D1)-(D2,
p.14) and the smoothness/boundedness of Assumption 6(i) (p.31).
"""

from __future__ import annotations

import numpy as np

from src.config import KAPPA_MAX, KAPPA_MIN, KAPPA_SCALE


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def h(z: np.ndarray, kappa_min: float = KAPPA_MIN, kappa_max: float = KAPPA_MAX,
      kappa_scale: float = KAPPA_SCALE) -> np.ndarray:
    """Smooth strictly convex function h with h' in (kappa_min, kappa_max)."""
    z = np.asarray(z, dtype=float)
    # log(1+exp(z/scale)) computed with the stable softplus identity.
    scaled = z / kappa_scale
    softplus = np.logaddexp(0.0, scaled)
    return kappa_min * z + (kappa_max - kappa_min) * kappa_scale * softplus


def h_prime(z: np.ndarray, kappa_min: float = KAPPA_MIN, kappa_max: float = KAPPA_MAX,
            kappa_scale: float = KAPPA_SCALE) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return kappa_min + (kappa_max - kappa_min) * _sigmoid(z / kappa_scale)


def h_double_prime(z: np.ndarray, kappa_min: float = KAPPA_MIN, kappa_max: float = KAPPA_MAX,
                    kappa_scale: float = KAPPA_SCALE) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    s = _sigmoid(z / kappa_scale)
    return (kappa_max - kappa_min) / kappa_scale * s * (1.0 - s)


def hamiltonian(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """H(a, b) = h(b - a/2), the Hamiltonian H = F* (Remark 3.3)."""
    return h(b - 0.5 * a)


def hamiltonian_gradient(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(alpha, beta) = grad_{(a,b)} H(a,b) = (-1/2 h'(z), h'(z)), z = b - a/2 (eq. 86, p.35)."""
    z = b - 0.5 * a
    hp = h_prime(z)
    return -0.5 * hp, hp


def h_prime_inverse(beta: np.ndarray, kappa_min: float = KAPPA_MIN, kappa_max: float = KAPPA_MAX,
                     kappa_scale: float = KAPPA_SCALE) -> np.ndarray:
    """z = (h')^{-1}(beta), the logit inverse of the logistic h' (used for the Fenchel equality)."""
    beta = np.asarray(beta, dtype=float)
    p = np.clip((beta - kappa_min) / (kappa_max - kappa_min), 1e-12, 1 - 1e-12)
    return kappa_scale * np.log(p / (1 - p))


def running_cost_f(beta: np.ndarray, kappa_min: float = KAPPA_MIN, kappa_max: float = KAPPA_MAX,
                    kappa_scale: float = KAPPA_SCALE) -> np.ndarray:
    """F(alpha, beta) restricted to the effective domain alpha = -beta/2 (Section 2.2, D1-D2, p.14).

    Since alpha = -(1/2) h'(z) and beta = h'(z) always hold at a feedback pair
    produced by grad H, alpha = -beta/2 identically; the running cost reduces
    to the one-dimensional convex conjugate F(alpha,beta) = h*(beta) = z*beta
    - h(z) with z = (h')^{-1}(beta), the standard Legendre-Fenchel identity
    F(grad H(a,b)) = a.grad_a H + b.grad_b H - H(a,b) for H = F*.
    """
    z = h_prime_inverse(beta, kappa_min, kappa_max, kappa_scale)
    return z * beta - h(z, kappa_min, kappa_max, kappa_scale)


def hamiltonian_hessian_quadratic_form(a: np.ndarray, b: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hess H(a,b)[(p,q),(p,q)] = h''(b - a/2) * (q - p/2)^2, Remark 3.3 (p.32).

    Equivalent to the paper's stated (1/4) h''(b-a/2) (grad^2 u - grad u)^2
    with p = grad_x u, q = (1/2) grad_x^2 u.
    """
    z = b - 0.5 * a
    return h_double_prime(z) * (q - 0.5 * p) ** 2
