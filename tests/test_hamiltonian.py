"""Unit tests for src/hamiltonian.py: convexity, Legendre duality, Fenchel equality."""

from __future__ import annotations

import numpy as np

from src.hamiltonian import (
    h,
    h_double_prime,
    h_prime,
    h_prime_inverse,
    hamiltonian,
    hamiltonian_gradient,
    hamiltonian_hessian_quadratic_form,
    running_cost_f,
)
from src.config import KAPPA_MAX, KAPPA_MIN


def test_h_prime_strictly_convex():
    z = np.linspace(-5, 5, 200)
    assert np.all(h_double_prime(z) > 0)


def test_h_prime_range_matches_kappa_bounds():
    z = np.linspace(-50, 50, 500)
    hp = h_prime(z)
    assert np.all(hp > KAPPA_MIN)
    assert np.all(hp < KAPPA_MAX)
    assert abs(hp[0] - KAPPA_MIN) < 1e-3
    assert abs(hp[-1] - KAPPA_MAX) < 1e-3


def test_hamiltonian_convex_in_ab():
    """H(a,b) = h(b - a/2) must be convex: check along random segments."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        a1, b1 = rng.uniform(-1, 1, 2)
        a2, b2 = rng.uniform(-1, 1, 2)
        theta = rng.uniform(0, 1)
        mid_a, mid_b = theta * a1 + (1 - theta) * a2, theta * b1 + (1 - theta) * b2
        h_mid = hamiltonian(np.array([mid_a]), np.array([mid_b]))[0]
        h_convex_combo = theta * hamiltonian(np.array([a1]), np.array([b1]))[0] + \
            (1 - theta) * hamiltonian(np.array([a2]), np.array([b2]))[0]
        assert h_mid <= h_convex_combo + 1e-9


def test_hamiltonian_gradient_feedback_identity():
    """alpha = -beta/2 always holds for the log-martingale Hamiltonian (Remark 3.3)."""
    a = np.linspace(-2, 2, 50)
    b = np.linspace(-1, 1, 50)
    alpha, beta = hamiltonian_gradient(a, b)
    assert np.allclose(alpha, -0.5 * beta)
    assert np.all(beta > KAPPA_MIN - 1e-9)
    assert np.all(beta < KAPPA_MAX + 1e-9)


def test_h_prime_inverse_is_inverse_of_h_prime():
    z = np.linspace(-10, 10, 100)
    beta = h_prime(z)
    z_rec = h_prime_inverse(beta)
    assert np.allclose(z, z_rec, atol=1e-6)


def test_fenchel_equality_running_cost():
    """F(alpha,beta) = a.alpha + b.beta - H(a,b) at the Legendre pair (alpha,beta)=grad H(a,b)."""
    a = np.linspace(-2, 2, 30)
    b = np.linspace(-1, 1, 30)
    alpha, beta = hamiltonian_gradient(a, b)
    h_val = hamiltonian(a, b)
    expected = a * alpha + b * beta - h_val
    actual = running_cost_f(beta)
    assert np.allclose(expected, actual, atol=1e-6)


def test_hessian_quadratic_form_nonnegative():
    a = np.linspace(-2, 2, 20)
    b = np.linspace(-1, 1, 20)
    p = np.ones_like(a)
    q = np.ones_like(a) * 0.5
    quad = hamiltonian_hessian_quadratic_form(a, b, p, q)
    assert np.all(quad >= 0)
