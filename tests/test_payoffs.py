"""Unit tests for src/payoffs.py: log-put payoff family bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from src.payoffs import PayoffFamily


def test_rejects_duplicate_strikes():
    with pytest.raises(ValueError):
        PayoffFamily(maturities=(1.0,), strikes_per_maturity=((1.0, 1.0),))


def test_rejects_nonpositive_strikes():
    with pytest.raises(ValueError):
        PayoffFamily(maturities=(1.0,), strikes_per_maturity=((-1.0, 1.0),))


def test_flat_index_bookkeeping(small_payoffs):
    assert small_payoffs.P == 2
    assert small_payoffs.MP == 4
    assert small_payoffs.flat_index == [[0, 1], [2, 3]]


def test_payoff_is_nonnegative_and_bounded(small_payoffs):
    x = np.linspace(-30, 30, 200)
    for i in range(small_payoffs.MP):
        g = small_payoffs.payoff_value(i, x)
        _, k = small_payoffs.global_to_pk[i]
        assert np.all(g >= 0.0)
        assert np.all(g <= k)
        # BUC: tends to k as x -> -infty, to 0 as x -> +infty
        assert g[0] == pytest.approx(k, abs=1e-6)
        assert g[-1] == pytest.approx(0.0, abs=1e-6)


def test_g_lambda_linear_in_lambda(small_payoffs):
    x = np.linspace(-2, 2, 50)
    lam1 = np.array([1.0, 0.0, 2.0, -1.0])
    lam2 = np.array([0.5, 1.0, -0.5, 0.0])
    g1 = small_payoffs.g_lambda(0, lam1, x)
    g2 = small_payoffs.g_lambda(0, lam2, x)
    g_sum = small_payoffs.g_lambda(0, lam1 + lam2, x)
    assert np.allclose(g1 + g2, g_sum)


def test_terminal_derivative_matches_payoff_only_for_own_maturity(small_payoffs):
    x = np.linspace(-2, 2, 30)
    for p in range(small_payoffs.P):
        for i in range(small_payoffs.MP):
            d = small_payoffs.terminal_derivative(p, i, x)
            if i in small_payoffs.flat_index[p]:
                assert np.allclose(d, small_payoffs.payoff_value(i, x))
            else:
                assert np.allclose(d, 0.0)
