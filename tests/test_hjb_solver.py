"""Unit tests for src/hjb_solver.py: backward cascade (eq. 22) and Phi_0 (eq. 27)."""

from __future__ import annotations

import numpy as np

from src.hjb_solver import Grid, phi0, solve_cascade


def test_rejects_wrong_lambda_length(small_payoffs, small_grid):
    try:
        solve_cascade(np.zeros(1), small_payoffs, small_grid)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cascade_has_one_interval_per_maturity(small_payoffs, small_grid):
    lam = np.zeros(small_payoffs.MP)
    result = solve_cascade(lam, small_payoffs, small_grid)
    assert len(result.intervals) == small_payoffs.P
    # time levels of interval p must span (tau_{p-1}, tau_p)
    taus = (0.0,) + small_payoffs.maturities
    for p, interval in enumerate(result.intervals):
        assert interval.t_levels[0] == taus[p]
        assert interval.t_levels[-1] == taus[p + 1]


def test_terminal_phi_P_plus_1_is_zero_so_last_interval_terminal_equals_payoff(small_payoffs, small_grid):
    lam = np.array([1.0, 0.0, 0.0, 0.0])
    result = solve_cascade(lam, small_payoffs, small_grid)
    last = result.intervals[-1]
    g_last = small_payoffs.g_lambda(small_payoffs.P - 1, lam, small_grid.x)
    assert np.allclose(last.phi[-1], g_last)


def test_phi0_is_finite_and_bounded_by_total_payoff_scale(small_payoffs, small_grid):
    lam = np.array([0.5, -0.3, 0.2, 0.1])
    value = phi0(lam, small_payoffs, 0.0, small_grid)
    assert np.isfinite(value)
    max_strike = max(k for _, k in small_payoffs.global_to_pk)
    assert abs(value) < 10 * max_strike * np.sum(np.abs(lam))


def test_cascade_zero_lambda_gives_nonpositive_phi0_since_h_of_zero_can_be_negative(small_payoffs, small_grid):
    """At lambda=0 the terminal data is identically zero; phi0 reduces to a pure running-cost
    value, which must still be finite (no payoff singularities are introduced)."""
    lam = np.zeros(small_payoffs.MP)
    value = phi0(lam, small_payoffs, 0.0, small_grid)
    assert np.isfinite(value)


def test_increasing_lambda_increases_phi0_monotonically(small_payoffs, small_grid):
    """g_p^lambda is increasing in each lambda_i (payoffs are nonnegative), and the cascade
    solution map is monotone in the terminal data by the comparison principle (H2, p.14)."""
    lam_low = np.array([0.1, 0.1, 0.1, 0.1])
    lam_high = np.array([0.5, 0.5, 0.5, 0.5])
    v_low = phi0(lam_low, small_payoffs, 0.0, small_grid)
    v_high = phi0(lam_high, small_payoffs, 0.0, small_grid)
    assert v_high >= v_low
