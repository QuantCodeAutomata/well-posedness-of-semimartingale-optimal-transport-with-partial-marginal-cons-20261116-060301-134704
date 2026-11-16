"""Minimal sanity check for the AW2 common-noise coupling used by exp_5."""

from __future__ import annotations

import numpy as np

from data.generate_synthetic import reference_price_vector
from src.calibration import calibrate
from src.feedback_sde import FeedbackField
from src.payoffs import PayoffFamily
from src.hjb_solver import Grid
from src.wasserstein import common_noise_aw2_upper_bound


def test_common_noise_aw2_upper_bound_is_finite_and_nonnegative(small_payoffs: PayoffFamily, small_grid: Grid,
                                                                 rng: np.random.Generator) -> None:
    x0 = 0.0
    pi = reference_price_vector(small_payoffs, x0, vol=0.22)
    calib = calibrate(pi, small_payoffs, x0, small_grid)
    assert calib.converged
    field = FeedbackField(calib.cascade)

    pi_perturbed = pi + 0.001 * np.array([1.0, -1.0, 0.5, -0.5])
    calib_n = calibrate(pi_perturbed, small_payoffs, x0, small_grid, lam0=calib.lam_star)
    assert calib_n.converged
    field_n = FeedbackField(calib_n.cascade)

    estimate = common_noise_aw2_upper_bound(field_n, field, x0, small_payoffs.maturities[-1],
                                             n_steps_per_unit_time=50, n_paths=200, rng=rng)

    assert np.isfinite(estimate.aw2_upper_bound_sq)
    assert np.isfinite(estimate.trajectory_sup_sq)
    assert estimate.aw2_upper_bound_sq >= 0.0
    assert estimate.trajectory_sup_sq >= 0.0
    assert estimate.n_paths == 200
