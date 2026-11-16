"""Minimal smoke tests for the exp_4 stability checks (src/stability.py)."""

from __future__ import annotations

import numpy as np

from src.config import STRONG_CONVEXITY_BOX, X0
from src.stability import UniquenessCheck, check_uniqueness, convexity_defect_modulus
from data.generate_synthetic import reference_price_vector


def test_convexity_defect_modulus_finite(small_payoffs, small_grid, rng):
    m_K, defects = convexity_defect_modulus(small_payoffs, X0, small_grid, STRONG_CONVEXITY_BOX, 5, rng)
    assert np.isfinite(m_K)
    assert defects.shape == (5,)


def test_check_uniqueness_shape(small_payoffs, small_grid, rng):
    pi0 = reference_price_vector(small_payoffs, X0, 0.22)
    result = check_uniqueness(pi0, small_payoffs, X0, small_grid, 2, rng)
    assert isinstance(result, UniquenessCheck)
    assert result.lam_stars.shape == (2, small_payoffs.MP)
    assert np.isfinite(result.max_pairwise_distance)
