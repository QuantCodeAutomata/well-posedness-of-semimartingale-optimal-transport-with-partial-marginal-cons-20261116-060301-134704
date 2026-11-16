"""Shared pytest fixtures. No experiment in this repository uses market data;
all data is synthetic (payoff family, maturities, reference Black-Scholes
prices), so no market_data marker or live-API fixture is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.hjb_solver import Grid
from src.payoffs import PayoffFamily


@pytest.fixture
def small_grid() -> Grid:
    """A coarse, fast grid used throughout the unit tests."""
    return Grid(L=2.0, n_x=41)


@pytest.fixture
def small_payoffs() -> PayoffFamily:
    """A minimal two-maturity log-put payoff family."""
    return PayoffFamily(maturities=(0.5, 1.0), strikes_per_maturity=((0.95, 1.05), (0.90, 1.10)))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
