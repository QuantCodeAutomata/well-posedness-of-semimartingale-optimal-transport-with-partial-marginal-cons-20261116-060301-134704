"""Synthetic calibration data: reference Black-Scholes log-put prices.

No external market data is used by any of the five experiments (all data
requirements in the paper's Sections 1.4, 2, and 3 are synthetic: a payoff
family, maturities, an initial state, and a reference price vector used only
to produce a calibrateable (feasible) target). The reference price vector pi
is generated from a driftless Black-Scholes model in log-price coordinates,
i.e. the martingale measure for S_t = exp(X_t) with volatility
REFERENCE_VOL, exactly the model class referred to in the methodology of
exp_3 ("Generate prices from a known reference model (e.g., Black-Scholes
with volatility sigma_ref in log-price coordinates)").

Closed-form Black-Scholes put pricing (Hull, or any standard reference) is
used instead of Monte Carlo simulation for speed and because it is exact for
this constant-volatility reference model; forward = S0 = exp(x0), no discount
rate (Section 1.4 fixes forward prices and discount factors).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.payoffs import PayoffFamily


def bs_put_price(forward: float, strike: float, vol: float, maturity: float) -> float:
    """Undiscounted Black-Scholes put price E[(K - F*exp(sigma W_t - sigma^2 t/2))^+]."""
    if maturity <= 0:
        return max(strike - forward, 0.0)
    d1 = (np.log(forward / strike) + 0.5 * vol ** 2 * maturity) / (vol * np.sqrt(maturity))
    d2 = d1 - vol * np.sqrt(maturity)
    return float(strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def reference_price_vector(payoffs: PayoffFamily, x0: float, vol: float) -> np.ndarray:
    """pi_i = E[(K_i - exp(X_tau_p))^+] under the driftless log-Brownian reference model."""
    forward = np.exp(x0)
    pi = np.zeros(payoffs.MP)
    for p in range(payoffs.P):
        maturity = payoffs.maturities[p]
        for i in payoffs.flat_index[p]:
            _, strike = payoffs.global_to_pk[i]
            pi[i] = bs_put_price(forward, strike, vol, maturity)
    return pi
