"""Log-put payoff family and maturity/strike bookkeeping (Section 1.4, p.10-12).

The payoffs G_{i(p,m)}(x) = (K_{i(p,m)} - exp(x))^+ are bounded and uniformly
continuous on R (BUC(R)) despite the kink at x = log(K): as x -> -infty the
payoff tends to the constant K, and as x -> +infty it tends to 0. Distinct
positive strikes per maturity give Assumption 2 (linear independence modulo
constants) for the index family used in Section 3.
"""

from __future__ import annotations

import numpy as np


class PayoffFamily:
    """Flattened index set i(p, m) <-> (maturity index p, strike index m)."""

    def __init__(self, maturities: tuple[float, ...], strikes_per_maturity: tuple[tuple[float, ...], ...]):
        if len(maturities) != len(strikes_per_maturity):
            raise ValueError("maturities and strikes_per_maturity must have the same length P")
        for strikes in strikes_per_maturity:
            if len(strikes) != len(set(strikes)):
                raise ValueError("strikes within one maturity must be distinct (Assumption 2, p.10)")
            if any(k <= 0 for k in strikes):
                raise ValueError("strikes must be strictly positive")
        self.maturities = tuple(float(t) for t in maturities)
        self.strikes_per_maturity = tuple(tuple(float(k) for k in ks) for ks in strikes_per_maturity)
        self.P = len(maturities)
        self.M_p = tuple(len(ks) for ks in self.strikes_per_maturity)
        self.MP = sum(self.M_p)
        # flat_index[p] = list of global indices i(p, m) for m = 0, ..., M_p-1
        self.flat_index: list[list[int]] = []
        counter = 0
        for m_p in self.M_p:
            self.flat_index.append(list(range(counter, counter + m_p)))
            counter += m_p
        # global index -> (p, strike)
        self.global_to_pk: list[tuple[int, float]] = []
        for p, strikes in enumerate(self.strikes_per_maturity):
            for k in strikes:
                self.global_to_pk.append((p, k))

    def payoff_value(self, global_index: int, x: np.ndarray) -> np.ndarray:
        """G_i(x) = (K_i - exp(x))^+ for the payoff with flat global index i."""
        _, k = self.global_to_pk[global_index]
        return np.maximum(k - np.exp(x), 0.0)

    def g_lambda(self, p: int, lam: np.ndarray, x: np.ndarray) -> np.ndarray:
        """g_p^lambda(x) = sum_m lambda_{i(p,m)} G_{i(p,m)}(x), the terminal increment at tau_p."""
        total = np.zeros_like(x, dtype=float)
        for i in self.flat_index[p]:
            total = total + lam[i] * self.payoff_value(i, x)
        return total

    def terminal_derivative(self, p: int, i: int, x: np.ndarray) -> np.ndarray:
        """d g_p^lambda / d lambda_i = G_i(x) if i indexes maturity p, else 0 (used in eq. 75)."""
        if i in self.flat_index[p]:
            return self.payoff_value(i, x)
        return np.zeros_like(x, dtype=float)
