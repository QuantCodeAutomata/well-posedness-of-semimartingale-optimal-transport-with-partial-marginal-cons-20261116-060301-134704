"""Experiment 4: strong convexity, nonempty interior of P, and stability of lambda*(pi).

Verifies, numerically, the three results of Section 3.2 (pages 31-34):

  * Theorem 3.1 (p.33, local strong convexity of Phi_0, eq. 80): for every
    compact convex set K there is a modulus m_K > 0 such that

        Phi_0((1-theta)l1 + theta l2)
            <= (1-theta)Phi_0(l1) + theta Phi_0(l2) - (m_K/2) theta(1-theta)|l2-l1|^2

    for all lambda_1, lambda_2 in K and theta in [0,1]. This is checked by
    Monte Carlo sampling of (lambda_1, lambda_2, theta) triples over a
    compact box K and estimating m_K as the minimum of the (rearranged)
    normalised defect; m_K > 0 corroborates the theorem.

  * Corollary 3.5 (p.34, nonempty interior of the admissible price set P):
    a reference calibrateable price vector pi0, generated from a driftless
    Black-Scholes reference model, is shown to admit a finite, converged
    dual optimiser, and so do small random perturbations of pi0. This is
    consistent with pi0 lying in the interior of P.

  * Proposition 3.6 (p.34, uniqueness and local Lipschitz stability of
    lambda*(pi), and the C^1 envelope identity grad V(pi0) = lambda*(pi0),
    eq. 31): the dual optimiser is re-computed from several random starting
    points to check that it is unique at pi0; pi0 is perturbed in several
    random directions to estimate the local Lipschitz constant of the map
    pi -> lambda*(pi); and a finite-difference approximation of the value
    function V = Phi_0^* is compared against lambda*(pi0).

Run as ``python -m exp.exp_4``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import reference_price_vector
from src.calibration import calibrate
from src.config import (
    MATURITIES,
    PRICE_N_DIRECTIONS,
    PRICE_PERTURBATION_SCALE,
    RANDOM_SEED,
    REFERENCE_VOL,
    STRIKES_PER_MATURITY,
    STRONG_CONVEXITY_BOX,
    STRONG_CONVEXITY_N_SAMPLES_PAPER,
    STRONG_CONVEXITY_N_SAMPLES_RUN,
    UNIQUENESS_N_STARTS,
    VALUE_GRADIENT_FD_STEP,
    X0,
)
from src.hjb_solver import Grid
from src.payoffs import PayoffFamily
from src.stability import (
    check_uniqueness,
    convexity_defect_modulus,
    estimate_lipschitz_constant,
    value_function_gradient_check,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def plot_stability(defects: np.ndarray, m_K: float, pi_distances: np.ndarray,
                    lambda_distances: np.ndarray, estimated_constant: float, path: Path) -> None:
    """(a) histogram of the convexity-defect ratios with the m_K line; (b) Lipschitz scatter."""
    finite_defects = defects[np.isfinite(defects)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(finite_defects, bins=20, color="C0", alpha=0.8)
    axes[0].axvline(m_K, color="k", linestyle="--", label=f"$m_K$ = {m_K:.3e}")
    axes[0].set_xlabel("normalised convexity defect")
    axes[0].set_ylabel("count")
    axes[0].set_title("Local strong convexity modulus $m_K$ (Thm. 3.1, eq. 80)")
    axes[0].legend()

    axes[1].scatter(pi_distances, lambda_distances, color="C1", label="sampled directions")
    x_line = np.array([0.0, float(np.max(pi_distances))])
    axes[1].plot(x_line, estimated_constant * x_line, "k--",
                 label=f"slope = {estimated_constant:.3f}")
    axes[1].set_xlabel(r"$|\pi_{\mathrm{pert}} - \pi_0|$")
    axes[1].set_ylabel(r"$|\lambda^*(\pi_{\mathrm{pert}}) - \lambda^*(\pi_0)|$")
    axes[1].set_title("Local Lipschitz stability of $\\lambda^*(\\pi)$ (Prop. 3.6)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(full: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payoffs = PayoffFamily(MATURITIES, STRIKES_PER_MATURITY)
    grid = Grid()
    rng = np.random.default_rng(RANDOM_SEED)
    pi0 = reference_price_vector(payoffs, X0, REFERENCE_VOL)

    # Each calibrate() call costs about 14s on this grid (an L-BFGS-B run of
    # ~14 cascade+gradient solves), and convexity_defect_modulus costs about
    # 1.3s/sample (3 cascade solves/sample). With UNIQUENESS_N_STARTS=5 and
    # PRICE_N_DIRECTIONS=8 kept at their full scale, the reduced
    # STRONG_CONVEXITY_N_SAMPLES_RUN=40 default keeps the whole experiment
    # under about 6 minutes, so no further reduction of n_samples is applied
    # here; `--full` restores STRONG_CONVEXITY_N_SAMPLES_PAPER.
    n_samples = STRONG_CONVEXITY_N_SAMPLES_PAPER if full else STRONG_CONVEXITY_N_SAMPLES_RUN

    print(f"[exp_4] payoff family: P={payoffs.P} maturities, MP={payoffs.MP} payoffs total")
    print(f"[exp_4] reference (Black-Scholes, vol={REFERENCE_VOL}) price vector pi0 = {pi0}")
    print(f"[exp_4] strong-convexity sampling: n_samples={n_samples} (full={full})")

    t0 = time.time()

    # 1. Local strong convexity of Phi_0 on a compact box K (Theorem 3.1, eq. 80).
    m_K, defects = convexity_defect_modulus(payoffs, X0, grid, STRONG_CONVEXITY_BOX, n_samples, rng)
    strong_convexity_positive = bool(m_K > 0)
    print(f"[exp_4] strong convexity modulus m_K = {m_K:.6e} (m_K > 0: {strong_convexity_positive})")

    # 2. Nonempty interior of P: the dual optimiser exists, is finite and converges at pi0.
    calib0 = calibrate(pi0, payoffs, X0, grid)
    assert calib0.converged, "dual optimisation (eq. 28) did not converge at pi0"
    assert np.isfinite(calib0.value), "V(pi0) must be finite for pi0 to be in the interior of P"
    print(f"[exp_4] calibration at pi0 converged: lambda*(pi0) = {calib0.lam_star}, "
          f"V(pi0) = {calib0.value:.6f} (Cor. 3.5: pi0 in int(P))")

    # 3. Uniqueness of the dual optimiser lambda*(pi0) from several random starting points.
    uniqueness = check_uniqueness(pi0, payoffs, X0, grid, UNIQUENESS_N_STARTS, rng)
    print(f"[exp_4] uniqueness check: max pairwise distance = "
          f"{uniqueness.max_pairwise_distance:.3e}, unique = {uniqueness.unique}")

    # 4. Local Lipschitz stability of pi -> lambda*(pi) (Proposition 3.6).
    lipschitz = estimate_lipschitz_constant(pi0, calib0.lam_star, payoffs, X0, grid,
                                             PRICE_PERTURBATION_SCALE, PRICE_N_DIRECTIONS, rng)
    print(f"[exp_4] Lipschitz check: estimated constant = {lipschitz.estimated_constant:.4f}, "
          f"ratios = {np.round(lipschitz.ratios, 4)}")

    # 5. Envelope / C^1 property grad V(pi0) = lambda*(pi0) (eq. 31).
    grad_v, lam_star = value_function_gradient_check(pi0, calib0.lam_star, payoffs, X0, grid,
                                                       VALUE_GRADIENT_FD_STEP)
    grad_error = float(np.max(np.abs(grad_v - lam_star)))
    print(f"[exp_4] envelope check: max |grad V(pi0) - lambda*(pi0)| = {grad_error:.3e}")

    elapsed = time.time() - t0

    plot_stability(defects, m_K, lipschitz.pi_distances, lipschitz.lambda_distances,
                    lipschitz.estimated_constant, RESULTS_DIR / "exp4_stability.png")

    results = {
        "experiment": "exp_4",
        "title": "Uniqueness and local Lipschitz stability of dual optimizers: "
                  "strong convexity and nonempty interior of the admissible price set",
        "full_configuration": full,
        "n_convexity_samples": n_samples,
        "pi0": pi0.tolist(),
        "lambda_star": calib0.lam_star.tolist(),
        "strong_convexity_modulus_mK": m_K,
        "strong_convexity_positive": strong_convexity_positive,
        "uniqueness_max_pairwise_distance": uniqueness.max_pairwise_distance,
        "uniqueness_confirmed": uniqueness.unique,
        "lipschitz_estimated_constant": lipschitz.estimated_constant,
        "lipschitz_ratios": lipschitz.ratios.tolist(),
        "grad_V_vs_lambda_star_max_error": grad_error,
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "exp4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp_4] done in {elapsed:.1f}s; results saved to results/exp4_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strong convexity and stability of dual optimizers experiment")
    parser.add_argument("--full", action="store_true",
                         help="use the paper-scale number of convexity-defect samples instead of the reduced default")
    args = parser.parse_args()
    main(full=args.full)
