"""Experiment 5: continuity of optimal calibrated laws in AW2 (Section 3.3, p.36-39).

Definition 3.1 (p.36) introduces bicausal couplings between two feedback-SDE
laws, and Definition 3.2 (eq. 87, p.36) defines the adapted Wasserstein
distance AW2 as the infimum, over bicausal couplings, of the usual L2
transportation cost on path space. Theorem 3.2 (p.36-39) proves that the map
pi -> P^{*,pi} (the optimal calibrated law of the feedback SDE (74) built
from the dual optimiser lambda*(pi)) is continuous for AW2: pi_n -> pi in the
admissible price set P implies AW2(P^{*,pi_n}, P^{*,pi}) -> 0. The proof
chains three facts: (i) the dual optimiser lambda*(.) is locally Lipschitz
on int(P) (Proposition 3.1 / exp_4), so pi_n -> pi implies lambda_n -> lambda*;
(ii) the HJB cascade and its feedback coefficients (alpha^lambda, beta^lambda)
depend continuously on lambda (eq. 86, p.35); (iii) the common-noise coupling
of eq. (89), driving both feedback SDEs with the same Brownian motion and
deterministic initial condition x0, gives the explicit upper bounds

    E[sup_{t<=T} |X_t^n - X_t|^2] -> 0                                  (eq. 96, p.39)
    AW2^2(P^{*,pi_n}, P^{*,pi}) <=
        E[int_0^T ||sigma^{lambda_n} - sigma^lambda||^2 dt]
        + E[(int_0^T |alpha^{lambda_n} - alpha^lambda| dt)^2] -> 0      (eq. 97, p.39)

by dominated convergence, using the convergence of the feedback coefficients
established in (ii). Remark 3.6 (p.39) stresses that this argument is
qualitative only: no quantitative convergence rate for AW2 in terms of
|pi_n - pi| is established by the paper (the Lipschitz constant of
lambda*(.) does not, by itself, give a modulus of continuity for the
nonlinear cascade-to-coefficient map). This experiment numerically verifies
the qualitative statement: it builds a geometrically decreasing sequence
epsilon_n -> 0 of perturbation sizes for a fixed random direction in price
space, computes lambda_n = lambda*(pi + epsilon_n direction) by the dual
optimisation of src/calibration.py, builds the associated feedback fields,
and uses the common-noise coupling already implemented in src/wasserstein.py
to estimate both quantities in (96)-(97) via Monte Carlo, alongside the
Lipschitz-stability quantity |lambda_n - lambda*| studied in exp_4.

Run as ``python -m exp.exp_5``.
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
    AW2_MC_PATHS_PAPER,
    AW2_MC_PATHS_RUN,
    AW2_N_PERTURBATIONS_PAPER,
    AW2_N_PERTURBATIONS_RUN,
    AW2_N_STEPS,
    MATURITIES,
    RANDOM_SEED,
    REFERENCE_VOL,
    STRIKES_PER_MATURITY,
    X0,
)
from src.feedback_sde import FeedbackField
from src.hjb_solver import Grid
from src.payoffs import PayoffFamily
from src.wasserstein import common_noise_aw2_upper_bound

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main(full: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    payoffs = PayoffFamily(MATURITIES, STRIKES_PER_MATURITY)
    grid = Grid()
    rng = np.random.default_rng(RANDOM_SEED)
    n_paths = AW2_MC_PATHS_PAPER if full else AW2_MC_PATHS_RUN
    n_pert = AW2_N_PERTURBATIONS_PAPER if full else AW2_N_PERTURBATIONS_RUN

    pi = reference_price_vector(payoffs, X0, REFERENCE_VOL)
    calib = calibrate(pi, payoffs, X0, grid)
    assert calib.converged, "reference dual optimisation (eq. 28) did not converge"
    field = FeedbackField(calib.cascade)
    print(f"[exp_5] reference price vector pi = {pi}")
    print(f"[exp_5] reference dual optimiser lambda* = {calib.lam_star}")

    direction = rng.normal(size=payoffs.MP)
    direction /= np.linalg.norm(direction)
    # Largest epsilon kept at 40% of the distance to the boundary of the positive
    # orthant along `direction` (a necessary condition for pi_n to remain in the
    # admissible price set P, Section 1.4), so that every pi_n stays calibrateable;
    # the sequence then halves geometrically down to epsilon_n -> 0.
    eps_cap = 0.4 * float(np.min(pi / np.abs(direction)))
    epsilons = eps_cap * (0.5 ** np.arange(n_pert))
    print(f"[exp_5] perturbation direction (unit vector in R^MP) = {direction}")
    print(f"[exp_5] epsilon_n sequence -> 0: {epsilons}")

    lambda_gaps: list[float] = []
    aw2_upper_bound_sq: list[float] = []
    trajectory_sup_sq: list[float] = []

    for eps in epsilons:
        pi_n = pi + eps * direction
        calib_n = calibrate(pi_n, payoffs, X0, grid, lam0=calib.lam_star)
        assert calib_n.converged, f"perturbed dual optimisation did not converge at epsilon={eps}"
        field_n = FeedbackField(calib_n.cascade)
        lambda_gap = float(np.linalg.norm(calib_n.lam_star - calib.lam_star))
        estimate = common_noise_aw2_upper_bound(field_n, field, X0, MATURITIES[-1], AW2_N_STEPS, n_paths, rng)
        lambda_gaps.append(lambda_gap)
        aw2_upper_bound_sq.append(estimate.aw2_upper_bound_sq)
        trajectory_sup_sq.append(estimate.trajectory_sup_sq)
        print(f"[exp_5] epsilon={eps:.5f}  |lambda_n-lambda*|={lambda_gap:.3e}  "
              f"AW2_upper_bound^2={estimate.aw2_upper_bound_sq:.3e}  "
              f"sup|X^n-X|^2={estimate.trajectory_sup_sq:.3e}")

    def is_decreasing(values: list[float]) -> bool:
        return values[-1] < values[0] and values[-1] < 0.25 * values[0]

    decreasing_confirmed = (
        is_decreasing(lambda_gaps) and is_decreasing(aw2_upper_bound_sq) and is_decreasing(trajectory_sup_sq)
    )
    print(f"[exp_5] lambda_gap ratio (last/first) = {lambda_gaps[-1] / lambda_gaps[0]:.3e}")
    print(f"[exp_5] AW2_upper_bound^2 ratio (last/first) = {aw2_upper_bound_sq[-1] / aw2_upper_bound_sq[0]:.3e}")
    print(f"[exp_5] sup|X^n-X|^2 ratio (last/first) = {trajectory_sup_sq[-1] / trajectory_sup_sq[0]:.3e}")
    print(f"[exp_5] decreasing_confirmed (AW2 continuity, Thm 3.2) = {decreasing_confirmed}")

    elapsed = time.time() - t0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].loglog(epsilons, lambda_gaps, "o-")
    axes[0].set_xlabel(r"$\epsilon_n = \|\pi_n - \pi\|$")
    axes[0].set_ylabel(r"$\|\lambda_n - \lambda^*\|$")
    axes[0].set_title("Lipschitz stability of the dual optimiser (exp_4)")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].loglog(epsilons, np.sqrt(aw2_upper_bound_sq), "o-", label=r"$\sqrt{AW2^2}$ upper bound (eq. 97)")
    axes[1].loglog(epsilons, np.sqrt(trajectory_sup_sq), "s-", label=r"$\sqrt{E[\sup_t|X^n_t-X_t|^2]}$ (eq. 96)")
    axes[1].set_xlabel(r"$\epsilon_n = \|\pi_n - \pi\|$")
    axes[1].set_ylabel("distance")
    axes[1].set_title("AW2 continuity of the optimal calibrated law (Thm 3.2)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_aw2_stability.png", dpi=150)
    plt.close(fig)

    results = {
        "experiment": "exp_5",
        "title": "Continuity of optimal calibrated laws in adapted Wasserstein distance (AW2 stability)",
        "full_configuration": full,
        "n_paths": n_paths,
        "epsilons": epsilons.tolist(),
        "lambda_gaps": lambda_gaps,
        "aw2_upper_bound_sq": aw2_upper_bound_sq,
        "trajectory_sup_sq": trajectory_sup_sq,
        "decreasing_confirmed": decreasing_confirmed,
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "exp5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp_5] done in {elapsed:.1f}s; results saved to results/exp5_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AW2 continuity (stability) of optimal calibrated laws")
    parser.add_argument("--full", action="store_true",
                         help="use the paper-scale Monte Carlo sample / perturbation count instead of the reduced default")
    args = parser.parse_args()
    main(full=args.full)
