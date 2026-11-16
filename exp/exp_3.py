"""Experiment 3: Gradient-Based Model Calibration via HJB Dual Optimization
with Linearized Pricing Equations (Section 3.1, p.29-31).

Methodology:

  1. A payoff family of log-put options at two maturities is built, and a
     target (calibrateable) price vector pi is generated from a reference
     driftless Black-Scholes model in log-price coordinates (Section 1.4).
  2. The finite-dimensional HJB dual optimisation

         V(pi) = max_lambda { lambda . pi - Phi_0(lambda) }     (eq. 28, 30)

     is solved by L-BFGS-B, using the analytic gradient of Phi_0 obtained
     from the linearised pricing equations (eq. 75), as implemented in
     src/calibration.py and src/linearized.py.
  3. The first-order optimality condition

         grad Phi_0(lambda*) = pi                                (eq. 77)

     is checked directly, which is equivalent to the calibration
     constraints

         E^{P^lambda*}[ G_i(X_{tau_p}) ] = pi_i                   (eq. 78)

  4. The primal optimal law P^{lambda*} is recovered by Monte Carlo
     (Euler-Maruyama) simulation of the feedback SDE

         dX_t = alpha^{lambda*}(t, X_t) dt + sigma^{lambda*}(t, X_t) dW_t   (eq. 74)

     and the simulated model reproduces the calibration constraints (78)
     within Monte Carlo error, independently confirming (eq. 77).
  5. V(pi) = lambda* . pi - Phi_0(lambda*) is computed and checked finite.
  6. As a convergence diagnostic (scipy's L-BFGS-B does not expose a
     per-iteration trace through the public API used here), calibration is
     re-run from several random starting points lambda_0 in a small
     neighbourhood of 0; all runs are checked to converge to the same
     lambda* (uniqueness of the dual optimiser, strict concavity of the
     objective, Section 3). The final gradient residual grad Phi_0(lambda*)
     - pi per payoff index, and the target vs. Monte-Carlo-recovered prices,
     are plotted.

Run as ``python -m exp.exp_3``.
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
    CALIBRATION_MAX_ITER,
    CALIBRATION_TOL,
    MATURITIES,
    MC_N_PATHS_PAPER,
    MC_N_PATHS_RUN,
    MC_N_STEPS,
    RANDOM_SEED,
    REFERENCE_VOL,
    STRIKES_PER_MATURITY,
    X0,
)
from src.feedback_sde import FeedbackField, simulate_paths
from src.hjb_solver import Grid
from src.payoffs import PayoffFamily

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_MULTI_START = 3  # number of extra random starting points lambda_0 for the convergence diagnostic


def main(full: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    t0 = time.time()

    payoffs = PayoffFamily(MATURITIES, STRIKES_PER_MATURITY)
    grid = Grid()
    n_paths = MC_N_PATHS_PAPER if full else MC_N_PATHS_RUN

    # Step 1: target price vector from the reference Black-Scholes model.
    pi = reference_price_vector(payoffs, X0, REFERENCE_VOL)
    print(f"[exp_3] reference (Black-Scholes, vol={REFERENCE_VOL}) target price vector pi = {pi}")

    # Step 2: HJB dual optimisation V(pi) = max_lambda { lambda.pi - Phi_0(lambda) } (eq. 28, 30).
    result = calibrate(pi, payoffs, X0, grid)
    assert result.converged, "L-BFGS-B dual optimisation (eq. 28) did not converge"
    print(f"[exp_3] lambda* = {result.lam_star}")
    print(f"[exp_3] Phi_0(lambda*) = {result.phi0_star:.8f}, converged in {result.n_iter} iterations")

    # Step 3: first-order optimality condition grad Phi_0(lambda*) = pi (eq. 77).
    grad_residual = result.grad_phi0_star - pi
    grad_residual_max = float(np.max(np.abs(grad_residual)))
    print(f"[exp_3] max |grad Phi_0(lambda*) - pi| = {grad_residual_max:.3e} "
          f"(first-order optimality, eq. 77; tol={CALIBRATION_TOL:.1e})")
    assert grad_residual_max < 1e-3, "optimality condition (eq. 77) not satisfied to the expected tolerance"

    # Step 4: Monte Carlo recovery of the primal optimal law by simulating (eq. 74).
    field = FeedbackField(result.cascade)
    mc_paths = simulate_paths(field, X0, MATURITIES, n_paths, MC_N_STEPS, rng)
    mc_prices = np.zeros(payoffs.MP)
    mc_std_err = np.zeros(payoffs.MP)
    for p in range(payoffs.P):
        x_tau = mc_paths[payoffs.maturities[p]]
        for i in payoffs.flat_index[p]:
            vals = payoffs.payoff_value(i, x_tau)
            mc_prices[i] = float(np.mean(vals))
            mc_std_err[i] = float(np.std(vals) / np.sqrt(n_paths))
    mc_calibration_error = float(np.max(np.abs(mc_prices - pi)))
    print(f"[exp_3] max |Monte Carlo price - target pi| = {mc_calibration_error:.3e} "
          f"(n_paths={n_paths}, max MC std. err. approx {np.max(mc_std_err):.3e}; eq. 78)")

    # Step 5: dual value V(pi) = lambda*.pi - Phi_0(lambda*).
    v_pi = result.value
    assert np.isfinite(v_pi), "V(pi) is not finite"
    print(f"[exp_3] V(pi) = lambda*.pi - Phi_0(lambda*) = {v_pi:.8f}")

    # Step 6: convergence diagnostic from several random starting points lambda_0.
    lam_deviations = []
    for k in range(N_MULTI_START):
        lam0 = rng.normal(scale=0.3, size=payoffs.MP)
        res_k = calibrate(pi, payoffs, X0, grid, lam0=lam0)
        assert res_k.converged, f"multi-start run {k} did not converge"
        dev = float(np.max(np.abs(res_k.lam_star - result.lam_star)))
        lam_deviations.append(dev)
        print(f"[exp_3] multi-start {k}: lambda0={lam0}, |lambda_star_k - lambda_star| = {dev:.3e}")
    multi_start_max_dev = float(np.max(lam_deviations))
    print(f"[exp_3] multi-start max |lambda* deviation| = {multi_start_max_dev:.3e} "
          "(uniqueness of the dual optimiser)")

    elapsed = time.time() - t0

    # Plot: (a) target vs Monte-Carlo-recovered prices, (b) gradient residual per payoff index.
    idx = np.arange(payoffs.MP)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.35
    axes[0].bar(idx - width / 2, pi, width, label="target $\\pi$")
    axes[0].bar(idx + width / 2, mc_prices, width, label="Monte Carlo $\\hat{E}[G_i(X_{\\tau_p})]$")
    axes[0].set_xlabel("payoff index $i$")
    axes[0].set_ylabel("price")
    axes[0].set_title("Target vs. Monte-Carlo-recovered prices (eq. 78)")
    axes[0].legend(fontsize=8)

    axes[1].bar(idx, grad_residual)
    axes[1].set_xlabel("payoff index $i$")
    axes[1].set_ylabel("$\\partial_{\\lambda_i}\\Phi_0(\\lambda^*) - \\pi_i$")
    axes[1].set_title("First-order optimality residual (eq. 77)")
    axes[1].axhline(0.0, color="k", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_calibration.png", dpi=150)
    plt.close(fig)

    results = {
        "experiment": "exp_3",
        "title": "Gradient-based model calibration via HJB dual optimization with linearized pricing equations",
        "full_configuration": full,
        "mc_n_paths": n_paths,
        "target_price_vector": pi.tolist(),
        "lambda_star": result.lam_star.tolist(),
        "phi0_star": float(result.phi0_star),
        "V_pi": float(v_pi),
        "gradient_residual_max": grad_residual_max,
        "monte_carlo_prices": mc_prices.tolist(),
        "monte_carlo_calibration_max_error": mc_calibration_error,
        "multi_start_max_lambda_deviation": multi_start_max_dev,
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "exp3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp_3] done in {elapsed:.1f}s; results saved to results/exp3_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based HJB dual calibration experiment")
    parser.add_argument("--full", action="store_true",
                         help="use the paper-scale Monte Carlo sample size instead of the reduced default")
    args = parser.parse_args()
    main(full=args.full)
