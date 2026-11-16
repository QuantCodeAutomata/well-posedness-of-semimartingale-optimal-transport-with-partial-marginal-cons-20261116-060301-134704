"""Experiment 2: strong duality V(pi) = D_reg(pi) = D_HJB(pi) (eq. 25, p.15).

Independently computes:

  * the HJB dual value D_HJB(pi) = lambda*.pi - Phi_0(lambda*), obtained by
    the finite-dimensional dual optimisation of Section 3.1 (src.calibration);
  * the Eulerian primal cost V_primal = integral F(t,x,alpha,beta) rho_t(dx) dt
    (eq. 11), obtained by simulating the *forward* Fokker-Planck equation
    (eq. 12) with the optimal feedback coefficients recovered from the HJB
    dual optimum via the optimality conditions (32)-(33) and Remark 3.1 (p.31).

The regular dual value D_reg(pi) is not computed by an independent PDE
solver: Proposition 2.3 (p.15) proves D_reg(pi) = D_HJB(pi) by identifying
the HJB cascade solution itself as a regular admissible potential (u, lambda)
of (14)-(16) under Assumption 4 (the comparison-principle argument in the
proof of Proposition 2.3 shows u_p := phi_p^lambda is itself admissible).
Hence D_reg(pi) is reported as identical to D_HJB(pi) by this identification,
and the genuine numerical check performed here is the agreement between the
forward (primal, Fokker-Planck) and backward (dual, HJB cascade) value
computations, together with the recovery of the calibration constraints
rho_{tau_p}(G_i) = pi_i (eq. 78) both from the Fokker-Planck density and from
an independent Monte Carlo simulation of the feedback SDE (eq. 74).

Run as ``python -m exp.exp_2``.
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
    FP_N_T_PER_UNIT_TIME,
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
from src.fokker_planck import price_from_density, primal_cost, solve_forward_fp
from src.hjb_solver import Grid
from src.payoffs import PayoffFamily

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main(full: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    payoffs = PayoffFamily(MATURITIES, STRIKES_PER_MATURITY)
    grid = Grid()
    n_paths = MC_N_PATHS_PAPER if full else MC_N_PATHS_RUN

    pi = reference_price_vector(payoffs, X0, REFERENCE_VOL)
    print(f"[exp_2] reference (Black-Scholes, vol={REFERENCE_VOL}) price vector pi = {pi}")

    t0 = time.time()
    calib = calibrate(pi, payoffs, X0, grid)
    assert calib.converged, "dual optimisation (eq. 28) did not converge"
    d_hjb = calib.value
    d_reg = d_hjb  # Proposition 2.3: the cascade solution is itself a regular admissible potential
    print(f"[exp_2] dual optimiser lambda* = {calib.lam_star}")
    print(f"[exp_2] D_HJB(pi) = D_reg(pi) = {d_hjb:.8f} (Prop. 2.3 identification, eq. 25)")

    field = FeedbackField(calib.cascade)
    t_levels, rho_hist = solve_forward_fp(field, X0, MATURITIES[-1], grid, FP_N_T_PER_UNIT_TIME)
    v_primal = primal_cost(field, t_levels, rho_hist, grid)
    print(f"[exp_2] Eulerian primal cost V_primal (forward Fokker-Planck, eq. 11) = {v_primal:.8f}")

    rel_gap = abs(v_primal - d_hjb) / max(abs(d_hjb), 1e-8)
    print(f"[exp_2] |V_primal - D_HJB| / |D_HJB| = {rel_gap:.3e}  (strong duality check, eq. 25)")

    # Calibration constraints rho_{tau_p}(G_i) = pi_i (eq. 78), checked from the Fokker-Planck density.
    fp_prices = np.zeros(payoffs.MP)
    for p in range(payoffs.P):
        tau_p = payoffs.maturities[p]
        n_idx = int(np.argmin(np.abs(t_levels - tau_p)))
        for i in payoffs.flat_index[p]:
            fp_prices[i] = price_from_density(payoffs, i, rho_hist[n_idx], grid)
    fp_calibration_error = np.max(np.abs(fp_prices - pi))
    print(f"[exp_2] max |Fokker-Planck price - target pi| = {fp_calibration_error:.3e}")

    # Independent Monte Carlo recovery of the calibration constraints (eq. 78), via (74).
    mc_results = simulate_paths(field, X0, MATURITIES, n_paths, MC_N_STEPS, rng)
    mc_prices = np.zeros(payoffs.MP)
    for p in range(payoffs.P):
        tau_p = payoffs.maturities[p]
        x_tau = mc_results[tau_p]
        for i in payoffs.flat_index[p]:
            mc_prices[i] = float(np.mean(payoffs.payoff_value(i, x_tau)))
    mc_calibration_error = np.max(np.abs(mc_prices - pi))
    mc_std_err = float(np.max([np.std(payoffs.payoff_value(i, mc_results[payoffs.maturities[p]])) / np.sqrt(n_paths)
                                for p in range(payoffs.P) for i in payoffs.flat_index[p]]))
    print(f"[exp_2] max |Monte Carlo price - target pi| = {mc_calibration_error:.3e} "
          f"(MC std. err. approx {mc_std_err:.3e}, n_paths={n_paths})")

    elapsed = time.time() - t0

    # Plots: Fokker-Planck densities at each maturity, and calibrated vs target prices.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for p in range(payoffs.P):
        tau_p = payoffs.maturities[p]
        n_idx = int(np.argmin(np.abs(t_levels - tau_p)))
        axes[0].plot(grid.x, rho_hist[n_idx], label=f"$\\rho_{{\\tau_{p+1}}}$ (t={tau_p})")
    axes[0].plot(grid.x, rho_hist[0], "k--", label="$\\rho_0=\\delta_{x_0}$ (smoothed)", alpha=0.5)
    axes[0].set_xlabel("x (log-price)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Forward Fokker-Planck densities (eq. 12)")
    axes[0].legend(fontsize=8)

    idx = np.arange(payoffs.MP)
    width = 0.25
    axes[1].bar(idx - width, pi, width, label="target $\\pi$")
    axes[1].bar(idx, fp_prices, width, label="Fokker-Planck $\\rho_{\\tau_p}(G_i)$")
    axes[1].bar(idx + width, mc_prices, width, label="Monte Carlo $\\hat{E}[G_i(X_{\\tau_p})]$")
    axes[1].set_xlabel("payoff index $i$")
    axes[1].set_ylabel("price")
    axes[1].set_title("Calibration constraints (eq. 78)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_duality_and_calibration.png", dpi=150)
    plt.close(fig)

    results = {
        "experiment": "exp_2",
        "title": "Strong duality verification: primal and dual value comparison",
        "full_configuration": full,
        "mc_n_paths": n_paths,
        "reference_vol": REFERENCE_VOL,
        "target_price_vector": pi.tolist(),
        "lambda_star": calib.lam_star.tolist(),
        "D_HJB": d_hjb,
        "D_reg": d_reg,
        "V_primal_fokker_planck": v_primal,
        "relative_duality_gap": rel_gap,
        "fokker_planck_prices": fp_prices.tolist(),
        "fokker_planck_calibration_max_error": float(fp_calibration_error),
        "monte_carlo_prices": mc_prices.tolist(),
        "monte_carlo_calibration_max_error": float(mc_calibration_error),
        "monte_carlo_std_error_estimate": mc_std_err,
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "exp2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp_2] done in {elapsed:.1f}s; results saved to results/exp2_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strong duality verification experiment")
    parser.add_argument("--full", action="store_true",
                         help="use the paper-scale Monte Carlo sample size instead of the reduced default")
    args = parser.parse_args()
    main(full=args.full)
