"""Experiment 1: HJB cascade backward induction and the dual value map Phi_0.

Reproduces the backward induction of the HJB cascade (eq. 22, p.14) across
the P maturity intervals, evaluates the dual value map

    Phi_0(lambda) = phi_1^lambda(0, x0)                                (eq. 27, p.16)

and numerically verifies two properties proved in Section 2.2:

  * Proposition 2.4 (p.16): Phi_0 is finite, convex, and continuous on
    R^{MP}. Convexity is checked through the Jensen-type inequality

        Phi_0(theta*l1 + (1-theta)*l2) <= theta*Phi_0(l1) + (1-theta)*Phi_0(l2).

  * Continuous differentiability of Phi_0: the analytic gradient obtained
    from the linearised pricing equations (eq. 75-76, p.29-30) is compared
    with a central finite-difference approximation of Phi_0.

Run as ``python -m exp.exp_1``.
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

from src.config import (
    CONVEXITY_LAMBDA_BOX,
    CONVEXITY_N_PAIRS_PAPER,
    CONVEXITY_N_PAIRS_RUN,
    CONVEXITY_N_THETA_PAPER,
    CONVEXITY_N_THETA_RUN,
    FD_STEP_GRADIENT,
    MATURITIES,
    RANDOM_SEED,
    STRIKES_PER_MATURITY,
    X0,
)
from src.hjb_solver import Grid, solve_cascade
from src.linearized import gradient_phi0
from src.payoffs import PayoffFamily

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def verify_convexity(
    payoffs: PayoffFamily, grid: Grid, rng: np.random.Generator, n_pairs: int, n_theta: int
) -> dict:
    """Sample (lambda_1, lambda_2, theta) triples and check the Jensen inequality (p.16)."""
    defects = np.empty(n_pairs * n_theta)
    thetas = np.linspace(0.05, 0.95, n_theta)
    idx = 0
    for _ in range(n_pairs):
        l1 = rng.uniform(-CONVEXITY_LAMBDA_BOX, CONVEXITY_LAMBDA_BOX, size=payoffs.MP)
        l2 = rng.uniform(-CONVEXITY_LAMBDA_BOX, CONVEXITY_LAMBDA_BOX, size=payoffs.MP)
        phi_1 = solve_cascade(l1, payoffs, grid).phi0_value(X0)
        phi_2 = solve_cascade(l2, payoffs, grid).phi0_value(X0)
        for theta in thetas:
            mix = theta * l1 + (1 - theta) * l2
            phi_mix = solve_cascade(mix, payoffs, grid).phi0_value(X0)
            convex_combo = theta * phi_1 + (1 - theta) * phi_2
            # Convexity requires phi_mix <= convex_combo, i.e. defect >= 0.
            defects[idx] = convex_combo - phi_mix
            idx += 1
    min_defect = float(np.min(defects))
    violations = int(np.sum(defects < -1e-6))
    return {
        "n_triples": int(defects.size),
        "min_defect": min_defect,
        "mean_defect": float(np.mean(defects)),
        "n_violations_below_tol": violations,
        "convexity_holds": violations == 0,
    }


def verify_differentiability(payoffs: PayoffFamily, grid: Grid, rng: np.random.Generator) -> dict:
    """Compare the linearised gradient (eq. 75-76) with a central finite difference of Phi_0."""
    lam = rng.uniform(-1.0, 1.0, size=payoffs.MP)
    cascade = solve_cascade(lam, payoffs, grid)
    analytic_grad = gradient_phi0(lam, payoffs, cascade, X0)

    fd_grad = np.zeros(payoffs.MP)
    for i in range(payoffs.MP):
        lam_up = lam.copy()
        lam_up[i] += FD_STEP_GRADIENT
        lam_dn = lam.copy()
        lam_dn[i] -= FD_STEP_GRADIENT
        phi_up = solve_cascade(lam_up, payoffs, grid).phi0_value(X0)
        phi_dn = solve_cascade(lam_dn, payoffs, grid).phi0_value(X0)
        fd_grad[i] = (phi_up - phi_dn) / (2 * FD_STEP_GRADIENT)

    abs_err = np.abs(analytic_grad - fd_grad)
    rel_err = abs_err / np.maximum(np.abs(fd_grad), 1e-8)
    return {
        "lambda_probe": lam.tolist(),
        "analytic_gradient": analytic_grad.tolist(),
        "finite_difference_gradient": fd_grad.tolist(),
        "max_abs_error": float(np.max(abs_err)),
        "max_rel_error": float(np.max(rel_err)),
    }


def plot_phi0_slice(payoffs: PayoffFamily, grid: Grid, rng: np.random.Generator, path: Path) -> None:
    """1-D slice lambda(t) = l0 + t*direction, showing convexity of t -> Phi_0(lambda(t))."""
    l0 = rng.uniform(-0.5, 0.5, size=payoffs.MP)
    direction = rng.normal(size=payoffs.MP)
    direction /= np.linalg.norm(direction)
    ts = np.linspace(-CONVEXITY_LAMBDA_BOX, CONVEXITY_LAMBDA_BOX, 25)
    values = [solve_cascade(l0 + t * direction, payoffs, grid).phi0_value(X0) for t in ts]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, values, marker="o", ms=3)
    ax.set_xlabel("t (along a random direction in R^MP)")
    ax.set_ylabel(r"$\Phi_0(\lambda_0 + t\,\eta)$")
    ax.set_title("Dual value map $\\Phi_0$ along a one-dimensional slice (Prop. 2.4)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cascade_profile(payoffs: PayoffFamily, grid: Grid, path: Path) -> None:
    """phi_1^lambda(0, .) profile for lambda=0, illustrating a bounded continuous HJB solution."""
    cascade = solve_cascade(np.zeros(payoffs.MP), payoffs, grid)
    fig, ax = plt.subplots(figsize=(6, 4))
    for p, interval in enumerate(cascade.intervals):
        ax.plot(grid.x, interval.phi[0], label=f"$\\varphi_{p+1}^0(\\tau_{p},\\cdot)$")
    ax.set_xlabel("x (log-price)")
    ax.set_ylabel("$\\varphi_p^\\lambda$")
    ax.set_title("HJB cascade profiles at the start of each maturity interval ($\\lambda=0$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(full: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    payoffs = PayoffFamily(MATURITIES, STRIKES_PER_MATURITY)
    grid = Grid()
    n_pairs = CONVEXITY_N_PAIRS_PAPER if full else CONVEXITY_N_PAIRS_RUN
    n_theta = CONVEXITY_N_THETA_PAPER if full else CONVEXITY_N_THETA_RUN

    print(f"[exp_1] grid: L={grid.L}, n_x={grid.n_x}, dx={grid.dx:.5f}")
    print(f"[exp_1] payoff family: P={payoffs.P} maturities, MP={payoffs.MP} payoffs total")
    print(f"[exp_1] convexity sampling: n_pairs={n_pairs}, n_theta={n_theta} (full={full})")

    t0 = time.time()
    lam0 = np.zeros(payoffs.MP)
    cascade0 = solve_cascade(lam0, payoffs, grid)
    phi0_at_zero = cascade0.phi0_value(X0)
    assert np.isfinite(phi0_at_zero), "Phi_0(0) must be finite (Proposition 2.4)"
    print(f"[exp_1] Phi_0(0) = {phi0_at_zero:.6f} (finite, as required by Prop. 2.4)")

    convexity = verify_convexity(payoffs, grid, rng, n_pairs, n_theta)
    print(f"[exp_1] convexity check: min defect = {convexity['min_defect']:.3e}, "
          f"violations = {convexity['n_violations_below_tol']} / {convexity['n_triples']}")

    differentiability = verify_differentiability(payoffs, grid, rng)
    print(f"[exp_1] differentiability check: max |analytic - finite-difference| = "
          f"{differentiability['max_abs_error']:.3e}")

    plot_phi0_slice(payoffs, grid, rng, RESULTS_DIR / "exp1_phi0_convexity_slice.png")
    plot_cascade_profile(payoffs, grid, RESULTS_DIR / "exp1_cascade_profiles.png")

    elapsed = time.time() - t0
    results = {
        "experiment": "exp_1",
        "title": "HJB cascade solver: backward induction and dual value map computation",
        "full_configuration": full,
        "grid": {"L": grid.L, "n_x": grid.n_x, "dx": grid.dx},
        "payoff_family": {"P": payoffs.P, "MP": payoffs.MP, "M_p": payoffs.M_p,
                           "maturities": payoffs.maturities,
                           "strikes_per_maturity": payoffs.strikes_per_maturity},
        "phi0_at_lambda_zero": phi0_at_zero,
        "convexity_check": convexity,
        "differentiability_check": differentiability,
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "exp1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp_1] done in {elapsed:.1f}s; results saved to results/exp1_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HJB cascade solver and dual value map experiment")
    parser.add_argument("--full", action="store_true",
                         help="run the paper-scale convexity sampling instead of the reduced default")
    args = parser.parse_args()
    main(full=args.full)
