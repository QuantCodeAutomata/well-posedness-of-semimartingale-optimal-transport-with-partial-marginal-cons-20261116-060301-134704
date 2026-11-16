"""Paper-derived parameters for every experiment (module-level constants only).

Section references are to "Well-Posedness of Semimartingale Optimal Transport
with Partial Marginal Constraints for Model Calibration".
"""

# ---------------------------------------------------------------------------
# Common calibration setup (Sections 2-3): dimension d=1, maturities, payoffs.
# ---------------------------------------------------------------------------
DIM = 1  # spatial dimension d, Section 1.4 ("d=1" test case of Remark 3.3)
X0 = 0.0  # deterministic initial log-price x0, mu_0 = delta_{x0} (Section 2.2, eq. 27)
T_HORIZON = 1.0  # time horizon T (Section 1.2), not specified numerically in the paper
MATURITIES = (0.5, 1.0)  # tau_1 < ... < tau_P = T (Section 1.2); not specified, chosen for the example
STRIKES_PER_MATURITY = (
    (0.90, 1.05),  # strikes K_{i(1,m)} at maturity tau_1, distinct positive strikes (Section 1.4)
    (0.85, 1.00, 1.15),  # strikes K_{i(2,m)} at maturity tau_2
)  # log-put payoffs G_i(x) = (K_i - exp(x))^+, Section 1.4 and Remark 3.3

# ---------------------------------------------------------------------------
# Log-martingale running cost / Hamiltonian H(a,b) = h(b - a/2), Remark 3.3 (p.32).
# h is a custom smooth, strictly convex function with bounded, globally Lipschitz
# Hessian and a gradient range bounded away from zero, satisfying Assumption 3
# (D1)-(D2) [bounded effective domain, beta >= kappa > 0] and Assumption 6(i).
# Not specified explicitly by the paper; this is a custom implementation (see
# src/hamiltonian.py for the formula and derivation).
# ---------------------------------------------------------------------------
KAPPA_MIN = 0.02  # lower bound of h'(z), i.e. of beta^lambda = local variance (uniform ellipticity constant)
KAPPA_MAX = 0.50  # upper bound of h'(z): beta^lambda in (KAPPA_MIN, KAPPA_MAX)
KAPPA_SCALE = 5.0  # smoothing scale of the logistic barrier defining h (chosen so that
                    # h''(z) stays bounded away from 0 and from saturation on the range
                    # of z = b - a/2 encountered by the finite-difference second derivatives)

# ---------------------------------------------------------------------------
# Reference calibrateable model (Section 1.4, exp_3 methodology step 1):
# driftless Black-Scholes in log-price coordinates, i.e. the martingale measure
# for S_t = exp(X_t). Not specified by the paper; used only to generate a
# feasible (calibrateable) price vector pi in the relative interior of P.
# ---------------------------------------------------------------------------
REFERENCE_VOL = 0.22  # sigma_ref, with sigma_ref^2 in (KAPPA_MIN, KAPPA_MAX)

# ---------------------------------------------------------------------------
# Finite-difference grid for the HJB cascade (22) and linearized equations (75).
# The paper works on the full R^d (Section 1.4); spatial truncation to a bounded
# domain [-L, L] with Neumann boundary conditions is a numerical necessity not
# specified by the paper, justified because the payoffs are BUC with flat tails.
# ---------------------------------------------------------------------------
GRID_HALF_WIDTH = 2.5  # L: truncated spatial domain [-L, L] in log-price units
GRID_N_X = 161  # number of spatial grid points (odd, so x=0 is a node)
GRID_N_T_PER_UNIT_TIME = 400  # number of pseudo-time steps per unit of maturity length
PICARD_SWEEPS = 3  # policy-iteration sweeps per implicit time step for the nonlinear cascade

# ---------------------------------------------------------------------------
# exp_1: convexity / differentiability verification of Phi_0.
# ---------------------------------------------------------------------------
CONVEXITY_LAMBDA_BOX = 1.5  # half-width Lambda of the sampling box [-Lambda,Lambda]^{MP} for lambda_1, lambda_2
CONVEXITY_N_PAIRS_PAPER = 100  # number of (lambda_1, lambda_2) pairs, per implementation_notes ("at least 100 pairs")
CONVEXITY_N_THETA_PAPER = 10  # number of theta values in [0,1] sampled per pair, per implementation_notes
CONVEXITY_N_PAIRS_RUN = 20  # reduced number of pairs used for the executed run (time budget)
CONVEXITY_N_THETA_RUN = 5  # reduced number of theta values used for the executed run (time budget)
FD_STEP_GRADIENT = 1e-3  # finite-difference step for numerical differentiation of Phi_0
RANDOM_SEED = 20261116  # seed for every synthetic / Monte Carlo draw in this repository

# ---------------------------------------------------------------------------
# exp_2: Fokker-Planck primal vs HJB dual vs regular dual strong-duality check.
# ---------------------------------------------------------------------------
FP_GRID_N_X = 161  # spatial grid for the Fokker-Planck primal solver (same as the cascade grid)
FP_N_T_PER_UNIT_TIME = 400  # pseudo-time steps per unit maturity length for the primal solver
FP_MAX_ITER = 60  # outer alternating-projection iterations for the primal feedback optimisation

# ---------------------------------------------------------------------------
# exp_3: gradient-based dual calibration.
# ---------------------------------------------------------------------------
CALIBRATION_TOL = 1e-7  # optimisation tolerance on the dual gradient (L-BFGS-B 'gtol')
CALIBRATION_MAX_ITER = 200  # maximum L-BFGS-B iterations
MC_N_PATHS_PAPER = 200_000  # N_mc in implementation_notes (10^4-10^6); full configuration
MC_N_PATHS_RUN = 20_000  # reduced Monte Carlo sample used for the executed run (time budget)
MC_N_STEPS = 200  # Euler-Maruyama steps per unit maturity length for the Monte Carlo recovery

# ---------------------------------------------------------------------------
# exp_4: local strong convexity, admissible-price-set interior, Lipschitz stability.
# ---------------------------------------------------------------------------
STRONG_CONVEXITY_BOX = 1.0  # half-width of the compact box K used to estimate the modulus m_K
STRONG_CONVEXITY_N_SAMPLES_PAPER = 100  # paper-scale number of (lambda_1,lambda_2,theta) triples
STRONG_CONVEXITY_N_SAMPLES_RUN = 40  # reduced number of triples used for the executed run
PRICE_PERTURBATION_SCALE = 0.01  # size of the pi-perturbations used to explore int(P) and estimate Lipschitz L
PRICE_N_DIRECTIONS = 8  # number of random perturbation directions of pi sampled per maturity block
UNIQUENESS_N_STARTS = 5  # number of random starting points for the dual optimiser uniqueness check
VALUE_GRADIENT_FD_STEP = 1e-3  # finite-difference step for the C^1 check grad V = lambda* (eq. 31)

# ---------------------------------------------------------------------------
# exp_5: adapted Wasserstein stability of the optimal calibrated law.
# ---------------------------------------------------------------------------
AW2_N_PERTURBATIONS_PAPER = 6  # paper-scale number of pi_n -> pi perturbation sizes epsilon_n -> 0
AW2_N_PERTURBATIONS_RUN = 6  # executed number of perturbation sizes (same order, kept small)
AW2_MC_PATHS_PAPER = 20_000  # Monte Carlo paths for the common-noise coupled SDE simulation
AW2_MC_PATHS_RUN = 4_000  # reduced Monte Carlo sample used for the executed run (time budget)
AW2_N_STEPS = 200  # Euler-Maruyama steps per unit maturity length for the coupled simulation
