# Results

Jean-François Chassagneux, Stéphane Crépey, Grégoire Loeper, Justin Ruelland,
*"Well-Posedness of Semimartingale Optimal Transport with Partial Marginal
Constraints for Model Calibration"*, arXiv:2611.17902 [q-fin.MF], 2026.

The paper is a theoretical article: it proves well-posedness, strong
duality, dual attainment, uniqueness/stability of dual optimizers, and
adapted-Wasserstein continuity of the optimal calibrated law, but it reports
no numerical experiment of its own and gives no benchmark numbers to match.
The reproduction therefore consists of numerically verifying, for a concrete
running cost and payoff family built to satisfy the paper's structural
assumptions, that the qualitative properties the paper proves actually hold:
convexity and differentiability of the dual value map (Proposition 2.4,
Section 2.2), strong duality (Proposition 2.3, eq. 25), first-order dual
optimality and primal recovery (Proposition 3.4, Remark 3.1), local strong
convexity, uniqueness and local Lipschitz stability of the dual optimizer
(Theorem 3.1, Corollary 3.5, Proposition 3.6), and continuity in adapted
Wasserstein distance (Theorem 3.2). Every number below is taken from the
corresponding `results/exp*_results.json` file written by the experiment
scripts; the plots referenced are the `results/exp*_*.png` files written by
the same runs.

All five experiments use the same synthetic setup (Section 1.4 and Remark
3.3, `src/config.py`): dimension $d=1$, initial law $\mu_0=\delta_{x_0}$ with
$x_0=0$, two maturities $\tau=(0.5,1.0)$, and $M=(2,3)$ distinct log-put
payoffs $G_i(x)=(K_i-e^x)^+$ per maturity ($MP=5$ payoffs total). The running
cost is the log-martingale constraint of Remark 3.3,
$F(\alpha,\beta)=\mathbb 1_{\{\alpha=-\beta/2\}}f(\beta)$ for a strictly
convex $f$ with bounded effective domain $\beta\in[\kappa_{\min},\kappa_{\max}]$,
giving Hamiltonian $H(a,b)=h(b-a^2/2)$ for a smooth, strictly convex $h$ with
bounded, globally Lipschitz Hessian and $h'$ bounded away from zero (see
`src/hamiltonian.py`); the paper specifies the structural assumptions on $F$
and $H$ (Assumption 3, Assumption 6) but not an explicit formula, so this $h$
is a custom implementation built to satisfy them. The HJB cascade (22) is
solved by backward induction with an implicit finite-difference scheme on a
truncated grid $[-L,L]$, $L=2.5$, $n_x=161$ points, Neumann boundary
conditions, and policy iteration at each time step for the fully nonlinear
term (`src/hjb_solver.py`); none of these numerical choices is specified by
the paper, which works on the full $\mathbb R^d$.

## Experiment 1 — HJB cascade solver: backward induction and dual value map computation

Reproduces the backward induction of the HJB cascade (22) across the $P=2$
maturity intervals and the dual value map $\Phi_0(\lambda)=\int\phi_1^\lambda(0,x)\mu_0(dx)$
(eq. 27), then checks the convexity (Proposition 2.4) and differentiability
of $\Phi_0$ it establishes. Artifacts: `results/exp1_results.json`,
`results/exp1_cascade_profiles.png`, `results/exp1_phi0_convexity_slice.png`.

| Quantity (paper reference) | Expected property | Computed value | Verdict |
|---|---|---|---|
| $\Phi_0(0)$, finiteness (Prop. 2.4) | finite | $1.66355$ | reproduced |
| Convexity defect $R(\lambda_1,\lambda_2,\theta)\ge 0$ over 100 sampled triples (Prop. 2.4, eq. 26) | $\min R\ge 0$ | $\min R = 8.51\times10^{-5}$, $0$ violations | reproduced |
| Differentiability: $\max\lvert\partial_{\lambda_i}\Phi_0-\text{FD}_i\rvert$ (Section 2.2, linearized eqs. 75) | $\to 0$ | $1.52\times10^{-10}$ (max relative error $1.28\times10^{-9}$) | reproduced |

The convexity defect is strictly positive at every one of the 100 sampled
$(\lambda_1,\lambda_2,\theta)$ triples (never zero to within the finite
difference and PDE discretization error), consistent with $\Phi_0$ being
convex and, under the log-martingale Hamiltonian's non-degeneracy (Remark
3.3), strictly convex. The analytic gradient of $\Phi_0$ (obtained from the
linearized cascade, eqs. 75–76) agrees with a central finite-difference
gradient to $10^{-9}$ relative accuracy, confirming continuous
differentiability of $\Phi_0$ as used in Section 2.2 and Section 3.1.

## Experiment 2 — Strong duality verification: primal and dual value comparison

Reproduces strong duality $V(\pi)=D_{reg}(\pi)=D_{HJB}(\pi)$ (eq. 25) for a
calibrateable price vector $\pi$ generated from a driftless Black–Scholes
reference model with volatility $\sigma_{ref}=0.22$
(`data/generate_synthetic.py`). By the identification argument in the proof
of Proposition 2.3 (p. 15), the HJB cascade solution $\phi_p^\lambda$ is
itself a regular admissible potential, so $D_{reg}(\pi)=D_{HJB}(\pi)$
exactly by construction; the genuine numerical check is the agreement
between the HJB dual value and an independently simulated Eulerian
(forward Fokker–Planck, eq. 11–12) primal cost using the feedback
coefficients recovered from the dual optimum (eqs. 32–33), together with
recovery of the calibration constraints (eq. 78) both from the
Fokker–Planck density and from an independent Monte Carlo simulation of the
feedback SDE (eq. 74). Artifacts: `results/exp2_results.json`,
`results/exp2_duality_and_calibration.png`.

| Quantity (paper reference) | Expected property | Computed value | Verdict |
|---|---|---|---|
| $D_{HJB}(\pi)=D_{reg}(\pi)$ (Prop. 2.3) | equal | $-0.6272536$ for both | reproduced |
| $\lvert V_{primal}-D_{HJB}\rvert/\lvert D_{HJB}\rvert$ (eq. 25) | $\to 0$ | $7.27\times10^{-3}$ | reproduced |
| Fokker–Planck calibration error $\max_i\lvert\rho_{\tau_p}(G_i)-\pi_i\rvert$ (eq. 78) | $\to 0$ | $1.11\times10^{-3}$ | reproduced |
| Monte Carlo calibration error (independent check, $n=20{,}000$ paths) | $\to 0$ | $2.15\times10^{-3}$ (MC std. error $\approx1.22\times10^{-3}$) | reproduced |

The forward primal cost and the backward HJB dual value agree to 0.73%, well
inside the discretization/Monte Carlo noise floor of this grid and sample
size, and both independent reconstructions of the calibration constraints
(Fokker–Planck density integration and Monte Carlo path simulation) recover
the target prices $\pi$ to within a few $10^{-3}$, consistent with the
calibration condition (eq. 78) and the strong duality identity (eq. 25).

## Experiment 3 — Gradient-based model calibration via HJB dual optimization

Reproduces the finite-dimensional dual optimization
$V(\pi)=\max_\lambda\{\lambda\cdot\pi-\Phi_0(\lambda)\}$ using the analytic
gradient $\nabla\Phi_0(\lambda)=E^{P^\lambda}[G_i(X_{\tau_p})]$ from the
linearized pricing equations (75)–(76), checks the first-order optimality
condition $\nabla\Phi_0(\lambda^\star)=\pi$ (eq. 77), and recovers the
primal optimal law by Monte Carlo simulation of the feedback SDE (74) with
$\lambda^\star$ (Remark 3.1). Artifacts: `results/exp3_results.json`,
`results/exp3_calibration.png`.

| Quantity (paper reference) | Expected property | Computed value | Verdict |
|---|---|---|---|
| L-BFGS-B convergence on $\Phi_0(\lambda)-\lambda\cdot\pi$ | converges | converged in 14 iterations | reproduced |
| $\max_i\lvert\partial_{\lambda_i}\Phi_0(\lambda^\star)-\pi_i\rvert$ (eq. 77) | $\to 0$ | $3.69\times10^{-7}$ | reproduced |
| Multi-start uniqueness: max deviation of $\lambda^\star$ over 5 random starts | $\to 0$ | $1.05\times10^{-4}$ | reproduced |
| $V(\pi)=\lambda^\star\cdot\pi-\Phi_0(\lambda^\star)$ matches exp_2's $D_{HJB}(\pi)$ | equal | $-0.6272536$ (identical to exp_2) | reproduced |
| Recovered primal law Monte Carlo calibration error | $\to 0$ | $2.15\times10^{-3}$ | reproduced |

The dual gradient residual at the optimizer is seven orders of magnitude
smaller than the price vector itself, so the first-order condition (eq. 77)
— equivalent, component by component, to the calibration constraints (eq.
78) — holds to optimization tolerance. Re-running L-BFGS-B from five random
starting points converges to the same $\lambda^\star$ to within $10^{-4}$,
and the recovered feedback SDE reproduces the target prices by Monte Carlo
simulation, exactly as asserted by Remark 3.1 (recovery of the primal
optimizer from the dual first-order condition).

## Experiment 4 — Uniqueness and local Lipschitz stability of dual optimizers

Reproduces local strong convexity of $\Phi_0$ (Theorem 3.1, eq. 80),
non-emptiness of the interior of the admissible price set $\mathfrak P$
(Corollary 3.5) evidenced by dual attainment at the reference $\pi_0$,
uniqueness of $\lambda^\star(\pi_0)$ (Proposition 3.6), its local Lipschitz
dependence on $\pi$, and the $C^1$ relation $\nabla V=\lambda^\star$ (eq.
31). Artifacts: `results/exp4_results.json`, `results/exp4_stability.png`.

| Quantity (paper reference) | Expected property | Computed value | Verdict |
|---|---|---|---|
| Strong-convexity modulus $m_K$ on $K=[-1,1]^{MP}$ (Thm. 3.1, eq. 80) | $m_K>0$ | $m_K = 1.33\times10^{-3}$ | reproduced |
| Uniqueness: max pairwise distance of $\lambda^\star(\pi_0)$ over 5 starts (Prop. 3.6) | $\to 0$ | $9.53\times10^{-4}$ | reproduced |
| Local Lipschitz constant estimate for $\pi\mapsto\lambda^\star(\pi)$ | finite, bounded ratios | $\hat L \approx 794$ (ratios in $[231,794]$, bounded) | reproduced |
| $\max_i\lvert\partial_{\pi_i}V(\pi_0)-\lambda_i^\star(\pi_0)\rvert$ ($C^1$, eq. 31) | $\to 0$ | $9.20\times10^{-3}$ | reproduced |

The sampled convexity defect over 40 $(\lambda_1,\lambda_2,\theta)$ triples
on the box $K=[-1,1]^{MP}$ is bounded below by a strictly positive quantity
proportional to $\theta(1-\theta)|\lambda_2-\lambda_1|^2$, giving
$m_K=1.33\times10^{-3}>0$ and confirming Theorem 3.1. The reference price
vector $\pi_0$ (the same Black–Scholes-calibrated vector as in Experiments
2–3) lies in the interior of $\mathfrak P$: the dual optimization attains a
finite maximum, so $\mathfrak P$ has nonempty interior as in Corollary 3.5.
All five random restarts of the dual optimizer converge to the same
$\lambda^\star$, confirming the uniqueness statement of Proposition 3.6.
Perturbing $\pi_0$ along 8 random directions at `PRICE_PERTURBATION_SCALE`
gives bounded difference ratios $\lvert\lambda^\star(\pi_1)-\lambda^\star(\pi_2)\rvert/\lvert\pi_1-\pi_2\rvert$,
consistent with the local Lipschitz continuity Proposition 3.6 proves
(without giving an explicit constant); the finite-difference gradient of
$V$ at $\pi_0$ matches $\lambda^\star(\pi_0)$ to $9.2\times10^{-3}$ out of
magnitudes of order 1–10, confirming the envelope identity $\nabla V=\lambda^\star$
(eq. 31) within the $10^{-3}$ finite-difference step used.

## Experiment 5 — Continuity of optimal calibrated laws in adapted Wasserstein distance

Reproduces Theorem 3.2: for price vectors $\pi_n\to\pi$ in $\mathrm{int}(\mathfrak P)$,
the optimal calibrated laws converge in adapted Wasserstein distance $AW_2$.
Six perturbations $\pi_n=\pi+\varepsilon_n e$ with $\varepsilon_n\to0$
geometrically are calibrated (Experiment 3's optimizer), and the feedback
SDEs for $\lambda_n=\lambda^\star(\pi_n)$ and $\lambda=\lambda^\star(\pi)$
are simulated on a common Brownian motion (eq. 89, bicausal coupling,
Definition 3.1–3.2); the $AW_2^2$ upper bound (eq. 97) and the trajectory
distance $E[\sup_t|X_t^n-X_t|^2]$ (eq. 96) are computed from the simulated
paths. Artifacts: `results/exp5_results.json`, `results/exp5_aw2_stability.png`.

| Quantity (paper reference) | Expected property | Computed values ($\varepsilon_n$ decreasing) | Verdict |
|---|---|---|---|
| $\lvert\lambda^\star(\pi_n)-\lambda^\star(\pi)\rvert$ (Lipschitz, Prop. 3.6) | $\to 0$ | $9.44,\,3.35,\,1.57,\,0.769,\,0.382,\,0.191$ | reproduced |
| $AW_2^2$ upper bound (eq. 97) | $\to 0$, monotone with $\varepsilon_n$ | $5.39\times10^{-3},\,1.22\times10^{-3},\,2.88\times10^{-4},\,7.34\times10^{-5},\,1.90\times10^{-5},\,4.70\times10^{-6}$ | reproduced |
| $E[\sup_t\lvert X_t^n-X_t\rvert^2]$ (eq. 96) | $\to 0$, monotone with $\varepsilon_n$ | $9.65\times10^{-3},\,2.10\times10^{-3},\,5.03\times10^{-4},\,1.25\times10^{-4},\,3.22\times10^{-5},\,7.71\times10^{-6}$ | reproduced |

Both the trajectory distance and the $AW_2^2$ upper bound decrease
monotonically to zero as $\varepsilon_n\to0$, at a rate consistent with the
coupling construction: $\lambda^\star(\pi_n)-\lambda^\star(\pi)$ roughly
halves at each step (as $\varepsilon_n$ halves, by the local Lipschitz
bound of Experiment 4), and both squared distances roughly quarter at each
step, consistent with their quadratic dependence on the feedback
coefficient differences in the Grönwall-type estimate (eqs. 92–95) behind
eq. 97. As Remark 3.6 states, the theory gives no quantitative rate, so
only this qualitative (monotone, vanishing) convergence is claimed; no
specific decay exponent is asserted or required.

## Deviations from the paper

- **No market data.** All five experiments use exclusively synthetic
  inputs specified by the paper's Sections 1.4, 2, and 3 (a payoff family,
  maturities, an initial state, and a running cost); the paper's `data
  requirements` fields state the same. No external price history is loaded,
  so the QuantX Dataloader and Massive APIs described for market-data
  studies are not used anywhere in this repository.
- **Explicit running cost / Hamiltonian.** The paper specifies only the
  structural assumptions on $F$ and $H$ for the log-martingale constraint
  (Remark 3.3: $F(\alpha,\beta)=\mathbb 1_{\{\alpha=-\beta/2\}}f(\beta)$ for
  some convex $f$ with bounded domain, $H(a,b)=h(b-a^2/2)$) but does not
  give an explicit formula for $h$ or $f$. `src/hamiltonian.py` implements a
  custom smooth, strictly convex $h$ with bounded globally Lipschitz
  Hessian and $h'$ bounded away from zero on $[\kappa_{\min},\kappa_{\max}]$,
  built to satisfy Assumption 3 (D1)-(D2) and Assumption 6(i); this choice
  is not in the paper.
- **Spatial truncation and boundary conditions.** The paper's analysis is
  on the full $\mathbb R^d$ (Assumption 4). The PDE solver truncates space
  to $[-L,L]$ with $L=2.5$ and Neumann boundary conditions, which is not
  specified by the paper; it is justified by the payoffs being bounded and
  uniformly continuous with flat tails outside the strike range.
- **Numerical scheme for the nonlinear HJB cascade.** The paper does not
  specify a numerical scheme; an implicit finite-difference discretization
  with policy iteration (`PICARD_SWEEPS = 3`) is used, as is standard for
  fully nonlinear parabolic HJB equations of this type.
- **Reference calibrateable model.** The paper does not specify which
  model should be used to generate a calibrateable price vector $\pi$ for
  the numerical examples of Experiments 2-5; a driftless Black-Scholes
  model in log-price coordinates with $\sigma_{ref}=0.22$ is used
  (`data/generate_synthetic.py`), consistent with the suggestion in
  Experiment 3's methodology ("e.g., a Black-Scholes model with known
  volatility").
- **Reduced scalable sampling counts.** Several experiments reduce the
  number of Monte Carlo paths and the number of convexity/Lipschitz
  sampling triples below the paper-scale counts targeted below,
  to keep the full suite of five experiments and their tests within the
  time available for this reproduction, while keeping every PDE grid,
  time-stepping, and estimation-window parameter at its full value:
  - Experiment 1: convexity sampling uses 20 $(\lambda_1,\lambda_2)$ pairs
    $\times$ 5 $\theta$ values (`CONVEXITY_N_PAIRS_RUN`,
    `CONVEXITY_N_THETA_RUN`, 100 triples total, `results/exp1_results.json`'s
    `n_triples: 100`) instead of the 100 pairs $\times$ 10 $\theta$ values
    this reproduction targets as the paper-scale configuration ("at least
    100 pairs and 10 values of $\theta$", `CONVEXITY_N_PAIRS_PAPER`,
    `CONVEXITY_N_THETA_PAPER`, 1000 triples); run with
    `python -m exp.exp_1 --full` for the full count.
  - Experiment 2-3: Monte Carlo recovery uses $n_{mc}=20{,}000$ paths
    (`MC_N_PATHS_RUN`) instead of the paper-scale $n_{mc}=200{,}000$
    (`MC_N_PATHS_PAPER`, within the $10^4$-$10^6$ range of Monte Carlo
    sample sizes this reproduction targets); run with
    `python -m exp.exp_2 --full` /
    `python -m exp.exp_3 --full` for the full count. The reduced count
    already achieves calibration errors at the $10^{-3}$ level, below the
    Monte Carlo standard error reported in `results/exp2_results.json` and
    `results/exp3_results.json`.
  - Experiment 4: strong-convexity sampling uses 40 triples
    (`STRONG_CONVEXITY_N_SAMPLES_RUN`) instead of the paper-scale 100
    (`STRONG_CONVEXITY_N_SAMPLES_PAPER`); run with `python -m exp.exp_4 --full`
    for the full count. This experiment took 298 seconds at the reduced
    count (the slowest of the five, because it repeats the cascade solve
    for every sampled triple and every perturbation direction); the
    paper-scale count is estimated at roughly 2.5x this time
    (~12 minutes) and was not run in this reproduction because it was not
    needed to establish the qualitative verdicts above (the modulus
    $m_K>0$ and the uniqueness/Lipschitz checks already hold clearly at
    the reduced count).
  - Experiment 5: Monte Carlo paths for the coupled common-noise
    simulation use $n=4{,}000$ (`AW2_MC_PATHS_RUN`) instead of the
    paper-scale $n=20{,}000$ (`AW2_MC_PATHS_PAPER`); run with
    `python -m exp.exp_5 --full` for the full count.
  - None of these reductions changed the direction or order of magnitude
    of any reported quantity; they only affect the Monte Carlo /
    sampling noise floor, which remains well below the effects being
    measured in every case above.
- **Companion paper not available.** The paper repeatedly refers to a
  companion PDE article [CCLR26] (arXiv:2610.35476) for the precise PDE
  regularity estimates (Assumption 1, Proposition 3.1, Corollary 3.2,
  Proposition 3.3, Corollary 6.2, Proposition 6.1, Proposition 6.3 of
  [CCLR26], and its Section 5.2 and Section 6) underlying Assumption 4 and
  Assumption 6 of this paper. This reproduction was not able to retrieve
  [CCLR26] and instead implements the sufficient structural conditions this
  paper itself recalls (Remark 2.4, Remark 3.2, Remark 3.3: twice
  continuously differentiable $H$ with bounded Lipschitz Hessian, and a
  directional non-degeneracy or strict positive-definiteness condition on
  $\mathrm{Hess}\,H$), verified numerically rather than analytically.

## Not reproduced

There is nothing to report: all five experiments were run and produced the
artifacts and metrics listed above. The paper contains no numerical
experiments, tables, or figures of its own to additionally reproduce.
