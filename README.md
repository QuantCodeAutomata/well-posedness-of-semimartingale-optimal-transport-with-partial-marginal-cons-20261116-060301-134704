# Well-Posedness of Semimartingale Optimal Transport with Partial Marginal Constraints for Model Calibration

Reproduction of selected numerical experiments from:

Jean-François Chassagneux, Stéphane Crépey, Grégoire Loeper, Justin Ruelland,
*"Well-Posedness of Semimartingale Optimal Transport with Partial Marginal
Constraints for Model Calibration"*, arXiv:2611.17902 [q-fin.MF], 2026.

The paper studies a semimartingale optimal transport problem with partial
marginal (calibration) constraints at several maturities, connects its
probabilistic (Eulerian) and dual (HJB cascade) formulations, proves strong
duality, dual attainment, uniqueness and local Lipschitz stability of the
dual optimizer, and continuity of the optimal calibrated law in adapted
Wasserstein distance. All data used below is synthetic: a payoff family of
log-put options, a set of maturities, an initial state, and a running cost,
exactly as specified in Sections 1.4, 2 and 3 of the paper. No market data is
required or used.

## Repository layout

```
src/    HJB cascade solver, Hamiltonian, payoffs, linearized pricing
        equations, dual calibration, Fokker-Planck primal solver, feedback
        SDE, stability/strong-convexity diagnostics, adapted Wasserstein
        distance, and src/config.py with every parameter used below.
exp/    One script per experiment, exp_1 .. exp_5, run as python -m exp.exp_N.
data/   data/generate_synthetic.py produces the reference (Black-Scholes)
        calibration price vector used by exp_2 .. exp_5.
tests/  Unit tests for every src/ module.
results/  Metrics (JSON), tables, and plots written by the experiment scripts.
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or external data are required: every experiment is run on
synthetically specified payoffs, maturities and running costs.

## Tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp_1   # HJB cascade backward induction, Phi_0, convexity and differentiability
python -m exp.exp_2   # Strong duality: primal (Fokker-Planck) vs. regular dual vs. HJB dual
python -m exp.exp_3   # Gradient-based calibration via the linearized pricing equations
python -m exp.exp_4   # Local strong convexity of Phi_0, interior of the admissible price set,
                       # uniqueness and local Lipschitz stability of lambda*(pi)
python -m exp.exp_5   # Continuity of the optimal calibrated law in adapted Wasserstein distance
```

Each script also accepts `--full` to run the paper-scale configuration
instead of the reduced configuration used by default; see
`results/RESULTS.md` for the configurations actually executed and their
runtimes.

`reproduce.sh` runs the full test suite followed by all five experiments in
their default (reduced, where applicable) configuration.

## Results

See `results/RESULTS.md` for the comparison between the computed metrics and
the paper's expected outcomes, with a verdict for every compared quantity,
and the "Deviations from the paper" / "Not reproduced" sections.
