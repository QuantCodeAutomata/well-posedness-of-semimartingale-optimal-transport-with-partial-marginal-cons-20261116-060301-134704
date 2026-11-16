# Repository guide

This repository reproduces five numerical experiments from Jean-François
Chassagneux, Stéphane Crépey, Grégoire Loeper, Justin Ruelland,
*"Well-Posedness of Semimartingale Optimal Transport with Partial Marginal
Constraints for Model Calibration"* (arXiv:2611.17902): the HJB cascade
solver and dual value map (Section 2.2), strong duality between the
Eulerian primal and the HJB dual (Section 2.1-2.2), gradient-based dual
calibration and primal recovery (Section 3.1), local strong convexity and
Lipschitz stability of the dual optimizer (Section 3.2), and continuity of
the optimal calibrated law in adapted Wasserstein distance (Section 3.3).
All data is synthetic (a payoff family of log-put options, maturities, an
initial state, and a running cost); no market data is used.

## Layout

```
src/     Library modules: payoffs, Hamiltonian/running cost, HJB cascade
         solver (backward induction), linearized pricing equations, dual
         calibration, Fokker-Planck primal solver, feedback SDE simulation,
         stability/strong-convexity diagnostics, adapted Wasserstein
         distance, and config.py (every parameter, module-level constants
         only, each with a comment citing its paper section or "not
         specified in the paper").
exp/     One script per experiment (exp_1 .. exp_5), each runnable as
         `python -m exp.exp_N` and accepting `--full` for the paper-scale
         configuration (see results/RESULTS.md for what differs).
data/    data/generate_synthetic.py: the reference Black-Scholes price
         vector used to calibrate Experiments 2-5. No market data loader is
         needed or present; no external data is downloaded by this
         repository.
tests/   Unit tests for every src/ module and for each experiment's shared
         code path.
results/ Metrics (JSON), plots (PNG), and RESULTS.md written by the
         experiment scripts. Re-running an experiment overwrites its files
         here; the prose, tables, and verdicts in RESULTS.md must then be
         re-checked by hand against the new files and updated if they
         changed.
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests (no market data is used anywhere, so there is no `market_data` marker
to exclude; the full suite always runs):

```bash
python -m pytest
```

Experiments (reduced configuration by default; add `--full` for the
paper-scale configuration described in `results/RESULTS.md`):

```bash
python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4
python -m exp.exp_5
```

`reproduce.sh` runs the test suite followed by all five experiments in
their default configuration.

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  UPPER_CASE constant with a one-line comment giving its meaning, unit, and
  source (a paper section/equation, or "not specified in the paper"); no
  parameter is hardcoded inside `src/` or `exp/` modules. Where a parameter
  was reduced to fit the time available for this reproduction, both the
  paper-scale and the executed value are defined side by side
  (`..._PAPER` / `..._RUN`), and the experiment's `--full` flag selects the
  paper-scale value.
- All library code is in flat modules directly under `src/`; experiment
  scripts live only in `exp/`; all tests live only in `tests/`.
- Deviations from the paper (data, parameters, numerical schemes not
  specified by the paper, or reduced sampling/Monte Carlo counts) are
  recorded only in `results/RESULTS.md`, never left implicit in code.
- API keys are not applicable to this repository (no market data is
  loaded), so there is no `.env` or `.env.example` file.
