"""Smoke test for Experiment 3 (gradient-based HJB dual calibration)."""

from __future__ import annotations

from exp.exp_3 import main


def test_exp3_main_runs_and_calibrates() -> None:
    results = main(full=False)

    expected_keys = {
        "experiment", "title", "full_configuration", "mc_n_paths",
        "target_price_vector", "lambda_star", "phi0_star", "V_pi",
        "gradient_residual_max", "monte_carlo_prices",
        "monte_carlo_calibration_max_error", "multi_start_max_lambda_deviation",
        "converged", "n_iter", "elapsed_seconds",
    }
    assert expected_keys.issubset(results.keys())

    assert results["converged"] is True
    assert results["gradient_residual_max"] < 0.05
    assert results["monte_carlo_calibration_max_error"] < 0.05
    assert results["multi_start_max_lambda_deviation"] < 0.05
