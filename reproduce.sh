#!/usr/bin/env bash
# Runs the test suite and every experiment of the reproduction in its
# default (reduced where applicable, see results/RESULTS.md) configuration.
set -uo pipefail

cd "$(dirname "$0")"

python -m pytest -q
python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4
python -m exp.exp_5
