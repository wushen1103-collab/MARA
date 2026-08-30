#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

python scripts/run_tkde_compound_suite.py \
  --out artifacts/smoke \
  --seeds 20260831 \
  --n-train 2000 \
  --n-validation 1000 \
  --n-test 2000 \
  --mask-rates 0.1 0.3

test -s artifacts/smoke/compound_risk_mean_std.csv
test -s artifacts/smoke/remediation_mean_std.csv
echo "Smoke run completed: artifacts/smoke"
