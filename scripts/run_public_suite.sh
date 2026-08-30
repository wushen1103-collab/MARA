#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

eval "$(python scripts/select_runtime_resources.py --workers-requested "${MARA_WORKERS_REQUESTED:-16}" --reserve-cpu "${MARA_RESERVE_CPU:-2}" --max-gpus "${MARA_MAX_GPUS:-1}" --out logs/resource_selection.json)"
export OMP_NUM_THREADS="${MARA_WORKERS}"
export OPENBLAS_NUM_THREADS="${MARA_WORKERS}"
export MKL_NUM_THREADS="${MARA_WORKERS}"
export NUMEXPR_NUM_THREADS="${MARA_WORKERS}"

python scripts/run_mara_public_suite.py \
  --datasets "${MARA_DATASET_1:-tdc:Caco2_Wang}" "${MARA_DATASET_2:-tdc:HIA_Hou}" moleculeace:auto:3 \
  --splits random scaffold \
  --seeds 13 29 47 \
  --workers "${MARA_WORKERS}" \
  --external-root "${MARA_EXTERNAL_ROOT:-external}" \
  --anchor-alpha "${MARA_ANCHOR_ALPHA:-0.9}" \
  --artifacts-root "${MARA_ARTIFACTS_ROOT:-artifacts/public_v1_anchor}" \
  ${MARA_EXTRA_ARGS:-} \
  2>&1 | tee "logs/$(basename "${MARA_ARTIFACTS_ROOT:-artifacts/public_v1_anchor}").log"

python scripts/build_paper_tables.py --artifacts-root "${MARA_ARTIFACTS_ROOT:-artifacts/public_v1_anchor}" --out "artifacts/tables_$(basename "${MARA_ARTIFACTS_ROOT:-artifacts/public_v1_anchor}")" 2>&1 | tee "logs/build_tables_$(basename "${MARA_ARTIFACTS_ROOT:-artifacts/public_v1_anchor}").log"
