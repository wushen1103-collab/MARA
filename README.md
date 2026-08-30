# MARA

MARA (Multi-Axis Reliability Attribution) diagnoses prediction failures under
heterogeneous distribution shift. It retains five reliability axes before
scalar risk aggregation:

1. instance support;
2. source or domain support;
3. supervision reliability;
4. distribution frontier;
5. model conflict.

This repository contains the experiment code and compact reference summaries
used to evaluate failure ranking, calibration, selective prediction,
attribution, missing evidence, remediation, predictor transfer, and
scalability. Raw datasets, trained models, caches, and per-instance experiment
outputs are intentionally excluded.

## Quick Start

The reference environment uses Python 3.10. The setup script tries configurable
package mirrors before falling back to PyPI.

```bash
git clone <repository-url>
cd MARA
bash scripts/setup_env.sh
source .venv/bin/activate
```

Run a data-free smoke experiment:

```bash
bash scripts/run_smoke.sh
```

The command writes controlled failure-detection, cause-recovery,
missing-evidence, and remediation outputs under `artifacts/smoke/`.

## Public Molecular Benchmark

Install optional public-dataset dependencies, then run the compact benchmark.
TDC data are downloaded by PyTDC and MoleculeACE is shallow-cloned on demand.

```bash
INSTALL_OPTIONAL=1 bash scripts/setup_env.sh
source .venv/bin/activate
bash scripts/run_public_suite.sh
```

For the complete 30-task MoleculeACE evaluation:

```bash
python scripts/run_mara_public_suite.py \
  --datasets moleculeace:auto:30 \
  --splits random scaffold \
  --split-seed 20260811 \
  --seeds 13 29 47 \
  --workers 16 \
  --anchor-alpha 0.9 \
  --artifacts-root artifacts/moleculeace_seed20260811
```

The ten split seeds and aggregation command used by the matched molecular
benchmark are listed in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Other Experiment Families

The cross-domain commands use the core environment. Install the optional
dataset helpers as well when running TDC-backed experiments.

```bash
# ACSIncome; data are downloaded on first use.
python scripts/run_mara_crossdomain.py \
  --domain acs --data-root data/acs --out artifacts/acs_clean \
  --noise-strength 0 --workers 16

# OGBN-Arxiv; the official archive is downloaded on first use.
python scripts/run_mara_crossdomain.py \
  --domain arxiv --data-root data/arxiv --out artifacts/arxiv_clean \
  --noise-strength 0 --workers 16

# Controlled compound failures, incomplete evidence, and remediation.
python scripts/run_tkde_compound_suite.py \
  --out artifacts/controlled_full

# Approximate-neighbor scalability.
python scripts/benchmark_mara_scalability.py \
  --out artifacts/scalability --workers 16
```

Neural graph and ChemBERTa experiments require the optional packages in
`requirements-graph.txt`. Chemprop experiments use an independently installed
Chemprop CLI; pass its executable directory with `--chemprop-bin-dir`.

## Repository Layout

| Path | Contents |
|---|---|
| `scripts/` | Experiment runners, data preparation, aggregation, and audits |
| `docs/` | Dataset contracts, method formalization, and exact reproduction commands |
| `results/reference/` | Compact paper-level summaries for regression testing |
| `requirements*.txt` | Frozen core and optional dependency sets |

Each run writes a manifest, failure definition, predictions, metrics, and
task-specific summaries under `artifacts/`. Generated outputs remain ignored by
Git so that the repository stays small.

## Reproducibility Notes

- Proper training, validation, acquisition, and test roles are separated by the
  runners before fitting reliability heads.
- Dataset-run rank fusion uses unlabeled test-score ranks; the validation-CDF
  variant scores independent arrivals from frozen validation distributions.
- Controlled cause and counterfactual oracles are evaluation upper bounds and
  are never inputs to MARA.
- Reference CSV files are provided only to verify aggregate outputs; no private
  records or machine-specific manifests are included.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete command
matrix and [docs/DATASETS.md](docs/DATASETS.md) for data acquisition and schema
requirements.

## License

Code is released under the [MIT License](LICENSE). Dataset licenses and terms
remain with their respective providers.
