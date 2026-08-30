# Reproducibility Guide

## Environment

The recorded environment uses Python 3.10 and the versions pinned in
`requirements.txt`. Optional public-dataset and scalability packages are in
`requirements-optional.txt`; neural graph and pretrained molecular packages are
in `requirements-graph.txt`.

```bash
# Core environment.
bash scripts/setup_env.sh

# Add TDC, OGB helpers, HNSW, and reporting utilities.
INSTALL_OPTIONAL=1 bash scripts/setup_env.sh

# Add PyTorch, PyG, and Transformers when neural backbones are required.
INSTALL_OPTIONAL=1 INSTALL_GRAPH=1 bash scripts/setup_env.sh
```

Set `PIP_INDEX_URL` to override the default package mirror. Every runner accepts
explicit output roots, so experiments can be executed on local disks or a job
scheduler without changing source code.

## Fast Integrity Run

The smoke command uses one seed and reduced controlled samples. It exercises
failure scoring, concurrent-cause attribution, missing-evidence handling, and
budgeted remediation without downloading data.

```bash
bash scripts/run_smoke.sh
```

## Primary Molecular Protocol

The matched benchmark uses ten split seeds. Within each split, predictor seeds
13, 29, and 47 form the frozen ensemble. The default method has no interaction
features and uses anchor strength 0.9.

```bash
for split_seed in \
  20260811 20260812 20260813 20260814 20260815 \
  20260816 20260817 20260818 20260819 20260820
do
  python scripts/run_mara_public_suite.py \
    --datasets moleculeace:auto:30 \
    --splits random scaffold \
    --split-seed "${split_seed}" \
    --seeds 13 29 47 \
    --workers 16 \
    --anchor-alpha 0.9 \
    --artifacts-root "artifacts/moleculeace_seed${split_seed}"
done

python scripts/summarize_multiseed_sota.py \
  --roots artifacts/moleculeace_seed* \
  --out artifacts/tables_moleculeace \
  --primary risk_mara_rank_fusion
```

ChEMBL and BindingDB use the same ten split seeds and predictor seeds. Prepare
their standardized CSV files as described in `docs/DATASETS.md`, then pass the
files in `--datasets` and use the protocol-specific split lists:

```bash
# ChEMBL: assay, temporal, target, and scaffold shifts.
python scripts/run_mara_public_suite.py \
  --datasets data/processed/chembl37_ic50_sample10k.csv \
  --splits assay temporal target scaffold \
  --split-seed 20260811 --seeds 13 29 47 --workers 16 \
  --anchor-alpha 0.9 --artifacts-root artifacts/chembl_seed20260811

# BindingDB: random, scaffold, target, and temporal shifts.
python scripts/run_mara_public_suite.py \
  --datasets data/processed/bindingdb_articles_ic50_sample10k.csv \
  --splits random scaffold target temporal \
  --split-seed 20260811 --seeds 13 29 47 --workers 16 \
  --anchor-alpha 0.9 --artifacts-root artifacts/bindingdb_seed20260811
```

## Cross-Domain Protocols

ACSIncome supports `state_shift`, `temporal_shift`, and
`state_time_compound`. OGBN-Arxiv supports its official temporal protocol and a
train-fitted structural-group protocol. Omitting `--protocols` runs all
protocols for the selected domain.

```bash
python scripts/run_mara_crossdomain.py \
  --domain acs --data-root data/acs --out artifacts/acs_clean \
  --noise-strength 0 --workers 16

python scripts/run_mara_crossdomain.py \
  --domain arxiv --data-root data/arxiv --out artifacts/arxiv_clean \
  --noise-strength 0 --workers 16 --graph-backbone sgc
```

To reproduce the label-noise conditions, rerun each command with
`--noise-strength 0.25`. Neural GCN and GraphSAGE sensitivity runs use
`--graph-backbone gcn` and `--graph-backbone sage`, respectively.

## Controlled, Missingness, and Action Protocols

```bash
python scripts/run_tkde_compound_suite.py \
  --out artifacts/controlled_full \
  --seeds 20260831 20260832 20260833 20260834 20260835 \
  --n-train 50000 --n-validation 25000 --n-test 50000 \
  --mask-rates 0.1 0.3 0.5 0.7

python scripts/run_support_intervention.py \
  --datasets tdc:Caco2_Wang tdc:HIA_Hou moleculeace:auto:3 \
  --splits random scaffold \
  --seeds 13 29 47 \
  --split-seed 20260811 \
  --workers 16 --budget 32 \
  --out artifacts/support_action_seed20260811
```

For the five-seed support-action estimate, repeat the final command with split
seeds 20260811 through 20260815 and aggregate their common parent directory:

```bash
python scripts/summarize_support_action.py \
  --root artifacts/support_action_runs \
  --out artifacts/tables_support_action
```

The action runner splits the original validation partition into disjoint
calibration and unlabeled acquisition pools. Failure labels are constructed
only for calibration records before selection.

## Scalability

```bash
python scripts/benchmark_mara_scalability.py \
  --out artifacts/scalability \
  --sizes 10000 50000 100000 250000 500000 1000000 \
  --repeats 3 --repeats-at-1m 3 --workers 16
```

## Output Verification

`results/reference/` contains only compact aggregate tables. Compare newly
generated summaries by key columns rather than raw row order. The SHA-256 file
`results/reference/SHA256SUMS` protects the distributed reference summaries.
