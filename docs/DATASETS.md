# Dataset Acquisition and Contracts

Raw datasets are not redistributed. The runners download open benchmark data
when an official programmatic source is available and otherwise accept a local
CSV or Parquet file.

## Automatically Retrieved

| Dataset | Runner specification | Local cache |
|---|---|---|
| TDC ADME/Tox | `tdc:<dataset-name>` | `data/tdc/` |
| MoleculeACE | `moleculeace:auto:<task-count>` | `external/MoleculeACE/` |
| ACSIncome | `run_mara_crossdomain.py --domain acs` | selected data root |
| OGBN-Arxiv | `run_mara_crossdomain.py --domain arxiv` | selected data root |

MoleculeACE is obtained from its public Git repository with a depth-one clone.
ACSIncome uses a public Parquet mirror with the upstream host as fallback.
OGBN-Arxiv uses the official OGB archive. Downloaded data retain the licenses
and usage terms of their providers.

## Generic Molecular File

CSV, TSV, and Parquet inputs are standardized automatically when they contain:

- a molecular string column named `smiles`, `canonical_smiles`, or a close
  equivalent;
- a target column named `y`, `label`, `pIC50`, or a close equivalent.

Optional metadata columns containing assay, source, year/date, confidence,
target, document, relation, unit, or cliff information are retained for shift
construction and reliability axes.

## ChEMBL Preparation

`prepare_chembl_sample.py` expects a ChEMBL-derived Parquet file with these
columns:

```text
canonical_smiles_rdkit, pIC50, assay_id, assay_chembl_id, assay_type,
confidence_score, standard_relation, standard_units, publication_year,
doc_id, target_chembl_id, bemis_murcko_scaffold, is_first_seen_scaffold
```

```bash
python scripts/prepare_chembl_sample.py \
  --source /path/to/chembl37_ic50.parquet \
  --out data/processed/chembl37_ic50_sample10k.csv \
  --n 10000 --seed 20260811
```

## BindingDB Preparation

`prepare_bindingdb_sample.py` accepts CSV or Parquet and detects common names
for SMILES, affinity, target, source, year, document, and scaffold fields. SMILES,
numeric affinity, and target identity are required.

```bash
python scripts/prepare_bindingdb_sample.py \
  --source /path/to/bindingdb_ic50.parquet \
  --out data/processed/bindingdb_articles_ic50_sample10k.csv \
  --max-rows 10000 --target-cap 20 --seed 20260811
```

## BOOM and DrugOOD

BOOM fixed-split inputs can be prepared with
`scripts/prepare_boom_fixed_splits.py`. DrugOOD official JSON files are passed
as `drugood:/path/to/file.json` after users obtain them under the benchmark's
distribution terms. Neither dataset is bundled in this repository.
