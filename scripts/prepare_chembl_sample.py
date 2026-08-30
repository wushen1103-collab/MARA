#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="ChEMBL-derived Parquet file; see docs/DATASETS.md for the required columns.",
    )
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--out", default="data/processed/chembl37_ic50_sample10k.csv")
    args = parser.parse_args()

    cols = [
        "canonical_smiles_rdkit",
        "pIC50",
        "assay_id",
        "assay_chembl_id",
        "assay_type",
        "confidence_score",
        "standard_relation",
        "standard_units",
        "publication_year",
        "doc_id",
        "target_chembl_id",
        "bemis_murcko_scaffold",
        "is_first_seen_scaffold",
    ]
    df = pd.read_parquet(args.source, columns=cols)
    df = df[df["canonical_smiles_rdkit"].notna() & df["pIC50"].notna()].copy()
    df = df.rename(columns={"canonical_smiles_rdkit": "smiles", "pIC50": "y"})
    if len(df) > args.n:
        per_year = max(50, args.n // max(1, df["publication_year"].nunique()))
        chunks = []
        for _, group in df.groupby("publication_year", sort=True):
            chunks.append(group.sample(n=min(len(group), per_year), random_state=args.seed))
        sampled = pd.concat(chunks, ignore_index=True)
        if len(sampled) < args.n:
            seen = set(sampled.index)
            remainder = df.loc[[i for i in df.index if i not in seen]]
            sampled = pd.concat(
                [sampled, remainder.sample(n=min(args.n - len(sampled), len(remainder)), random_state=args.seed + 1)],
                ignore_index=True,
            )
        df = sampled.sample(n=min(args.n, len(sampled)), random_state=args.seed + 2).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    out.with_suffix(".manifest.txt").write_text(
        f"source={args.source}\nrows={len(df)}\nseed={args.seed}\ncolumns={','.join(df.columns)}\n",
        encoding="utf-8",
    )
    print(f"wrote {out} rows={len(df)}")


if __name__ == "__main__":
    main()
