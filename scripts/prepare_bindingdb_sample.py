#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in columns:
        key = col.lower()
        if any(cand.lower() in key for cand in candidates):
            return col
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="BindingDB-derived CSV or Parquet file; see docs/DATASETS.md for the accepted schema.",
    )
    parser.add_argument("--out", default="data/processed/bindingdb_articles_ic50_sample10k.csv")
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--target-cap", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = read_table(source)
    smiles_col = find_column(list(raw.columns), ["smiles", "canonical_smiles", "drug_smiles"])
    y_col = find_column(list(raw.columns), ["y", "pic50", "pchembl", "label", "raw_y"])
    target_col = find_column(list(raw.columns), ["target_id", "target_chembl_id", "uniprot_accession", "target"])
    if smiles_col is None or y_col is None or target_col is None:
        raise RuntimeError(f"missing required BindingDB columns in {source}: {list(raw.columns)}")

    out_frame = pd.DataFrame(
        {
            "smiles": raw[smiles_col].astype(str),
            "y": pd.to_numeric(raw[y_col], errors="coerce"),
            "target_id": raw[target_col].astype(str),
        }
    )
    source_col = find_column(list(raw.columns), ["dataset", "source", "domain_id"])
    year_col = find_column(list(raw.columns), ["publication_year", "year", "Date of publication"])
    doc_col = find_column(list(raw.columns), ["PMID", "Article DOI", "doc_id", "doi"])
    scaffold_col = find_column(list(raw.columns), ["scaffold"])
    out_frame["source"] = raw[source_col].astype(str) if source_col else "BindingDB"
    out_frame["publication_year"] = pd.to_numeric(raw[year_col], errors="coerce") if year_col else np.nan
    out_frame["doc_id"] = raw[doc_col].astype(str) if doc_col else out_frame["target_id"]
    if scaffold_col:
        out_frame["scaffold"] = raw[scaffold_col].astype(str)

    out_frame = out_frame.replace({"nan": np.nan, "None": np.nan})
    out_frame = out_frame[out_frame["smiles"].notna() & out_frame["y"].notna() & out_frame["target_id"].notna()].copy()
    out_frame = out_frame.drop_duplicates(subset=["smiles", "target_id", "y"]).reset_index(drop=True)
    out_frame = out_frame[out_frame["target_id"].map(out_frame["target_id"].value_counts()) >= 3].reset_index(drop=True)

    if len(out_frame) > args.max_rows:
        rng_seed = int(args.seed)
        capped_parts = []
        for _, group in out_frame.groupby("target_id", sort=False):
            capped_parts.append(group.sample(n=min(len(group), args.target_cap), replace=False, random_state=rng_seed))
        capped = pd.concat(capped_parts, axis=0)
        if len(capped) >= args.max_rows:
            out_frame = capped.sample(n=args.max_rows, random_state=rng_seed).reset_index(drop=True)
        else:
            remaining = out_frame.drop(index=capped.index, errors="ignore")
            need = min(args.max_rows - len(capped), len(remaining))
            fill = remaining.sample(n=need, random_state=rng_seed + 1) if need else remaining.iloc[[]]
            out_frame = pd.concat([capped, fill], ignore_index=True).sample(frac=1.0, random_state=rng_seed + 2)
            out_frame = out_frame.head(args.max_rows).reset_index(drop=True)

    out_frame.to_csv(out, index=False)
    manifest = {
        "source": str(source),
        "out": str(out),
        "rows": int(len(out_frame)),
        "unique_smiles": int(out_frame["smiles"].nunique()),
        "unique_targets": int(out_frame["target_id"].nunique()),
        "publication_year_min": float(pd.to_numeric(out_frame["publication_year"], errors="coerce").min()),
        "publication_year_max": float(pd.to_numeric(out_frame["publication_year"], errors="coerce").max()),
        "seed": int(args.seed),
        "target_cap": int(args.target_cap),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
