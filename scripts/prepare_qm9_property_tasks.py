#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="QM9 CSV with smiles plus property columns")
    parser.add_argument("--out-dir", default="data/processed/qm9")
    parser.add_argument("--properties", nargs="+", default=["homo", "lumo", "gap"])
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(src)
    if "smiles" not in frame.columns:
        raise SystemExit("input must contain a smiles column")
    frame = frame.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    if args.sample_size > 0 and len(frame) > args.sample_size:
        frame = frame.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)

    manifest = []
    for prop in args.properties:
        if prop not in frame.columns:
            raise SystemExit(f"missing property column: {prop}")
        out = frame[["smiles", prop]].rename(columns={prop: "y"}).dropna().copy()
        out["source_dataset"] = "QM9"
        out["property"] = prop
        out_path = out_dir / f"qm9_{prop}_sample{len(out)}.csv"
        out.to_csv(out_path, index=False)
        manifest.append({"property": prop, "rows": len(out), "path": str(out_path)})

    pd.DataFrame(manifest).to_csv(out_dir / "qm9_property_manifest.csv", index=False)
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
