#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BOOM 10K official train/ID/OOD reliability splits")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260831, 20260832, 20260833, 20260834, 20260835])
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source)
    records: list[dict] = []
    for target in ["density", "hof"]:
        train_flag = f"{target}_train"
        id_flag = f"{target}_iid"
        ood_flag = f"{target}_ood"
        for seed in args.seeds:
            train_rows = raw.index[raw[train_flag].astype(int) == 1].to_numpy()
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(train_rows)
            validation_n = max(20, int(round(len(shuffled) * args.validation_fraction)))
            validation_rows = set(int(index) for index in shuffled[:validation_n])
            for evaluation, eval_flag in [("id", id_flag), ("ood", ood_flag)]:
                selected = raw[(raw[train_flag].astype(int) == 1) | (raw[eval_flag].astype(int) == 1)].copy()
                selected["preset_split_role"] = "proper_train"
                selected.loc[selected.index.isin(validation_rows), "preset_split_role"] = "validation"
                selected.loc[selected[eval_flag].astype(int) == 1, "preset_split_role"] = "test"
                frame = selected[["smiles", target, "preset_split_role"]].rename(columns={target: "y"})
                frame["boom_property"] = target
                frame["boom_evaluation_split"] = evaluation
                frame["source_id"] = frame["preset_split_role"]
                path = out / f"boom10k_{target}_{evaluation}_seed{seed}.csv"
                frame.to_csv(path, index=False)
                counts = frame["preset_split_role"].value_counts().to_dict()
                if min(counts.get(role, 0) for role in ["proper_train", "validation", "test"]) < 20:
                    raise ValueError(f"Invalid split counts for {path}: {counts}")
                records.append(
                    {
                        "file": str(path),
                        "property": target,
                        "evaluation_split": evaluation,
                        "seed": seed,
                        **{f"n_{role}": int(counts[role]) for role in ["proper_train", "validation", "test"]},
                    }
                )
    manifest = pd.DataFrame(records)
    manifest.to_csv(out / "boom10k_split_manifest.csv", index=False)
    (out / "boom10k_split_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
                "validation_fraction_within_official_train": args.validation_fraction,
                "seeds": args.seeds,
                "files": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
