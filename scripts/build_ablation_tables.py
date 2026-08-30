#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_metrics(root: Path, label: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/metrics.csv")):
        frame = pd.read_csv(path)
        frame["ablation"] = label
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_loc(root: Path, label: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/stress_localization.csv")):
        frame = pd.read_csv(path)
        frame["ablation"] = label
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, help="label=path entries")
    parser.add_argument("--out", default="artifacts/tables_ablation")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    loc_frames = []
    for item in args.roots:
        if "=" not in item:
            raise SystemExit(f"expected label=path, got {item}")
        label, path = item.split("=", 1)
        root = Path(path)
        metric_frames.append(load_metrics(root, label))
        loc_frames.append(load_loc(root, label))
    metrics = pd.concat([f for f in metric_frames if not f.empty], ignore_index=True)
    loc = pd.concat([f for f in loc_frames if not f.empty], ignore_index=True)
    metrics.to_csv(out / "ablation_metrics_long.csv", index=False)
    loc.to_csv(out / "ablation_localization_long.csv", index=False)

    mara = metrics[metrics["method"] == "risk_mara"].copy()
    keep = [
        "ablation",
        "dataset_id",
        "split",
        "failure_auroc",
        "failure_auprc",
        "risk_ece_10bin",
        "risk_coverage_auc",
        "selective_risk_at_80",
    ]
    mara[keep].to_csv(out / "ablation_mara_metrics.csv", index=False)
    summary = mara.groupby("ablation")[[
        "failure_auroc",
        "failure_auprc",
        "risk_ece_10bin",
        "risk_coverage_auc",
        "selective_risk_at_80",
    ]].mean().reset_index()
    if not loc.empty:
        macro = loc[loc["slice"] == "macro"].groupby("ablation")["macro_f1"].mean().reset_index()
        summary = summary.merge(macro, on="ablation", how="left")
    summary.to_csv(out / "ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
