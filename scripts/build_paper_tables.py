#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def to_markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="artifacts/public_v0")
    parser.add_argument("--out", default="artifacts/tables")
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    out = ensure_dir(args.out)
    metric_files = sorted(root.glob("*/metrics.csv"))
    loc_files = sorted(root.glob("*/stress_localization.csv"))
    if metric_files:
        metrics = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True)
        metrics.to_csv(out / "main_reliability_metrics.csv", index=False)
        slim_cols = [
            "dataset_id",
            "split",
            "method",
            "failure_auroc",
            "failure_auprc",
            "risk_ece_10bin",
            "risk_coverage_auc",
            "selective_risk_at_80",
            "selective_risk_at_90",
            "worst_slice_risk",
        ]
        slim = metrics[[c for c in slim_cols if c in metrics.columns]].copy()
        (out / "main_reliability_metrics.md").write_text(to_markdown_table(slim), encoding="utf-8")
        pivot = metrics.pivot_table(
            index=["dataset_id", "split"],
            columns="method",
            values=["failure_auroc", "failure_auprc", "risk_coverage_auc"],
            aggfunc="mean",
        )
        pivot.to_csv(out / "main_metric_pivot.csv")
    if loc_files:
        loc = pd.concat([pd.read_csv(p) for p in loc_files], ignore_index=True)
        loc.to_csv(out / "stress_localization.csv", index=False)
        (out / "stress_localization.md").write_text(to_markdown_table(loc), encoding="utf-8")
    print(f"wrote tables to {out}")


if __name__ == "__main__":
    main()
