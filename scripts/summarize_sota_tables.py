#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HIGHER_IS_BETTER = {"failure_auroc", "failure_auprc"}
LOWER_IS_BETTER = {"risk_nll", "risk_brier", "risk_ece_10bin", "risk_coverage_auc", "selective_risk_at_80", "selective_risk_at_90"}


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_metrics(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/metrics.csv")):
        frame = pd.read_csv(path)
        frame["run_dir"] = str(path.parent)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no metrics.csv under {root}")
    return pd.concat(frames, ignore_index=True)


def compare_metric(mara: float, other: float, metric: str) -> tuple[float, bool]:
    if pd.isna(mara) or pd.isna(other):
        return np.nan, False
    delta = mara - other if metric in HIGHER_IS_BETTER else other - mara
    return float(delta), bool(delta > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--primary", default="risk_mara")
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    out = ensure_dir(args.out)
    metrics = load_metrics(root)
    metrics.to_csv(out / "metrics_long.csv", index=False)

    metric_cols = [c for c in [*HIGHER_IS_BETTER, *LOWER_IS_BETTER] if c in metrics.columns]
    method_summary = (
        metrics.groupby("method")[metric_cols]
        .agg(["mean", "std", "median"])
        .sort_index(axis=1)
    )
    method_summary.to_csv(out / "method_metric_summary.csv")

    baselines = sorted(m for m in metrics["method"].unique() if m != args.primary)
    rows = []
    key_cols = ["dataset_id", "split", "task_type"]
    primary = metrics[metrics["method"] == args.primary][key_cols + metric_cols].copy()
    for baseline in baselines:
        other = metrics[metrics["method"] == baseline][key_cols + metric_cols].copy()
        merged = primary.merge(other, on=key_cols, suffixes=("_mara", "_baseline"))
        for metric in metric_cols:
            deltas = []
            wins = []
            for _, row in merged.iterrows():
                delta, win = compare_metric(row[f"{metric}_mara"], row[f"{metric}_baseline"], metric)
                deltas.append(delta)
                wins.append(win)
            clean = pd.Series(deltas, dtype=float).dropna()
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "n": int(len(clean)),
                    "mean_advantage": float(clean.mean()) if len(clean) else np.nan,
                    "median_advantage": float(clean.median()) if len(clean) else np.nan,
                    "win_rate": float(np.mean(wins)) if wins else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(out / "mara_vs_baselines.csv", index=False)

    best_rows = []
    for keys, block in metrics.groupby(key_cols):
        mara_row = block[block["method"] == args.primary]
        if mara_row.empty:
            continue
        mara_row = mara_row.iloc[0]
        competitors = block[block["method"] != args.primary]
        for metric in metric_cols:
            comp_values = competitors[["method", metric]].dropna()
            if comp_values.empty or pd.isna(mara_row[metric]):
                continue
            if metric in HIGHER_IS_BETTER:
                best = comp_values.sort_values(metric, ascending=False).iloc[0]
                advantage = float(mara_row[metric] - best[metric])
            else:
                best = comp_values.sort_values(metric, ascending=True).iloc[0]
                advantage = float(best[metric] - mara_row[metric])
            best_rows.append(
                {
                    "dataset_id": keys[0],
                    "split": keys[1],
                    "task_type": keys[2],
                    "metric": metric,
                    "mara_value": float(mara_row[metric]),
                    "best_competitor": best["method"],
                    "best_competitor_value": float(best[metric]),
                    "mara_advantage_vs_best": advantage,
                    "mara_is_best": bool(advantage >= 0),
                }
            )
    best = pd.DataFrame(best_rows)
    best.to_csv(out / "mara_vs_best_competitor.csv", index=False)
    if not best.empty:
        readiness = (
            best.groupby("metric")
            .agg(
                n=("mara_is_best", "size"),
                best_rate=("mara_is_best", "mean"),
                mean_advantage_vs_best=("mara_advantage_vs_best", "mean"),
                median_advantage_vs_best=("mara_advantage_vs_best", "median"),
            )
            .reset_index()
        )
        readiness.to_csv(out / "claim_readiness.csv", index=False)
        print(readiness.to_string(index=False))


if __name__ == "__main__":
    main()
