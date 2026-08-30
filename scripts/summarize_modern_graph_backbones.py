#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXTERNAL_METHODS = [
    "max_softmax_risk",
    "predictive_entropy",
    "ensemble_disagreement",
    "uncertainty_composite",
    "representation_support",
    "distribution_frontier",
    "scalar_axis_mean",
    "validation_knn_error",
    "rf_error_predictor",
]
METRICS = [
    "base_accuracy",
    "failure_rate",
    "failure_auroc",
    "failure_auprc",
    "risk_brier",
    "risk_nll",
    "risk_ece",
    "aurc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize modern OGBN-Arxiv backbone sensitivity runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification as backbone,condition=path (for example gcn,clean=artifacts/run).",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=float, default=0.002)
    return parser.parse_args()


def mean_std(values: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()), float(values.std(ddof=1))


def main() -> None:
    args = parse_args()
    frames = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        backbone, condition = label.split(",", 1)
        frame = pd.read_csv(Path(path) / "metrics_long.csv")
        frame.insert(2, "condition", condition)
        frame.insert(3, "backbone", backbone)
        frames.append(frame)

    long = pd.concat(frames, ignore_index=True)
    key = ["dataset", "backbone", "condition", "protocol", "method"]
    summary_rows = []
    for values, block in long.groupby(key, sort=True):
        row = dict(zip(key, values))
        row["seeds"] = int(block["seed"].nunique())
        for metric in METRICS:
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std(block[metric])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    comparison_rows = []
    group_key = ["dataset", "backbone", "condition", "protocol"]
    for values, block in long.groupby(group_key, sort=True):
        means = block.groupby("method")["failure_auroc"].mean()
        candidates = means[means.index.isin(EXTERNAL_METHODS)]
        external = str(candidates.idxmax())
        selected = block[block["method"].isin(["mara_nonnegative", external])]
        pivot = selected.pivot_table(index="seed", columns="method", values="failure_auroc", aggfunc="mean").dropna()
        delta = pivot["mara_nonnegative"] - pivot[external]
        delta_mean = float(delta.mean())
        verdict = "lead" if delta_mean > args.tolerance else "lag" if delta_mean < -args.tolerance else "tie"
        row = dict(zip(group_key, values))
        row.update(
            {
                "mara_method": "mara_nonnegative",
                "external_method": external,
                "mara_selection": "fixed nonnegative diagnostic head",
                "external_selection": "descriptive test-set upper envelope across nine rerun baselines",
                "paired_seed_n": int(len(pivot)),
                "mara_auroc_mean": float(pivot["mara_nonnegative"].mean()),
                "mara_auroc_std": float(pivot["mara_nonnegative"].std(ddof=1)),
                "external_auroc_mean": float(pivot[external].mean()),
                "external_auroc_std": float(pivot[external].std(ddof=1)),
                "paired_delta_mean": delta_mean,
                "verdict_at_tolerance": verdict,
                "tolerance": args.tolerance,
                "inference_status": "descriptive; external comparator selected by test-set mean",
            }
        )
        comparison_rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "modern_graph_metrics_long.csv", index=False)
    summary.to_csv(out / "modern_graph_metrics_mean_std.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out / "modern_graph_fixed_head_comparison.csv", index=False)
    verdicts = comparison["verdict_at_tolerance"].value_counts().to_dict()
    (out / "README.txt").write_text(
        "Modern OGBN-Arxiv backbone sensitivity (five seeds per condition).\n"
        "MARA is fixed to the nonnegative diagnostic head. The external method is a descriptive\n"
        "test-set upper envelope across nine rerun baselines, so no post-selection p-value is reported.\n"
        f"Verdicts at AUROC tolerance {args.tolerance}: {verdicts}.\n",
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
