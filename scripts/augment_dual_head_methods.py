#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from run_mara_public_suite import metric_table, rank_average_score, rate_match_score


DERIVED_METHODS = [
    "risk_mara_dual_guard_rate",
    "risk_mara_dual_blend_rate",
    "risk_mara_rank_fusion",
]


def expand_roots(specs: list[str]) -> list[Path]:
    roots: list[Path] = []
    for spec in specs:
        matches = glob.glob(spec)
        if matches:
            roots.extend(Path(match) for match in matches)
        else:
            roots.append(Path(spec))
    return sorted({root.resolve() for root in roots})


def augment_run(run_dir: Path, overwrite_predictions: bool) -> bool:
    pred_path = run_dir / "predictions.csv"
    axis_path = run_dir / "axis_manifest.csv"
    metrics_path = run_dir / "metrics.csv"
    if not pred_path.exists() or not axis_path.exists() or not metrics_path.exists():
        return False

    pred = pd.read_csv(pred_path)
    metrics = pd.read_csv(metrics_path)
    if not {"risk_mara", "risk_scalar_full", "failure", "split_role"}.issubset(pred.columns):
        return False
    val_idx = np.flatnonzero(pred["split_role"].to_numpy() == "validation")
    test_idx = np.flatnonzero(pred["split_role"].to_numpy() == "test")
    if len(val_idx) == 0 or len(test_idx) == 0:
        return False
    y_cal = pred.iloc[val_idx]["failure"].to_numpy(dtype=int)
    mara = pred["risk_mara"].to_numpy(dtype=float)
    scalar = pred["risk_scalar_full"].to_numpy(dtype=float)
    pred["risk_mara_dual_guard_rate"] = rate_match_score(np.maximum(mara, scalar), y_cal, val_idx, power=2.0)
    pred["risk_mara_dual_blend_rate"] = rate_match_score(0.5 * mara + 0.5 * scalar, y_cal, val_idx, power=2.0)
    rank_fusion_cols = [
        "risk_mara",
        "risk_scalar_full",
        "risk_validation_knn_error",
        "risk_calibrated_tanimoto",
        "risk_rf_error_predictor",
    ]
    pred["risk_mara_rank_fusion"] = rate_match_score(rank_average_score(pred, rank_fusion_cols), y_cal, val_idx, power=1.0)

    axis = pd.read_csv(axis_path)
    add_cols = [col for col in axis.columns if col not in pred.columns]
    frame = pred.merge(axis[["row_id", *add_cols]], on="row_id", how="left") if add_cols else pred.copy()
    task_type = str(metrics["task_type"].iloc[0])
    new_metrics = metric_table(frame, test_idx, DERIVED_METHODS, task_type)
    new_metrics.insert(0, "split", metrics["split"].iloc[0])
    new_metrics.insert(0, "task_type", task_type)
    new_metrics.insert(0, "dataset_id", metrics["dataset_id"].iloc[0])

    existing = metrics[~metrics["method"].isin(DERIVED_METHODS)].copy()
    if not (run_dir / "metrics_before_dual_head.csv").exists():
        metrics.to_csv(run_dir / "metrics_before_dual_head.csv", index=False)
    pd.concat([existing, new_metrics], ignore_index=True).to_csv(metrics_path, index=False)
    if overwrite_predictions:
        pred.to_csv(pred_path, index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--overwrite-predictions", action="store_true")
    args = parser.parse_args()

    roots = expand_roots(args.roots)
    count = 0
    for root in roots:
        for run_dir in sorted(path.parent for path in root.glob("*/metrics.csv")):
            count += int(augment_run(run_dir, args.overwrite_predictions))
    print(f"augmented_runs={count}")


if __name__ == "__main__":
    main()
