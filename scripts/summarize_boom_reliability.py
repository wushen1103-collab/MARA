#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from run_mara_public_suite import empirical_cdf_score, expected_calibration_error, rate_match_score, risk_coverage


COMPONENTS = [
    "risk_mara",
    "risk_scalar_full",
    "risk_validation_knn_error",
    "risk_calibrated_tanimoto",
    "risk_rf_error_predictor",
]


def metric_row(y: np.ndarray, risk: np.ndarray) -> dict:
    risk = np.clip(np.asarray(risk, dtype=float), 1e-6, 1.0 - 1e-6)
    aurc, _ = risk_coverage(y, risk)
    return {
        "failure_auroc": float(roc_auc_score(y, risk)),
        "failure_auprc": float(average_precision_score(y, risk)),
        "risk_nll": float(log_loss(y, risk, labels=[0, 1])),
        "risk_brier": float(brier_score_loss(y, risk)),
        "risk_ece_10bin": expected_calibration_error(y, risk),
        "risk_coverage_auc": aurc,
    }


def summarize(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    out = frame.groupby(groups, as_index=False)[metrics].agg(["mean", "std"])
    out.columns = ["_".join(str(value) for value in column if str(value)) for column in out.columns]
    counts = frame.groupby(groups, as_index=False).agg(seeds=("seed", "nunique"), cases=("seed", "size"))
    return counts.merge(out, on=groups, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MARA on BOOM 10K official property OOD splits")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pattern = re.compile(r"boom10k_(density|hof)_(id|ood)_seed(\d+)_official_seed(\d+)$")
    for root in [Path(path) for path in args.roots]:
        for run_dir in sorted(root.iterdir()):
            match = pattern.match(run_dir.name)
            if not match or not (run_dir / "predictions.csv").exists():
                continue
            target, evaluation, data_seed, run_seed = match.groups()
            if data_seed != run_seed:
                raise ValueError(f"Seed mismatch in {run_dir}")
            pred = pd.read_csv(run_dir / "predictions.csv")
            val_idx = np.flatnonzero(pred["split_role"].astype(str).to_numpy() == "validation")
            test_idx = np.flatnonzero(pred["split_role"].astype(str).to_numpy() == "test")
            y_val = pred.iloc[val_idx]["failure"].to_numpy(dtype=int)
            y_test = pred.iloc[test_idx]["failure"].to_numpy(dtype=int)
            for method in [column for column in pred.columns if column.startswith("risk_")]:
                row = metric_row(y_test, pred.iloc[test_idx][method].to_numpy(dtype=float))
                rows.append(
                    {
                        "property": target,
                        "evaluation_split": evaluation,
                        "seed": int(run_seed),
                        "method": method,
                        "failure_rate": float(y_test.mean()),
                        "base_mae": float(np.mean(np.abs(pred.iloc[test_idx]["y"] - pred.iloc[test_idx]["base_pred"]))),
                        "run_dir": str(run_dir),
                        **row,
                    }
                )
            cdf = {
                column: empirical_cdf_score(
                    pred.iloc[val_idx][column].to_numpy(dtype=float), pred[column].to_numpy(dtype=float)
                )
                for column in COMPONENTS
            }
            for name, selected in {
                "risk_mara_rank_fusion_inductive_cdf": COMPONENTS,
                "risk_rank_fusion_inductive_without_mara": COMPONENTS[1:],
            }.items():
                raw = np.column_stack([cdf[column] for column in selected]).mean(axis=1)
                risk = rate_match_score(raw, y_val, val_idx, power=1.0)
                rows.append(
                    {
                        "property": target,
                        "evaluation_split": evaluation,
                        "seed": int(run_seed),
                        "method": name,
                        "failure_rate": float(y_test.mean()),
                        "base_mae": float(np.mean(np.abs(pred.iloc[test_idx]["y"] - pred.iloc[test_idx]["base_pred"]))),
                        "run_dir": str(run_dir),
                        **metric_row(y_test, risk[test_idx]),
                    }
                )

    long = pd.DataFrame(rows)
    expected_cases = 2 * 2 * 5
    observed_cases = long[["property", "evaluation_split", "seed"]].drop_duplicates().shape[0]
    if observed_cases != expected_cases:
        raise ValueError(f"Expected {expected_cases} BOOM cases, found {observed_cases}")
    metrics = ["failure_auroc", "failure_auprc", "risk_nll", "risk_brier", "risk_ece_10bin", "risk_coverage_auc"]
    long.to_csv(out / "boom10k_reliability_long.csv", index=False)
    summary = summarize(long, ["property", "evaluation_split", "method"], metrics)
    summary.to_csv(out / "boom10k_reliability_mean_std.csv", index=False)
    selected_methods = [
        "risk_uncertainty_only",
        "risk_knn_tanimoto",
        "risk_mahalanobis",
        "risk_validation_knn_error",
        "risk_rf_error_predictor",
        "risk_scalar_full",
        "risk_mara",
        "risk_mara_rank_fusion",
        "risk_mara_rank_fusion_inductive_cdf",
        "risk_rank_fusion_inductive_without_mara",
    ]
    summary[summary["method"].isin(selected_methods)].to_csv(out / "boom10k_selected_mean_std.csv", index=False)
    print(f"Wrote {len(long)} method-case rows across {observed_cases} BOOM tasks to {out}")


if __name__ == "__main__":
    main()
