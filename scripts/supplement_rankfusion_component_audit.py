#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from run_mara_public_suite import empirical_cdf_score, expected_calibration_error, rank_average_score, rate_match_score, risk_coverage


COMPONENTS = {
    "mara": "risk_mara",
    "scalar": "risk_scalar_full",
    "validation_knn": "risk_validation_knn_error",
    "tanimoto": "risk_calibrated_tanimoto",
    "rf_error": "risk_rf_error_predictor",
}


def summarize(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, block in frame.groupby(groups, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {name: value for name, value in zip(groups, keys)}
        row["cases"] = int(len(block))
        row["dataset_ids"] = int(block["dataset_id"].nunique())
        row["seeds"] = int(block["split_seed"].nunique())
        for metric in metrics:
            values = pd.to_numeric(block[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def fitted_stack(frame: pd.DataFrame, columns: list[str], val_idx: np.ndarray) -> np.ndarray:
    x = frame[columns].to_numpy(dtype=float)
    y = frame["failure"].to_numpy(dtype=int)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    model.fit(x[val_idx], y[val_idx])
    raw = model.predict_proba(x)[:, 1]
    return rate_match_score(raw, y[val_idx], val_idx, power=1.0)


def add_variants(frame: pd.DataFrame) -> list[str]:
    missing = [column for column in COMPONENTS.values() if column not in frame]
    if missing:
        raise ValueError(f"Missing rank-fusion components: {missing}")
    val_idx = np.flatnonzero(frame["split_role"].astype(str).to_numpy() == "validation")
    if not len(val_idx):
        raise ValueError("No validation rows found")
    y_val = frame.iloc[val_idx]["failure"].to_numpy(dtype=int)
    columns = list(COMPONENTS.values())
    variants: list[str] = []

    def add_rank(name: str, selected: list[str]) -> None:
        raw = rank_average_score(frame, selected)
        frame[name] = rate_match_score(raw, y_val, val_idx, power=1.0)
        variants.append(name)

    add_rank("audit_rank_full_transductive", columns)
    add_rank("audit_rank_without_mara", [column for column in columns if column != COMPONENTS["mara"]])
    frame["audit_mara_only"] = frame[COMPONENTS["mara"]].to_numpy(dtype=float)
    variants.append("audit_mara_only")
    for label, omitted in COMPONENTS.items():
        if label == "mara":
            continue
        add_rank(f"audit_rank_without_{label}", [column for column in columns if column != omitted])

    values = frame[columns].to_numpy(dtype=float)
    frame["audit_probability_mean"] = rate_match_score(values.mean(axis=1), y_val, val_idx, power=1.0)
    frame["audit_probability_median"] = rate_match_score(np.median(values, axis=1), y_val, val_idx, power=1.0)
    frame["audit_validation_logistic_stack"] = fitted_stack(frame, columns, val_idx)
    variants.extend(["audit_probability_mean", "audit_probability_median", "audit_validation_logistic_stack"])

    cdf_components = np.column_stack(
        [empirical_cdf_score(frame.iloc[val_idx][column].to_numpy(dtype=float), frame[column].to_numpy(dtype=float)) for column in columns]
    )
    inductive_raw = cdf_components.mean(axis=1)
    frame["audit_rank_full_inductive_cdf"] = rate_match_score(inductive_raw, y_val, val_idx, power=1.0)
    variants.append("audit_rank_full_inductive_cdf")
    return variants


def evaluate(frame: pd.DataFrame, test_idx: np.ndarray, variants: list[str]) -> pd.DataFrame:
    test = frame.iloc[test_idx]
    y = test["failure"].to_numpy(dtype=int)
    rows: list[dict] = []
    for method in variants:
        risk = np.clip(test[method].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        aurc, _ = risk_coverage(y, risk)
        rows.append(
            {
                "method": method,
                "failure_rate": float(y.mean()),
                "failure_auroc": float(roc_auc_score(y, risk)),
                "failure_auprc": float(average_precision_score(y, risk)),
                "risk_nll": float(log_loss(y, risk, labels=[0, 1])),
                "risk_brier": float(brier_score_loss(y, risk)),
                "risk_ece_10bin": expected_calibration_error(y, risk),
                "risk_coverage_auc": aurc,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MARA rank-fusion component contribution and inductive inference")
    parser.add_argument("--metrics", required=True, help="Frozen molecular metrics_long.csv with run_dir metadata")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.metrics)
    source["phase"] = np.where(source["split_seed"] <= 20260815, "discovery", "confirmation")
    source_metric_cols = [
        "failure_auroc",
        "failure_auprc",
        "risk_nll",
        "risk_brier",
        "risk_ece_10bin",
        "risk_coverage_auc",
    ]
    summarize(source, ["phase", "method"], source_metric_cols).to_csv(
        out / "matched_protocol_by_phase.csv", index=False
    )
    source_suite_seed = source.groupby(["suite", "split_seed", "method"], as_index=False)[source_metric_cols].mean()
    source_equal_suite = source_suite_seed.groupby(["split_seed", "method"], as_index=False)[source_metric_cols].mean()
    source_equal_suite["phase"] = np.where(source_equal_suite["split_seed"] <= 20260815, "discovery", "confirmation")
    source_equal_suite.to_csv(out / "matched_protocol_equal_suite_seed_macro.csv", index=False)
    summarize(source_equal_suite.assign(dataset_id="suite_macro"), ["method"], source_metric_cols).to_csv(
        out / "matched_protocol_equal_suite_summary.csv", index=False
    )
    summarize(source_equal_suite.assign(dataset_id="suite_macro"), ["phase", "method"], source_metric_cols).to_csv(
        out / "matched_protocol_equal_suite_by_phase.csv", index=False
    )
    run_meta = source[["run_dir", "dataset_id", "task_type", "split", "split_seed", "suite"]].drop_duplicates()
    records: list[pd.DataFrame] = []
    for meta in run_meta.itertuples(index=False):
        frame = pd.read_csv(Path(meta.run_dir) / "predictions.csv")
        variants = add_variants(frame)
        test_idx = np.flatnonzero(frame["split_role"].astype(str).to_numpy() == "test")
        metrics = evaluate(frame, test_idx, variants)
        metrics.insert(0, "suite", meta.suite)
        metrics.insert(1, "dataset_id", meta.dataset_id)
        metrics.insert(2, "split", meta.split)
        metrics.insert(3, "split_seed", int(meta.split_seed))
        metrics.insert(4, "phase", "discovery" if int(meta.split_seed) <= 20260815 else "confirmation")
        metrics.insert(5, "run_dir", meta.run_dir)
        records.append(metrics)

    long = pd.concat(records, ignore_index=True)
    metric_cols = [
        "failure_auroc",
        "failure_auprc",
        "risk_nll",
        "risk_brier",
        "risk_ece_10bin",
        "risk_coverage_auc",
    ]
    long.to_csv(out / "rankfusion_component_audit_long.csv", index=False)
    summarize(long, ["method"], metric_cols).to_csv(out / "rankfusion_component_audit_overall.csv", index=False)
    summarize(long, ["phase", "method"], metric_cols).to_csv(out / "rankfusion_component_audit_by_phase.csv", index=False)
    summarize(long, ["suite", "method"], metric_cols).to_csv(out / "rankfusion_component_audit_by_suite.csv", index=False)

    suite_seed = long.groupby(["suite", "split_seed", "method"], as_index=False)[metric_cols].mean()
    equal_suite = suite_seed.groupby(["split_seed", "method"], as_index=False)[metric_cols].mean()
    equal_suite["phase"] = np.where(equal_suite["split_seed"] <= 20260815, "discovery", "confirmation")
    equal_suite.to_csv(out / "rankfusion_equal_suite_seed_macro.csv", index=False)
    summarize(equal_suite.assign(dataset_id="suite_macro"), ["method"], metric_cols).to_csv(
        out / "rankfusion_equal_suite_summary.csv", index=False
    )
    summarize(equal_suite.assign(dataset_id="suite_macro"), ["phase", "method"], metric_cols).to_csv(
        out / "rankfusion_equal_suite_by_phase.csv", index=False
    )
    print(f"Wrote {len(long)} case-method rows from {len(run_meta)} frozen runs to {out}")


if __name__ == "__main__":
    main()
