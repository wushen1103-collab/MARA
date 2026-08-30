#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


RISK_METHODS = [
    "risk_knn_tanimoto",
    "risk_applicability_domain",
    "risk_mahalanobis",
    "risk_uncertainty_only",
    "risk_ensemble_variance",
    "risk_isotonic_uq",
    "risk_conformal_uq",
    "risk_calibrated_tanimoto",
    "risk_calibrated_ad",
    "risk_validation_knn_error",
    "risk_rf_error_predictor",
    "risk_scalar_full",
    "risk_mara",
    "risk_mara_isotonic",
    "risk_mara_rank_fusion",
]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def expand_roots(specs: list[str]) -> list[Path]:
    roots: list[Path] = []
    for spec in specs:
        path_spec = spec.split("=", 1)[1] if "=" in spec else spec
        matches = glob.glob(path_spec)
        roots.extend(Path(m) for m in matches) if matches else roots.append(Path(path_spec))
    return sorted({p.resolve() for p in roots})


def infer_suite(dataset_id: str) -> str:
    key = str(dataset_id).lower()
    if key.startswith("moleculeace_"):
        return "MoleculeACE30"
    if "chembl" in key:
        return "ChEMBL10k"
    if "bindingdb" in key:
        return "BindingDB10k"
    if key.startswith("drugood"):
        return "DrugOOD"
    if key.startswith("tdc_"):
        return "TDC"
    return "Other"


def infer_split_seed(run_dir: Path, root: Path) -> int | float:
    match = re.search(r"_seed(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    match = re.search(r"seed(\d+)", root.name)
    return int(match.group(1)) if match else np.nan


def risk_coverage_auc(y_fail: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(risk)
    coverages = np.linspace(0.05, 1.0, 96)
    curve = []
    for cov in coverages:
        n_keep = max(1, int(round(cov * len(y_fail))))
        curve.append(float(np.mean(y_fail[order[:n_keep]])))
    return float(np.trapezoid(curve, coverages) / (coverages[-1] - coverages[0]))


def summarize_case(path: Path, root: Path, methods: list[str]) -> list[dict]:
    frame = pd.read_csv(path)
    if "split_role" not in frame or "failure" not in frame:
        return []
    test = frame[frame["split_role"].astype(str).eq("test")].copy()
    if len(test) < 2 or test["failure"].nunique() < 1:
        return []
    y_fail = pd.to_numeric(test["failure"], errors="coerce").fillna(0).to_numpy(dtype=int)
    n_fail = int(y_fail.sum())
    dataset_id = str(test["dataset_id"].iloc[0]) if "dataset_id" in test else path.parent.name
    split_match = re.search(r"_(random|scaffold|assay|source|target|temporal|official|drugood)_seed", path.parent.name)
    split = split_match.group(1) if split_match else ""
    rows = []
    for method in methods:
        if method not in test.columns:
            continue
        risk = pd.to_numeric(test[method], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(risk)
        if not finite.any():
            continue
        risk = np.where(finite, risk, float(np.nanmedian(risk[finite])))
        order_low = np.argsort(risk)
        order_high = order_low[::-1]
        row = {
            "suite": infer_suite(dataset_id),
            "dataset_id": dataset_id,
            "split": split,
            "split_seed": infer_split_seed(path.parent, root),
            "run_dir": str(path.parent),
            "method": method,
            "n_test": int(len(test)),
            "failure_rate": float(np.mean(y_fail)),
            "risk_coverage_auc_5_100": risk_coverage_auc(y_fail, risk),
        }
        for coverage in [0.95, 0.90, 0.80, 0.70]:
            n_keep = max(1, int(round(coverage * len(test))))
            keep = order_low[:n_keep]
            row[f"error_at_{int(coverage * 100)}coverage"] = float(np.mean(y_fail[keep]))
            if "base_abs_error" in test.columns:
                err = pd.to_numeric(test["base_abs_error"], errors="coerce").to_numpy(dtype=float)
                row[f"mae_at_{int(coverage * 100)}coverage"] = float(np.nanmean(err[keep]))
                row[f"rmse_at_{int(coverage * 100)}coverage"] = float(np.sqrt(np.nanmean(err[keep] ** 2)))
            elif "base_loss" in test.columns:
                loss = pd.to_numeric(test["base_loss"], errors="coerce").to_numpy(dtype=float)
                row[f"loss_at_{int(coverage * 100)}coverage"] = float(np.nanmean(loss[keep]))
        for reject in [0.05, 0.10, 0.20, 0.30]:
            n_reject = max(1, int(round(reject * len(test))))
            rejected = order_high[:n_reject]
            captured = int(y_fail[rejected].sum())
            row[f"failure_capture_at_top{int(reject * 100)}risk"] = float(captured / max(1, n_fail))
            row[f"rejected_failure_rate_top{int(reject * 100)}risk"] = float(np.mean(y_fail[rejected]))
        rows.append(row)
    return rows


def numeric_summary(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, block in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n"] = int(len(block))
        row["seed_n"] = int(block["split_seed"].nunique()) if "split_seed" in block else np.nan
        row["dataset_n"] = int(block["dataset_id"].nunique()) if "dataset_id" in block else np.nan
        for metric in metrics:
            vals = pd.to_numeric(block[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_formatted(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = summary.copy()
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in out:
            out[metric] = [
                f"{m:.4f} +/- {s:.4f}" if pd.notna(m) and pd.notna(s) else ""
                for m, s in zip(out[mean_col], out[std_col])
            ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", nargs="*", default=RISK_METHODS)
    args = parser.parse_args()

    roots = expand_roots(args.roots)
    out = ensure_dir(args.out)
    rows: list[dict] = []
    for root in roots:
        for path in sorted(root.glob("*/predictions.csv")):
            rows.extend(summarize_case(path, root, args.methods))
    if not rows:
        raise SystemExit("no predictions.csv rows found")
    long = pd.DataFrame(rows)
    long.to_csv(out / "risk_coverage_long.csv", index=False)
    metric_cols = [
        c
        for c in long.columns
        if c.startswith(("risk_coverage_auc", "error_at_", "mae_at_", "rmse_at_", "loss_at_", "failure_capture_", "rejected_failure_rate_"))
    ]
    overall = numeric_summary(long, ["method"], metric_cols)
    by_suite = numeric_summary(long, ["suite", "method"], metric_cols)
    by_suite_split = numeric_summary(long, ["suite", "split", "method"], metric_cols)
    overall.to_csv(out / "risk_coverage_overall.csv", index=False)
    by_suite.to_csv(out / "risk_coverage_by_suite.csv", index=False)
    by_suite_split.to_csv(out / "risk_coverage_by_suite_split.csv", index=False)
    add_formatted(overall, metric_cols).to_csv(out / "risk_coverage_overall_formatted.csv", index=False)
    key = "failure_capture_at_top10risk_mean"
    if key in overall.columns:
        ranked = overall.sort_values(key, ascending=False)
    else:
        ranked = overall.sort_values("risk_coverage_auc_5_100_mean", ascending=True)
    try:
        ranked.to_markdown(out / "table_risk_coverage_overall.md", index=False)
    except Exception:
        ranked.to_csv(out / "table_risk_coverage_overall.md", index=False)
    print(f"Wrote {len(long)} case-method rows to {out}")


if __name__ == "__main__":
    main()
