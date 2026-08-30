#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


KEY_METRICS = [
    "failure_auroc",
    "failure_auprc",
    "risk_nll",
    "risk_brier",
    "risk_coverage_auc",
    "selective_risk_at_80",
    "selective_risk_at_90",
]
MAIN_METHODS = [
    "risk_uncertainty_only",
    "risk_ensemble_variance",
    "risk_calibrated_tanimoto",
    "risk_validation_knn_error",
    "risk_rf_error_predictor",
    "risk_scalar_full",
    "risk_mara",
    "risk_mara_rank_fusion",
]
FOCUS_METHOD = "risk_mara_rank_fusion"
FOCUS_COMPARATORS = [
    "risk_calibrated_tanimoto",
    "risk_validation_knn_error",
    "risk_rf_error_predictor",
    "risk_scalar_full",
    "risk_mara",
]
HIGHER_IS_BETTER = {"failure_auroc", "failure_auprc"}


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
        return "MoleculeACE"
    if "chembl" in key:
        return "ChEMBL10k"
    if "bindingdb" in key:
        return "BindingDB10k"
    if key.startswith("qm"):
        return "QM"
    return "Other"


def infer_split_seed(run_dir: Path, root: Path) -> int | float:
    match = re.search(r"_seed(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    manifest = root / "suite_manifest.json"
    if manifest.exists():
        try:
            return int(json.loads(manifest.read_text(encoding="utf-8")).get("split_seed"))
        except Exception:
            return np.nan
    return np.nan


def load_metrics(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        for path in sorted(root.glob("*/metrics.csv")):
            frame = pd.read_csv(path)
            if "base_predictor" not in frame:
                frame["base_predictor"] = "ecfp_xgb"
            if "split_seed" not in frame:
                frame["split_seed"] = infer_split_seed(path.parent, root)
            frame["artifact_root"] = str(root)
            frame["run_dir"] = str(path.parent)
            frame["suite"] = frame["dataset_id"].map(infer_suite)
            frames.append(frame)
    if not frames:
        raise SystemExit(f"no metrics.csv found under {roots}")
    return pd.concat(frames, ignore_index=True)


def numeric_summary(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, block in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n"] = int(len(block))
        row["seed_n"] = int(block["split_seed"].nunique()) if "split_seed" in block else np.nan
        row["dataset_n"] = int(block["dataset_id"].nunique()) if "dataset_id" in block else np.nan
        for metric in metrics:
            vals = pd.to_numeric(block[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def seed_macro_summary(
    frame: pd.DataFrame, group_cols: list[str], metrics: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_cols = [*group_cols, "split_seed"]
    aggregations = {metric: (metric, "mean") for metric in metrics}
    seed_level = (
        frame.groupby(seed_cols, dropna=False)
        .agg(case_n=("method", "size"), dataset_n=("dataset_id", "nunique"), **aggregations)
        .reset_index()
    )
    rows = []
    for keys, block in seed_level.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["seed_n"] = int(block["split_seed"].nunique())
        row["case_n_total"] = int(block["case_n"].sum())
        row["dataset_n"] = int(block["dataset_n"].max())
        for metric in metrics:
            values = pd.to_numeric(block[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else np.nan
        rows.append(row)
    return seed_level, pd.DataFrame(rows)


def add_formatted(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = summary.copy()
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in out:
            continue
        out[metric] = [
            f"{m:.4f} +/- {s:.4f}" if pd.notna(m) and pd.notna(s) else ""
            for m, s in zip(out[mean_col], out[std_col])
        ]
    return out


def paired_focus_comparison(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    case_cols = ["dataset_id", "split", "split_seed"]
    rows = []
    for base_predictor, base_block in frame.groupby("base_predictor", dropna=False):
        for comparator in FOCUS_COMPARATORS:
            pair_block = base_block[base_block["method"].isin([FOCUS_METHOD, comparator])]
            for metric in metrics:
                paired = pair_block.pivot_table(index=case_cols, columns="method", values=metric, aggfunc="mean")
                if FOCUS_METHOD not in paired or comparator not in paired:
                    continue
                paired = paired[[FOCUS_METHOD, comparator]].dropna()
                proposed = paired[FOCUS_METHOD].astype(float)
                baseline = paired[comparator].astype(float)
                gain = proposed - baseline if metric in HIGHER_IS_BETTER else baseline - proposed
                scale = baseline.abs().replace(0.0, np.nan)
                relative_gain = 100.0 * gain / scale
                tol = 1e-12
                rows.append(
                    {
                        "base_predictor": base_predictor,
                        "metric": metric,
                        "comparator": comparator,
                        "n_paired": int(len(paired)),
                        "proposed_mean": float(proposed.mean()),
                        "comparator_mean": float(baseline.mean()),
                        "directional_gain_mean": float(gain.mean()),
                        "directional_gain_std": float(gain.std(ddof=1)) if len(gain) > 1 else 0.0,
                        "relative_gain_pct_mean": float(relative_gain.mean()),
                        "wins": int((gain > tol).sum()),
                        "ties": int((gain.abs() <= tol).sum()),
                        "losses": int((gain < -tol).sum()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", nargs="*", default=MAIN_METHODS)
    args = parser.parse_args()
    out = ensure_dir(args.out)
    metrics = load_metrics(expand_roots(args.roots))
    dedup_cols = [col for col in ["base_predictor", "dataset_id", "split", "split_seed", "method"] if col in metrics.columns]
    if dedup_cols:
        metrics = (
            metrics.sort_values(["artifact_root", "run_dir"])
            .drop_duplicates(subset=dedup_cols, keep="last")
            .reset_index(drop=True)
        )
    metrics = metrics[metrics["method"].isin(args.methods)].copy()
    metric_cols = [m for m in KEY_METRICS if m in metrics.columns]
    metrics.to_csv(out / "multibackbone_metrics_long.csv", index=False)
    overall = numeric_summary(metrics, ["base_predictor", "method"], metric_cols)
    by_suite = numeric_summary(metrics, ["base_predictor", "suite", "method"], metric_cols)
    by_suite_split = numeric_summary(metrics, ["base_predictor", "suite", "split", "method"], metric_cols)
    seed_macro, overall_seed_summary = seed_macro_summary(metrics, ["base_predictor", "method"], metric_cols)
    suite_seed_macro, suite_seed_summary = seed_macro_summary(
        metrics, ["base_predictor", "suite", "method"], metric_cols
    )
    overall.to_csv(out / "multibackbone_overall.csv", index=False)
    by_suite.to_csv(out / "multibackbone_by_suite.csv", index=False)
    by_suite_split.to_csv(out / "multibackbone_by_suite_split.csv", index=False)
    add_formatted(overall, metric_cols).to_csv(out / "multibackbone_overall_formatted.csv", index=False)
    seed_macro.to_csv(out / "multibackbone_seed_macro.csv", index=False)
    overall_seed_summary.to_csv(out / "multibackbone_overall_seed_mean_std.csv", index=False)
    add_formatted(overall_seed_summary, metric_cols).to_csv(
        out / "multibackbone_overall_seed_mean_std_formatted.csv", index=False
    )
    suite_seed_macro.to_csv(out / "multibackbone_suite_seed_macro.csv", index=False)
    suite_seed_summary.to_csv(out / "multibackbone_suite_seed_mean_std.csv", index=False)
    paired = paired_focus_comparison(metrics, metric_cols)
    paired.to_csv(out / "multibackbone_focus_paired_wtl.csv", index=False)
    try:
        overall.sort_values(["base_predictor", "failure_auroc_mean"], ascending=[True, False]).to_markdown(out / "table_multibackbone_overall.md", index=False)
        by_suite_split.sort_values(["base_predictor", "suite", "split", "failure_auroc_mean"], ascending=[True, True, True, False]).to_markdown(
            out / "table_multibackbone_by_suite_split.md", index=False
        )
    except Exception:
        pass
    print(f"Wrote multibackbone summary with {len(metrics)} rows to {out}")


if __name__ == "__main__":
    main()
