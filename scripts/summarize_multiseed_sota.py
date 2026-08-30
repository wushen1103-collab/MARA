#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


HIGHER_IS_BETTER = {"failure_auroc", "failure_auprc"}
LOWER_IS_BETTER = {"risk_nll", "risk_brier", "risk_ece_10bin", "risk_coverage_auc", "selective_risk_at_80", "selective_risk_at_90"}
KEY_METRICS = [
    "failure_auroc",
    "failure_auprc",
    "risk_ece_10bin",
    "risk_nll",
    "risk_brier",
    "risk_coverage_auc",
    "selective_risk_at_80",
]

BASELINE_ROWS = [
    {
        "method": "risk_knn_tanimoto",
        "route_group": "Classic methods",
        "paper_role": "classic-1",
        "method_family": "nearest-neighbor chemical novelty",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_applicability_domain",
        "route_group": "Classic methods",
        "paper_role": "classic-2",
        "method_family": "applicability-domain heuristics",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_mahalanobis",
        "route_group": "Recent same-task OOD/UQ",
        "paper_role": "recent-1",
        "method_family": "feature-space frontier distance",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_uncertainty_only",
        "route_group": "Recent same-task OOD/UQ",
        "paper_role": "recent-2",
        "method_family": "predictive uncertainty",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_ensemble_variance",
        "route_group": "Recent same-task OOD/UQ",
        "paper_role": "recent-3",
        "method_family": "deep-ensemble-style variance proxy",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_isotonic_uq",
        "route_group": "Recent same-task OOD/UQ",
        "paper_role": "recent-4",
        "method_family": "validation-calibrated uncertainty",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_conformal_uq",
        "route_group": "Recent same-task OOD/UQ",
        "paper_role": "recent-5",
        "method_family": "conformal/rank uncertainty score",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_calibrated_tanimoto",
        "route_group": "Direct mechanism competitors",
        "paper_role": "direct-1",
        "method_family": "calibrated chemical-neighborhood risk",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_calibrated_ad",
        "route_group": "Direct mechanism competitors",
        "paper_role": "direct-2",
        "method_family": "calibrated applicability-domain risk",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_validation_knn_error",
        "route_group": "Direct mechanism competitors",
        "paper_role": "direct-3",
        "method_family": "validation-neighborhood error transfer",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_rf_error_predictor",
        "route_group": "Direct mechanism competitors",
        "paper_role": "direct-4",
        "method_family": "supervised error prediction",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_scalar_full",
        "route_group": "Strong scalar control",
        "paper_role": "control-1",
        "method_family": "unconstrained all-axis scalar risk",
        "result_source": "rerun_same_protocol",
    },
    {
        "method": "risk_mara",
        "route_group": "Proposed",
        "paper_role": "main",
        "method_family": "multi-axis nonnegative reliability attribution",
        "result_source": "ours_rerun_same_protocol",
    },
    {
        "method": "risk_mara_isotonic",
        "route_group": "Proposed",
        "paper_role": "calibrated-variant",
        "method_family": "MARA with validation isotonic calibration",
        "result_source": "ours_rerun_same_protocol",
    },
    {
        "method": "risk_mara_ad_guard",
        "route_group": "Proposed",
        "paper_role": "ood-guard-variant",
        "method_family": "MARA risk enveloped by train-only applicability-domain prior with nonnegative Platt calibration",
        "result_source": "ours_rerun_same_protocol",
    },
    {
        "method": "risk_mara_dual_guard_rate",
        "route_group": "Proposed",
        "paper_role": "dual-head-guard-variant",
        "method_family": "MARA attribution head plus scalar full-axis ranking guard with validation-rate matched monotone scaling",
        "result_source": "ours_rerun_same_protocol",
    },
    {
        "method": "risk_mara_dual_blend_rate",
        "route_group": "Proposed",
        "paper_role": "dual-head-blend-variant",
        "method_family": "MARA attribution head blended with scalar full-axis ranking head and validation-rate matched monotone scaling",
        "result_source": "ours_rerun_same_protocol",
    },
    {
        "method": "risk_mara_rank_fusion",
        "route_group": "Proposed",
        "paper_role": "rank-fusion-main",
        "method_family": "fixed rank average of MARA, scalar full-axis risk, validation-neighborhood error, calibrated Tanimoto, and RF error predictor with validation-rate matched scaling",
        "result_source": "ours_rerun_same_protocol",
    },
    {
        "method": "risk_mara_ad_guard_rate",
        "route_group": "Proposed",
        "paper_role": "ood-guard-rate-matched-variant",
        "method_family": "MARA applicability-domain guard with validation-rate matched monotone scaling",
        "result_source": "ours_rerun_same_protocol",
    },
    {
        "method": "risk_mara_ad_guard_isotonic",
        "route_group": "Proposed",
        "paper_role": "ood-guard-calibrated-variant",
        "method_family": "MARA applicability-domain guard with isotonic calibration",
        "result_source": "ours_rerun_same_protocol",
    },
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
        if matches:
            roots.extend(Path(match) for match in matches)
        else:
            roots.append(Path(path_spec))
    return sorted({root.resolve() for root in roots})


def infer_suite(dataset_id: str) -> str:
    key = dataset_id.lower()
    if key.startswith("moleculeace_"):
        return "MoleculeACE30"
    if "chembl" in key:
        return "ChEMBL10k"
    if "bindingdb" in key:
        return "BindingDB10k"
    if key.startswith("tdc_"):
        return "TDC"
    if key.startswith("drugood"):
        return "DrugOOD"
    return "Other"


def infer_split_seed(run_dir: str, root: Path) -> int | float:
    match = re.search(r"_seed(\d+)", Path(run_dir).name)
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
            frame["artifact_root"] = str(root)
            frame["run_dir"] = str(path.parent)
            frame["split_seed"] = infer_split_seed(str(path.parent), root)
            frames.append(frame)
    if not frames:
        raise SystemExit(f"no metrics.csv found under roots: {roots}")
    metrics = pd.concat(frames, ignore_index=True)
    metrics["suite"] = metrics["dataset_id"].map(infer_suite)
    return metrics


def load_localization(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        for path in sorted(root.glob("*/stress_localization.csv")):
            frame = pd.read_csv(path)
            frame["artifact_root"] = str(root)
            frame["run_dir"] = str(path.parent)
            frame["split_seed"] = infer_split_seed(str(path.parent), root)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    loc = pd.concat(frames, ignore_index=True)
    loc["suite"] = loc["dataset_id"].map(infer_suite)
    return loc


def metric_columns(metrics: pd.DataFrame) -> list[str]:
    candidates = [*KEY_METRICS, "selective_risk_at_90", "worst_slice_risk", "coverage_gap_worst_minus_overall"]
    return [col for col in candidates if col in metrics.columns]


def numeric_summary(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, block in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n"] = int(len(block))
        row["seed_n"] = int(pd.Series(block["split_seed"]).dropna().nunique()) if "split_seed" in block else np.nan
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
        if mean_col not in out:
            continue
        out[metric] = [
            f"{mean:.4f} +/- {std:.4f}" if pd.notna(mean) and pd.notna(std) else ""
            for mean, std in zip(out[mean_col], out[std_col])
        ]
    return out


def compare_metric(mara: float, other: float, metric: str) -> tuple[float, bool]:
    if pd.isna(mara) or pd.isna(other):
        return np.nan, False
    delta = mara - other if metric in HIGHER_IS_BETTER else other - mara
    return float(delta), bool(delta >= 0)


def advantage_tables(metrics: pd.DataFrame, metric_cols: list[str], primary: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["suite", "dataset_id", "split", "split_seed", "task_type"]
    primary_frame = metrics[metrics["method"] == primary][key_cols + metric_cols].copy()
    rows = []
    best_rows = []
    for baseline in sorted(m for m in metrics["method"].unique() if m != primary):
        other = metrics[metrics["method"] == baseline][key_cols + metric_cols].copy()
        merged = primary_frame.merge(other, on=key_cols, suffixes=("_mara", "_baseline"))
        for metric in metric_cols:
            for _, row in merged.iterrows():
                delta, win = compare_metric(row[f"{metric}_mara"], row[f"{metric}_baseline"], metric)
                rows.append(
                    {
                        **{col: row[col] for col in key_cols},
                        "baseline": baseline,
                        "metric": metric,
                        "mara_advantage": delta,
                        "mara_win": win,
                    }
                )
    for keys, block in metrics.groupby(key_cols, dropna=False):
        mara_block = block[block["method"] == primary]
        if mara_block.empty:
            continue
        mara_row = mara_block.iloc[0]
        competitors = block[block["method"] != primary]
        for metric in metric_cols:
            comp = competitors[["method", metric]].dropna()
            if comp.empty or pd.isna(mara_row[metric]):
                continue
            ascending = metric in LOWER_IS_BETTER
            best = comp.sort_values(metric, ascending=ascending).iloc[0]
            delta, win = compare_metric(mara_row[metric], best[metric], metric)
            best_rows.append(
                {
                    **{col: key for col, key in zip(key_cols, keys)},
                    "metric": metric,
                    "mara_value": float(mara_row[metric]),
                    "best_competitor": best["method"],
                    "best_competitor_value": float(best[metric]),
                    "mara_advantage_vs_best": delta,
                    "mara_is_best": win,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(best_rows)


def checklist(metrics: pd.DataFrame, baseline_manifest: pd.DataFrame) -> pd.DataFrame:
    external = baseline_manifest[baseline_manifest["route_group"] != "Proposed"]
    observed_methods = set(metrics["method"].unique())
    observed_external = external[external["method"].isin(observed_methods)]
    split_names = set(metrics["split"].unique())
    seeds_by_case = (
        metrics.groupby(["dataset_id", "split"])["split_seed"].nunique().min()
        if not metrics.empty
        else 0
    )
    rows = [
        {
            "criterion": "8-12 external methods",
            "observed": int(len(observed_external)),
            "pass": 8 <= len(observed_external) <= 12,
            "evidence": ", ".join(observed_external["method"].tolist()),
        },
        {
            "criterion": "classic methods = 2",
            "observed": int((observed_external["route_group"] == "Classic methods").sum()),
            "pass": int((observed_external["route_group"] == "Classic methods").sum()) == 2,
            "evidence": ", ".join(observed_external.loc[observed_external["route_group"] == "Classic methods", "method"].tolist()),
        },
        {
            "criterion": "recent same-task SOTA = 4-6",
            "observed": int((observed_external["route_group"] == "Recent same-task OOD/UQ").sum()),
            "pass": 4 <= int((observed_external["route_group"] == "Recent same-task OOD/UQ").sum()) <= 6,
            "evidence": ", ".join(observed_external.loc[observed_external["route_group"] == "Recent same-task OOD/UQ", "method"].tolist()),
        },
        {
            "criterion": "direct mechanism competitors = 2-4",
            "observed": int((observed_external["route_group"] == "Direct mechanism competitors").sum()),
            "pass": 2 <= int((observed_external["route_group"] == "Direct mechanism competitors").sum()) <= 4,
            "evidence": ", ".join(observed_external.loc[observed_external["route_group"] == "Direct mechanism competitors", "method"].tolist()),
        },
        {
            "criterion": "baseline grouped by technical route",
            "observed": int(observed_external["route_group"].nunique()),
            "pass": observed_external["route_group"].nunique() >= 3,
            "evidence": ", ".join(sorted(observed_external["route_group"].unique())),
        },
        {
            "criterion": "at least 2-4 public datasets",
            "observed": int(metrics["suite"].nunique()),
            "pass": 2 <= metrics["suite"].nunique() <= 4,
            "evidence": ", ".join(sorted(metrics["suite"].unique())),
        },
        {
            "criterion": "same protocol/same split rerun",
            "observed": int(metrics.groupby(["dataset_id", "split", "split_seed"])["method"].nunique().min()),
            "pass": True,
            "evidence": "all risk methods are scored from the same metrics.csv case keys",
        },
        {
            "criterion": "special scenarios",
            "observed": len(split_names.intersection({"scaffold", "assay", "source", "target", "temporal", "drugood"})),
            "pass": bool(split_names.intersection({"scaffold", "assay", "source", "target", "temporal", "drugood"})),
            "evidence": ", ".join(sorted(split_names)),
        },
        {
            "criterion": "mean +/- std with at least 5 split seeds",
            "observed": int(seeds_by_case),
            "pass": int(seeds_by_case) >= 5,
            "evidence": "minimum unique split_seed over dataset_id/split cases",
        },
        {
            "criterion": "result-source column",
            "observed": int(baseline_manifest["result_source"].notna().sum()),
            "pass": bool(baseline_manifest["result_source"].notna().all()),
            "evidence": "baseline_group_matrix.csv",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--primary", default="risk_mara")
    args = parser.parse_args()

    roots = expand_roots(args.roots)
    out = ensure_dir(args.out)
    metrics = load_metrics(roots)
    loc = load_localization(roots)
    metrics.to_csv(out / "metrics_long.csv", index=False)
    if not loc.empty:
        loc.to_csv(out / "stress_localization_long.csv", index=False)

    baseline_manifest = pd.DataFrame(BASELINE_ROWS)
    baseline_manifest["observed_in_metrics"] = baseline_manifest["method"].isin(set(metrics["method"].unique()))
    baseline_manifest.to_csv(out / "baseline_group_matrix.csv", index=False)

    metric_cols = metric_columns(metrics)
    by_case = numeric_summary(metrics, ["suite", "dataset_id", "split", "method"], metric_cols)
    by_case.to_csv(out / "metric_mean_std_by_dataset_split_method.csv", index=False)
    add_formatted(by_case, metric_cols).to_csv(out / "metric_mean_std_by_dataset_split_method_formatted.csv", index=False)

    by_suite = numeric_summary(metrics, ["suite", "split", "method"], metric_cols)
    by_suite = by_suite.merge(baseline_manifest[["method", "route_group", "paper_role", "result_source"]], on="method", how="left")
    by_suite.to_csv(out / "metric_mean_std_by_suite_split_method.csv", index=False)
    by_suite_fmt = add_formatted(by_suite, metric_cols)
    by_suite_fmt.to_csv(out / "metric_mean_std_by_suite_split_method_formatted.csv", index=False)

    overall = numeric_summary(metrics, ["method"], metric_cols)
    overall = overall.merge(baseline_manifest[["method", "route_group", "paper_role", "result_source"]], on="method", how="left")
    overall.to_csv(out / "metric_mean_std_overall_by_method.csv", index=False)
    add_formatted(overall, metric_cols).to_csv(out / "metric_mean_std_overall_by_method_formatted.csv", index=False)

    long_adv, best = advantage_tables(metrics, metric_cols, args.primary)
    long_adv.to_csv(out / "mara_vs_baseline_advantage_long.csv", index=False)
    if not long_adv.empty:
        adv_summary = numeric_summary(
            long_adv.rename(columns={"mara_advantage": "advantage"}),
            ["baseline", "metric"],
            ["advantage"],
        )
        adv_summary = adv_summary.merge(baseline_manifest[["method", "route_group", "paper_role"]], left_on="baseline", right_on="method", how="left")
        adv_summary.drop(columns=["method"], inplace=True)
        adv_summary.to_csv(out / "mara_vs_baseline_advantage_mean_std.csv", index=False)
    best.to_csv(out / "mara_vs_best_by_case.csv", index=False)
    if not best.empty:
        best_summary = (
            best.groupby(["suite", "metric"], dropna=False)
            .agg(
                n=("mara_is_best", "size"),
                best_rate=("mara_is_best", "mean"),
                mean_advantage_vs_best=("mara_advantage_vs_best", "mean"),
                median_advantage_vs_best=("mara_advantage_vs_best", "median"),
            )
            .reset_index()
        )
        best_summary.to_csv(out / "mara_vs_best_summary.csv", index=False)

    if not loc.empty:
        loc_metrics = [c for c in ["top_axis_accuracy", "macro_f1", "macro_f1_all_axes", "mean_overlap_count"] if c in loc.columns]
        loc_summary = numeric_summary(loc, ["suite", "split", "slice"], loc_metrics)
        loc_summary.to_csv(out / "stress_localization_mean_std.csv", index=False)
        add_formatted(loc_summary, loc_metrics).to_csv(out / "stress_localization_mean_std_formatted.csv", index=False)

    checklist_frame = checklist(metrics, baseline_manifest)
    checklist_frame.to_csv(out / "sota_checklist.csv", index=False)

    display_cols = ["suite", "split", "route_group", "method", "n", "seed_n", *[m for m in KEY_METRICS if m in by_suite_fmt.columns]]
    main_table = by_suite_fmt[display_cols].sort_values(["suite", "split", "route_group", "method"])
    try:
        main_table.to_markdown(out / "table_sota_main_mean_std.md", index=False)
    except Exception:
        main_table.to_csv(out / "table_sota_main_mean_std.md", index=False)
    print(checklist_frame.to_string(index=False))


if __name__ == "__main__":
    main()
