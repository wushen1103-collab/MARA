#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PRIMARY_METHODS = ("risk_mara", "risk_mara_rank_fusion")
PRIMARY_METRICS = (
    "failure_auroc",
    "failure_auprc",
    "risk_ece_10bin",
    "risk_nll",
    "risk_brier",
    "risk_coverage_auc",
    "selective_risk_at_80",
)
LOWER_IS_BETTER = {
    "risk_ece_10bin",
    "risk_nll",
    "risk_brier",
    "risk_coverage_auc",
    "selective_risk_at_80",
}


def infer_label(path: Path) -> str:
    name = path.name
    match = re.search(r"abl_full_(.+?)_seed\d+", name)
    if match:
        return match.group(1)
    return name


def infer_split_seed(path: Path) -> int | float:
    for text in (path.name, path.parent.name):
        match = re.search(r"_seed(\d+)", text)
        if match:
            return int(match.group(1))
    return float("nan")


def load_case_files(root: Path, filename: str, label: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob(f"*/{filename}")):
        frame = pd.read_csv(path)
        frame["ablation"] = label
        frame["case_dir"] = path.parent.name
        if "split_seed" not in frame.columns:
            frame["split_seed"] = infer_split_seed(path.parent)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_roots(items: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    loc_frames: list[pd.DataFrame] = []
    for item in items:
        if "=" in item:
            label, path_text = item.split("=", 1)
            root = Path(path_text)
        else:
            root = Path(item)
            label = infer_label(root)
        metric_frames.append(load_case_files(root, "metrics.csv", label))
        loc_frames.append(load_case_files(root, "stress_localization.csv", label))
    metrics = pd.concat([f for f in metric_frames if not f.empty], ignore_index=True)
    loc = pd.concat([f for f in loc_frames if not f.empty], ignore_index=True)
    return metrics, loc


def format_pm(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    focus = metrics[metrics["method"].isin(PRIMARY_METHODS)].copy()
    id_cols = ["ablation", "method", "dataset_id", "split", "split_seed"]
    metric_cols = [col for col in PRIMARY_METRICS if col in focus.columns]
    grouped = focus.groupby(id_cols, dropna=False)[metric_cols].mean().reset_index()
    overall = (
        grouped.groupby(["ablation", "method"], dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    overall.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in overall.columns
    ]

    rows: list[dict[str, object]] = []
    for _, row in overall.iterrows():
        item: dict[str, object] = {
            "ablation": row["ablation"],
            "method": row["method"],
            "case_n": int(
                grouped[(grouped["ablation"] == row["ablation"]) & (grouped["method"] == row["method"])].shape[0]
            ),
            "dataset_n": int(
                grouped[(grouped["ablation"] == row["ablation"]) & (grouped["method"] == row["method"])][
                    "dataset_id"
                ].nunique()
            ),
            "seed_n": int(
                grouped[(grouped["ablation"] == row["ablation"]) & (grouped["method"] == row["method"])][
                    "split_seed"
                ].nunique()
            ),
        }
        for metric in metric_cols:
            item[f"{metric}_mean"] = row[f"{metric}_mean"]
            item[f"{metric}_std"] = row[f"{metric}_std"]
            item[f"{metric}_n"] = int(row[f"{metric}_count"])
            item[f"{metric}_formatted"] = format_pm(row[f"{metric}_mean"], row[f"{metric}_std"])
        rows.append(item)
    summary = pd.DataFrame(rows)

    if "full" in set(summary["ablation"]):
        base = summary[summary["ablation"] == "full"].set_index("method")
        for metric in metric_cols:
            delta_col = f"delta_vs_full_{metric}"
            summary[delta_col] = pd.NA
            for idx, row in summary.iterrows():
                method = row["method"]
                if method not in base.index:
                    continue
                base_mean = float(base.loc[method, f"{metric}_mean"])
                value = float(row[f"{metric}_mean"])
                summary.loc[idx, delta_col] = value - base_mean
            if metric in LOWER_IS_BETTER:
                summary[f"improvement_vs_full_{metric}"] = -summary[delta_col].astype(float)
            else:
                summary[f"improvement_vs_full_{metric}"] = summary[delta_col].astype(float)
    return summary


def summarize_localization(loc: pd.DataFrame) -> pd.DataFrame:
    if loc.empty or "slice" not in loc.columns:
        return pd.DataFrame()
    macro = loc[loc["slice"] == "macro"].copy()
    cols = [col for col in ["macro_f1", "macro_f1_full_axis_set"] if col in macro.columns]
    if not cols:
        return pd.DataFrame()
    case = macro.groupby(["ablation", "dataset_id", "split", "split_seed"], dropna=False)[cols].mean().reset_index()
    summary = (
        case.groupby("ablation", dropna=False)[cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def write_markdown(summary: pd.DataFrame, out_path: Path) -> None:
    cols = ["ablation", "method"]
    for metric in ("failure_auroc", "failure_auprc", "risk_nll", "risk_coverage_auc", "selective_risk_at_80"):
        col = f"{metric}_formatted"
        if col in summary.columns:
            cols.append(col)
    table = summary[cols].sort_values(["method", "ablation"])
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for col in cols:
            value = row[col]
            values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, help="label=path or path entries")
    parser.add_argument("--out", default="artifacts/tables_axis_ablation_full_v1")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics, loc = read_roots(args.roots)
    if metrics.empty:
        raise SystemExit("no metrics.csv files found")
    metrics.to_csv(out / "axis_ablation_metrics_long.csv", index=False)
    if not loc.empty:
        loc.to_csv(out / "axis_ablation_localization_long.csv", index=False)

    summary = summarize_metrics(metrics)
    summary.to_csv(out / "axis_ablation_summary.csv", index=False)
    loc_summary = summarize_localization(loc)
    if not loc_summary.empty:
        loc_summary.to_csv(out / "axis_ablation_localization_summary.csv", index=False)
        summary = summary.merge(loc_summary, on="ablation", how="left")
        summary.to_csv(out / "axis_ablation_summary_with_localization.csv", index=False)

    write_markdown(summary, out / "axis_ablation_summary.md")
    print(summary.sort_values(["method", "ablation"]).to_string(index=False))


if __name__ == "__main__":
    main()
