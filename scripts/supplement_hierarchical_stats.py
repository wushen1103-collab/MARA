#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HIGHER_IS_BETTER = {"failure_auroc", "failure_auprc"}
LOWER_IS_BETTER = {
    "risk_ece_10bin",
    "risk_nll",
    "risk_brier",
    "risk_coverage_auc",
    "selective_risk_at_80",
    "selective_risk_at_90",
}
METRICS = [
    "failure_auroc",
    "failure_auprc",
    "risk_ece_10bin",
    "risk_nll",
    "risk_brier",
    "risk_coverage_auc",
    "selective_risk_at_80",
    "selective_risk_at_90",
]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def compare(primary_value: float, baseline_value: float, metric: str) -> float:
    if metric in HIGHER_IS_BETTER:
        return primary_value - baseline_value
    return baseline_value - primary_value


def ci(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-long", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--primary", default="risk_mara_rank_fusion")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--tie-eps", type=float, default=1e-6)
    args = parser.parse_args()

    out = ensure_dir(args.out)
    metrics = pd.read_csv(args.metrics_long)
    metrics = metrics.copy()
    metric_cols = [m for m in METRICS if m in metrics.columns]
    key_cols = ["suite", "dataset_id", "split", "split_seed", "task_type"]
    primary = metrics[metrics["method"].eq(args.primary)][key_cols + metric_cols]
    rows = []
    unit_rows = []
    rng = np.random.default_rng(args.seed)
    dataset_units = np.array(sorted(metrics["dataset_id"].dropna().unique()))

    for baseline in sorted(m for m in metrics["method"].unique() if m != args.primary):
        other = metrics[metrics["method"].eq(baseline)][key_cols + metric_cols]
        merged = primary.merge(other, on=key_cols, suffixes=("_primary", "_baseline"))
        if merged.empty:
            continue
        for metric in metric_cols:
            deltas = []
            case_rows = []
            for _, row in merged.iterrows():
                delta = compare(float(row[f"{metric}_primary"]), float(row[f"{metric}_baseline"]), metric)
                if np.isfinite(delta):
                    deltas.append(delta)
                    case_rows.append(
                        {
                            "suite": row["suite"],
                            "dataset_id": row["dataset_id"],
                            "split": row["split"],
                            "split_seed": row["split_seed"],
                            "baseline": baseline,
                            "metric": metric,
                            "advantage": delta,
                        }
                    )
            case_frame = pd.DataFrame(case_rows)
            if case_frame.empty:
                continue
            unit_adv = (
                case_frame.groupby(["suite", "dataset_id"], dropna=False)["advantage"]
                .mean()
                .reset_index()
            )
            for _, urow in unit_adv.iterrows():
                unit_rows.append(urow.to_dict())
            wins = int((unit_adv["advantage"] > args.tie_eps).sum())
            ties = int((unit_adv["advantage"].abs() <= args.tie_eps).sum())
            losses = int((unit_adv["advantage"] < -args.tie_eps).sum())
            unit_map = unit_adv.set_index("dataset_id")["advantage"].to_dict()
            boot = []
            available_units = np.array(sorted(unit_map))
            for _ in range(args.bootstrap):
                sample = rng.choice(available_units, size=len(available_units), replace=True)
                boot.append(float(np.mean([unit_map[u] for u in sample])))
            lo, hi = ci(np.asarray(boot, dtype=float))
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "case_n": int(len(case_frame)),
                    "dataset_unit_n": int(len(unit_adv)),
                    "mean_case_advantage": float(np.mean(case_frame["advantage"])),
                    "mean_dataset_unit_advantage": float(np.mean(unit_adv["advantage"])),
                    "blocked_bootstrap_ci95_low": lo,
                    "blocked_bootstrap_ci95_high": hi,
                    "win_units": wins,
                    "tie_units": ties,
                    "loss_units": losses,
                    "win_tie_loss": f"{wins}/{ties}/{losses}",
                    "blocked_bootstrap_reps": int(args.bootstrap),
                    "blocked_by": "dataset_id",
                    "interpretation_positive_advantage": f"{args.primary} better than baseline",
                }
            )

    summary = pd.DataFrame(rows)
    unit_long = pd.DataFrame(unit_rows)
    summary.to_csv(out / "dataset_unit_blocked_bootstrap.csv", index=False)
    unit_long.to_csv(out / "dataset_unit_advantage_long.csv", index=False)
    focus = summary[
        summary["baseline"].isin(
            ["risk_mara", "risk_scalar_full", "risk_rf_error_predictor", "risk_calibrated_tanimoto", "risk_validation_knn_error"]
        )
    ].copy()
    focus.to_csv(out / "dataset_unit_blocked_bootstrap_focus.csv", index=False)
    try:
        focus.to_markdown(out / "table_dataset_unit_blocked_bootstrap_focus.md", index=False)
    except Exception:
        focus.to_csv(out / "table_dataset_unit_blocked_bootstrap_focus.md", index=False)
    print(f"Wrote blocked bootstrap stats for {summary['baseline'].nunique()} baselines and {len(metric_cols)} metrics to {out}")


if __name__ == "__main__":
    main()
