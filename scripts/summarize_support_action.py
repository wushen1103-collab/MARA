#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


METRICS = ["delta_mae", "delta_failure_rate", "delta_error_rate", "delta_log_loss"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated real-data support-acquisition actions.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for path in sorted(Path(args.root).glob("seed*/support_intervention.csv")):
        match = re.search(r"seed(\d+)", path.parent.name)
        frame = pd.read_csv(path)
        frame.insert(2, "split_seed", int(match.group(1)))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No seed*/support_intervention.csv under {args.root}")

    raw = pd.concat(frames, ignore_index=True)
    keys = ["dataset_id", "split", "split_seed", "task_type", "policy"]
    available_metrics = [metric for metric in METRICS if metric in raw.columns and raw[metric].notna().any()]
    unit = raw.groupby(keys, as_index=False)[available_metrics].mean()

    summary_rows = []
    for (task_type, policy), block in unit.groupby(["task_type", "policy"], sort=True):
        row = {
            "task_type": task_type,
            "policy": policy,
            "independent_units": int(len(block)),
            "datasets": int(block["dataset_id"].nunique()),
            "splits": int(block["split"].nunique()),
            "seeds": int(block["split_seed"].nunique()),
        }
        for metric in available_metrics:
            values = block[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    paired_rows = []
    pair_keys = ["dataset_id", "split", "split_seed", "task_type"]
    for competitor in sorted(set(unit["policy"]) - {"mara_axis"}):
        for metric in available_metrics:
            pivot = unit[unit["policy"].isin(["mara_axis", competitor])].pivot_table(
                index=pair_keys, columns="policy", values=metric, aggfunc="mean"
            ).dropna()
            if pivot.empty:
                continue
            delta = pivot["mara_axis"] - pivot[competitor]
            try:
                p_value = float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue)
            except ValueError:
                p_value = 1.0
            paired_rows.append(
                {
                    "metric": metric,
                    "competitor": competitor,
                    "paired_units": int(len(delta)),
                    "mara_minus_competitor_mean": float(delta.mean()),
                    "mara_minus_competitor_std": float(delta.std(ddof=1)),
                    "wilcoxon_p_uncorrected": p_value,
                    "interpretation": "positive favors MARA because all delta metrics are improvements",
                }
            )

    paired = pd.DataFrame(paired_rows)
    paired["wilcoxon_p_holm"] = float("nan")
    for _, indices in paired.groupby("metric").groups.items():
        ordered = paired.loc[indices].sort_values("wilcoxon_p_uncorrected").index.tolist()
        running = 0.0
        m = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, float(paired.at[index, "wilcoxon_p_uncorrected"]) * (m - rank))
            running = max(running, adjusted)
            paired.at[index, "wilcoxon_p_holm"] = running

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "support_action_raw.csv", index=False)
    unit.to_csv(out / "support_action_independent_units.csv", index=False)
    summary.to_csv(out / "support_action_mean_std.csv", index=False)
    paired.to_csv(out / "support_action_paired_comparison.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
