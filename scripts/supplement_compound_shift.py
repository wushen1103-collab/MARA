#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SLICE_COLS = [
    "slice_chemistry_stress",
    "slice_assay_source_stress",
    "slice_label_reliability_stress",
    "slice_frontier_stress",
    "slice_model_conflict_stress",
]
AXIS_LABELS = {
    "slice_chemistry_stress": "A1_chemistry",
    "slice_assay_source_stress": "A2_assay_source",
    "slice_label_reliability_stress": "A3_label_reliability",
    "slice_frontier_stress": "A4_frontier",
    "slice_model_conflict_stress": "A5_model_conflict",
}
CONTRIB_COLS = [
    "contrib_A1_chemistry",
    "contrib_A2_assay_source",
    "contrib_A3_label_reliability",
    "contrib_A4_frontier",
    "contrib_A5_model_conflict",
]


def expand_roots(specs: list[str]) -> list[Path]:
    roots: list[Path] = []
    for spec in specs:
        matches = glob.glob(spec)
        if matches:
            roots.extend(Path(m) for m in matches)
        else:
            roots.append(Path(spec))
    return sorted({p.resolve() for p in roots})


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


def infer_suite(dataset_id: str) -> str:
    key = str(dataset_id).lower()
    if key.startswith("moleculeace_"):
        return "MoleculeACE"
    if "chembl" in key:
        return "ChEMBL10k"
    if "bindingdb" in key:
        return "BindingDB10k"
    if key.startswith("qm9"):
        return "QM9"
    return "Other"


def load_case(run_dir: Path, root: Path) -> pd.DataFrame | None:
    axis_path = run_dir / "axis_manifest.csv"
    pred_path = run_dir / "predictions.csv"
    if not axis_path.exists() or not pred_path.exists():
        return None
    axis_cols = [
        "dataset_id",
        "row_id",
        "split_role",
        *[c for c in SLICE_COLS],
        *[f"slice_score_{c.replace('slice_', '')}" for c in SLICE_COLS],
    ]
    axis_head = pd.read_csv(axis_path, nrows=1)
    axis_cols = [c for c in axis_cols if c in axis_head.columns]
    axis = pd.read_csv(axis_path, usecols=axis_cols)
    pred_head = pd.read_csv(pred_path, nrows=1)
    pred_cols = ["row_id", "failure", "risk_mara", "risk_mara_rank_fusion", *CONTRIB_COLS]
    pred_cols = [c for c in pred_cols if c in pred_head.columns]
    pred = pd.read_csv(pred_path, usecols=pred_cols)
    frame = axis.merge(pred, on="row_id", how="left")
    frame = frame[frame["split_role"] == "test"].copy()
    if frame.empty:
        return None
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        metric = pd.read_csv(metrics_path, nrows=1)
        split = str(metric["split"].iloc[0]) if "split" in metric.columns else "unknown"
        task_type = str(metric["task_type"].iloc[0]) if "task_type" in metric.columns else "unknown"
    else:
        parts = run_dir.name.rsplit("_", 2)
        split = parts[-2] if len(parts) >= 2 else "unknown"
        task_type = "unknown"
    frame["split"] = split
    frame["task_type"] = task_type
    frame["split_seed"] = infer_split_seed(run_dir, root)
    frame["run_dir"] = str(run_dir)
    frame["suite"] = frame["dataset_id"].map(infer_suite)
    return frame


def load_all(roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in roots:
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            case = load_case(run_dir, root)
            if case is not None:
                frames.append(case)
    if not frames:
        raise SystemExit("no axis_manifest/predictions case pairs found")
    return pd.concat(frames, ignore_index=True)


def combo_label(row: pd.Series) -> str:
    active = [AXIS_LABELS[col] for col in SLICE_COLS if col in row and int(row[col]) == 1]
    return "+".join(active) if active else "none"


def summarize_by_case(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, block in frame.groupby(["suite", "dataset_id", "split", "split_seed"], dropna=False):
        n = len(block)
        stress_count = block[[c for c in SLICE_COLS if c in block.columns]].sum(axis=1)
        row: dict[str, object] = {
            "suite": keys[0],
            "dataset_id": keys[1],
            "split": keys[2],
            "split_seed": keys[3],
            "n_test": int(n),
            "failure_rate": float(pd.to_numeric(block.get("failure"), errors="coerce").mean()),
            "single_axis_rate": float((stress_count == 1).mean()),
            "compound_axis_rate": float((stress_count >= 2).mean()),
            "triple_plus_axis_rate": float((stress_count >= 3).mean()),
            "no_axis_stress_rate": float((stress_count == 0).mean()),
            "mean_active_axis_count": float(stress_count.mean()),
        }
        for col in SLICE_COLS:
            if col in block:
                row[f"{AXIS_LABELS[col]}_rate"] = float(pd.to_numeric(block[col], errors="coerce").mean())
        for col in CONTRIB_COLS:
            if col in block:
                row[f"{col}_mean"] = float(pd.to_numeric(block[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_combos(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for col in SLICE_COLS:
        if col not in work:
            work[col] = 0
    work["stress_combo"] = work.apply(combo_label, axis=1)
    rows: list[dict[str, object]] = []
    group_cols = ["suite", "split", "stress_combo"]
    for keys, block in work.groupby(group_cols, dropna=False):
        rows.append(
            {
                "suite": keys[0],
                "split": keys[1],
                "stress_combo": keys[2],
                "n": int(len(block)),
                "dataset_n": int(block["dataset_id"].nunique()),
                "seed_n": int(block["split_seed"].nunique()),
                "row_rate_within_group": float(len(block) / len(work[(work["suite"] == keys[0]) & (work["split"] == keys[1])])),
                "failure_rate": float(pd.to_numeric(block.get("failure"), errors="coerce").mean()),
                "risk_mara_mean": float(pd.to_numeric(block.get("risk_mara"), errors="coerce").mean()),
                "risk_mara_rank_fusion_mean": float(pd.to_numeric(block.get("risk_mara_rank_fusion"), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["suite", "split", "n"], ascending=[True, True, False])


def summarize_temporal(frame: pd.DataFrame) -> pd.DataFrame:
    temporal = frame[frame["split"] == "temporal"].copy()
    if temporal.empty:
        return pd.DataFrame()
    by_case = summarize_by_case(temporal)
    return (
        by_case.groupby(["suite", "split"], dropna=False)
        .agg(
            dataset_n=("dataset_id", "nunique"),
            seed_n=("split_seed", "nunique"),
            n_test_mean=("n_test", "mean"),
            failure_rate_mean=("failure_rate", "mean"),
            single_axis_rate_mean=("single_axis_rate", "mean"),
            compound_axis_rate_mean=("compound_axis_rate", "mean"),
            triple_plus_axis_rate_mean=("triple_plus_axis_rate", "mean"),
            mean_active_axis_count=("mean_active_axis_count", "mean"),
            A1_chemistry_rate_mean=("A1_chemistry_rate", "mean"),
            A2_assay_source_rate_mean=("A2_assay_source_rate", "mean"),
            A3_label_reliability_rate_mean=("A3_label_reliability_rate", "mean"),
            A4_frontier_rate_mean=("A4_frontier_rate", "mean"),
            A5_model_conflict_rate_mean=("A5_model_conflict_rate", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", default="artifacts/tables_compound_shift_v1")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_all(expand_roots(args.roots))
    frame.to_csv(out / "compound_shift_long.csv", index=False)
    by_case = summarize_by_case(frame)
    combos = summarize_combos(frame)
    temporal = summarize_temporal(frame)
    by_case.to_csv(out / "compound_shift_by_case.csv", index=False)
    combos.to_csv(out / "compound_shift_combos.csv", index=False)
    temporal.to_csv(out / "compound_shift_temporal_summary.csv", index=False)
    try:
        temporal.to_markdown(out / "compound_shift_temporal_summary.md", index=False)
        combos.head(60).to_markdown(out / "compound_shift_top_combos.md", index=False)
    except Exception:
        pass
    print("Temporal compound shift")
    print(temporal.to_string(index=False))
    print("\nTop combos")
    print(combos.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
