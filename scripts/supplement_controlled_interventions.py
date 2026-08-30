#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


AXES = ["A1_chemistry", "A2_assay_source", "A3_label_reliability", "A4_frontier", "A5_model_conflict"]


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
    return "Other"


def infer_split_seed(run_dir: Path, root: Path) -> int | float:
    match = re.search(r"_seed(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    match = re.search(r"seed(\d+)", root.name)
    return int(match.group(1)) if match else np.nan


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def fit_quantile_minmax(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.nanquantile(train, 0.01, axis=0)
    hi = np.nanquantile(train, 0.99, axis=0)
    same = np.isclose(lo, hi)
    hi[same] = lo[same] + 1.0
    lo = np.where(np.isfinite(lo), lo, 0.0)
    hi = np.where(np.isfinite(hi), hi, lo + 1.0)
    return lo, hi


def scale(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    arr = np.where(np.isfinite(x), x, lo)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def compute_contrib(
    x_scaled: np.ndarray,
    feature_cols: list[str],
    available_axes: dict[str, list[str]],
    weights: np.ndarray,
    intercept: float,
    anchor_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    groups = {
        axis: [feature_cols.index(c) for c in cols if c in feature_cols]
        for axis, cols in available_axes.items()
    }
    weighted = np.maximum(x_scaled, 0.0) * np.maximum(weights, 0.0)
    raw = {f"contrib_{axis}": weighted[:, idx].sum(axis=1) if idx else np.zeros(len(x_scaled)) for axis, idx in groups.items()}
    raw_df = pd.DataFrame(raw)
    risk = sigmoid(intercept + x_scaled @ weights)
    anchor = {}
    for axis, idx in groups.items():
        anchor[f"contrib_{axis}"] = np.max(x_scaled[:, idx], axis=1) if idx else np.zeros(len(x_scaled))
    anchor_df = pd.DataFrame(anchor)
    if anchor_alpha <= 0:
        return raw_df, raw_df.copy(), risk
    blended = (1.0 - anchor_alpha) * raw_df.to_numpy(dtype=float) + anchor_alpha * anchor_df.to_numpy(dtype=float)
    denom = np.where(blended.sum(axis=1, keepdims=True) > 1e-12, blended.sum(axis=1, keepdims=True), 1.0)
    props = blended / denom
    risk_mass = np.maximum(np.log(np.clip(risk, 1e-6, 1 - 1e-6) / np.clip(1 - risk, 1e-6, 1.0)) - intercept, 0.0)
    anchored = pd.DataFrame(props * risk_mass[:, None], columns=raw_df.columns)
    return raw_df, anchored, risk


def case_intervention(run_dir: Path, root: Path, levels: list[float]) -> list[dict]:
    model_path = run_dir / "mara_model.json"
    axis_path = run_dir / "axis_manifest.csv"
    if not model_path.exists() or not axis_path.exists():
        return []
    model = json.loads(model_path.read_text(encoding="utf-8"))
    feature_cols = model.get("base_feature_columns") or model.get("feature_columns") or []
    available_axes = model.get("available_axes") or {}
    if model.get("use_interactions"):
        return []
    weights_map = model.get("weights") or {}
    weights = np.asarray([float(weights_map.get(c, 0.0)) for c in feature_cols], dtype=float)
    intercept = float(model.get("intercept", 0.0))
    anchor_alpha = float(model.get("axis_attribution_anchor_alpha", 0.0))
    axis_frame = pd.read_csv(axis_path)
    if "split_role" not in axis_frame or not feature_cols:
        return []
    train = axis_frame[axis_frame["split_role"].astype(str).eq("proper_train")]
    if train.empty:
        return []
    raw_train = train[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lo, hi = fit_quantile_minmax(raw_train)
    scaled_train = scale(raw_train, lo, hi)
    baseline = np.nanmedian(scaled_train, axis=0)
    baseline = np.where(np.isfinite(baseline), baseline, 0.0)
    dataset_id = str(axis_frame["dataset_id"].iloc[0]) if "dataset_id" in axis_frame else run_dir.name
    split_match = re.search(r"_(random|scaffold|assay|source|target|temporal|official|drugood)_seed", run_dir.name)
    split = split_match.group(1) if split_match else ""
    rows = []
    for injected_axis in AXES:
        cols = [c for c in available_axes.get(injected_axis, []) if c in feature_cols]
        if not cols:
            continue
        idx = [feature_cols.index(c) for c in cols]
        grid = np.tile(baseline.reshape(1, -1), (len(levels), 1))
        for i, level in enumerate(levels):
            grid[i, idx] = float(level)
        raw_c, anchored_c, risk = compute_contrib(grid, feature_cols, available_axes, weights, intercept, anchor_alpha)
        for mode, contrib in [("raw_learned", raw_c), ("anchored_reported", anchored_c)]:
            contrib_cols = [c for c in contrib.columns if c.startswith("contrib_")]
            top_axis = [str(c).replace("contrib_", "") for c in contrib[contrib_cols].idxmax(axis=1)]
            for response_axis in AXES:
                col = f"contrib_{response_axis}"
                vals = contrib[col].to_numpy(dtype=float) if col in contrib else np.zeros(len(levels))
                slope = float(vals[-1] - vals[0])
                rows.append(
                    {
                        "suite": infer_suite(dataset_id),
                        "dataset_id": dataset_id,
                        "split": split,
                        "split_seed": infer_split_seed(run_dir, root),
                        "run_dir": str(run_dir),
                        "mode": mode,
                        "injected_axis": injected_axis,
                        "response_axis": response_axis,
                        "response_delta_level1_minus0": slope,
                        "response_at_level0": float(vals[0]),
                        "response_at_level1": float(vals[-1]),
                        "risk_delta_level1_minus0": float(risk[-1] - risk[0]),
                        "top_axis_at_level1": top_axis[-1] if top_axis else "",
                        "top_axis_correct_at_level1": bool(top_axis and top_axis[-1] == injected_axis),
                        "available_injected_axis": True,
                    }
                )
    return rows


def summarize(long: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, block in long.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n"] = int(len(block))
        row["dataset_n"] = int(block["dataset_id"].nunique())
        row["seed_n"] = int(block["split_seed"].nunique())
        vals = pd.to_numeric(block["response_delta_level1_minus0"], errors="coerce").dropna()
        row["response_delta_mean"] = float(vals.mean()) if len(vals) else np.nan
        row["response_delta_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
        row["top_axis_correct_rate"] = float(block["top_axis_correct_at_level1"].mean())
        risk_vals = pd.to_numeric(block["risk_delta_level1_minus0"], errors="coerce").dropna()
        row["risk_delta_mean"] = float(risk_vals.mean()) if len(risk_vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    roots = expand_roots(args.roots)
    out = ensure_dir(args.out)
    rows: list[dict] = []
    for root in roots:
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            rows.extend(case_intervention(run_dir, root, args.levels))
    if not rows:
        raise SystemExit("no intervention rows found")
    long = pd.DataFrame(rows)
    long.to_csv(out / "controlled_intervention_long.csv", index=False)
    matrix = summarize(long, ["mode", "injected_axis", "response_axis"])
    matrix.to_csv(out / "controlled_intervention_matrix.csv", index=False)
    by_suite = summarize(long, ["suite", "mode", "injected_axis", "response_axis"])
    by_suite.to_csv(out / "controlled_intervention_matrix_by_suite.csv", index=False)

    selectivity_rows = []
    for keys, block in matrix.groupby(["mode", "injected_axis"], dropna=False):
        mode, injected = keys
        diag = block.loc[block["response_axis"].eq(injected), "response_delta_mean"]
        off = block.loc[~block["response_axis"].eq(injected), "response_delta_mean"]
        selectivity_rows.append(
            {
                "mode": mode,
                "injected_axis": injected,
                "diagonal_response_delta": float(diag.iloc[0]) if len(diag) else np.nan,
                "max_offdiagonal_response_delta": float(off.max()) if len(off) else np.nan,
                "diagonal_minus_max_offdiagonal": float(diag.iloc[0] - off.max()) if len(diag) and len(off) else np.nan,
            }
        )
    selectivity = pd.DataFrame(selectivity_rows)
    selectivity.to_csv(out / "controlled_intervention_selectivity.csv", index=False)
    try:
        matrix.to_markdown(out / "table_controlled_intervention_matrix.md", index=False)
        selectivity.to_markdown(out / "table_controlled_intervention_selectivity.md", index=False)
    except Exception:
        pass
    print(f"Wrote controlled intervention matrix with {len(long)} rows to {out}")


if __name__ == "__main__":
    main()
