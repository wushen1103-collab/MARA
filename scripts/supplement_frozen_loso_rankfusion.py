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
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


COMPONENTS = [
    "risk_mara",
    "risk_scalar_full",
    "risk_validation_knn_error",
    "risk_calibrated_tanimoto",
    "risk_rf_error_predictor",
]
COMPARATORS = [
    "risk_mara",
    "risk_scalar_full",
    "risk_knn_tanimoto",
    "risk_calibrated_tanimoto",
    "risk_validation_knn_error",
    "risk_rf_error_predictor",
    "risk_mara_rank_fusion",
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
        return "MoleculeACE30"
    if "chembl" in key:
        return "ChEMBL10k"
    if "bindingdb" in key:
        return "BindingDB10k"
    if key.startswith("drugood"):
        return "DrugOOD"
    return "Other"


def infer_split_seed(run_dir: Path, root: Path) -> int | float:
    match = re.search(r"_seed(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    match = re.search(r"seed(\d+)", root.name)
    return int(match.group(1)) if match else np.nan


def rank_columns(frame: pd.DataFrame, components: list[str]) -> np.ndarray:
    ranks = []
    for col in components:
        score = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(score)
        fill = float(np.nanmedian(score[finite])) if finite.any() else 0.0
        score = np.where(finite, score, fill)
        ranks.append((pd.Series(score).rank(method="average").to_numpy(dtype=float) - 1.0) / max(1, len(score) - 1))
    return np.vstack(ranks).T


def fit_nonnegative_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> tuple[np.ndarray, float]:
    y = y.astype(float)
    n = x.shape[1]
    if len(np.unique(y)) < 2:
        return np.zeros(n, dtype=float), math.log(np.clip(y.mean(), 1e-5, 1 - 1e-5) / np.clip(1 - y.mean(), 1e-5, 1))

    def objective(theta: np.ndarray):
        w = theta[:n]
        b = theta[-1]
        z = b + x @ w
        per = np.logaddexp(0.0, z) - y * z
        loss = float(np.mean(per) + l2 * np.sum(w * w))
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        grad = np.concatenate([x.T @ (p - y) / len(y) + 2 * l2 * w, [float(np.mean(p - y))]])
        return loss, grad

    p0 = float(np.clip(np.mean(y), 1e-4, 1 - 1e-4))
    init = np.zeros(n + 1, dtype=float)
    init[-1] = math.log(p0 / (1 - p0))
    result = minimize(lambda th: objective(th), init, jac=True, method="L-BFGS-B", bounds=[(0, None)] * n + [(None, None)])
    return result.x[:n].astype(float), float(result.x[-1])


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def load_prediction_rows(roots: list[Path], components: list[str]) -> pd.DataFrame:
    frames = []
    need = set(components + COMPARATORS + ["dataset_id", "split_role", "failure"])
    for root in roots:
        for path in sorted(root.glob("*/predictions.csv")):
            frame = pd.read_csv(path)
            if not need.intersection(frame.columns):
                continue
            missing = [c for c in components if c not in frame.columns]
            if missing or "failure" not in frame or "split_role" not in frame:
                continue
            dataset_id = str(frame["dataset_id"].iloc[0]) if "dataset_id" in frame else path.parent.name
            split_match = re.search(r"_(random|scaffold|assay|source|target|temporal|official|drugood)_seed", path.parent.name)
            split = split_match.group(1) if split_match else ""
            ranks = rank_columns(frame, components)
            for i, col in enumerate(components):
                frame[f"rank_{col}"] = ranks[:, i]
            frame["suite"] = infer_suite(dataset_id)
            frame["dataset_id"] = dataset_id
            frame["split"] = split
            frame["split_seed"] = infer_split_seed(path.parent, root)
            frame["run_dir"] = str(path.parent)
            keep = [
                "suite",
                "dataset_id",
                "split",
                "split_seed",
                "run_dir",
                "split_role",
                "failure",
                *components,
                *[f"rank_{c}" for c in components],
                *[c for c in COMPARATORS if c in frame.columns and c not in components],
            ]
            keep_unique = list(dict.fromkeys(keep))
            frames.append(frame.loc[:, keep_unique].copy())
    if not frames:
        raise SystemExit("no prediction frames found")
    return pd.concat(frames, ignore_index=True)


def metric_row(y: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    risk = np.clip(np.asarray(risk, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    order = np.argsort(risk)
    coverages = np.linspace(0.05, 1.0, 96)
    curve = [float(np.mean(y[order[: max(1, int(round(c * len(y))))]])) for c in coverages]
    return {
        "failure_rate": float(np.mean(y)),
        "failure_auroc": float(roc_auc_score(y, risk)) if len(np.unique(y)) > 1 else np.nan,
        "failure_auprc": float(average_precision_score(y, risk)) if y.sum() > 0 else np.nan,
        "risk_nll": float(log_loss(y, risk, labels=[0, 1])) if len(y) else np.nan,
        "risk_brier": float(brier_score_loss(y, risk)) if len(y) else np.nan,
        "risk_coverage_auc": float(np.trapezoid(curve, coverages) / (coverages[-1] - coverages[0])),
        "selective_risk_at_80": float(np.mean(y[order[: max(1, int(round(0.8 * len(y))))]])),
        "selective_risk_at_90": float(np.mean(y[order[: max(1, int(round(0.9 * len(y))))]])),
    }


def summarize_metrics(long: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = ["failure_auroc", "failure_auprc", "risk_nll", "risk_brier", "risk_coverage_auc", "selective_risk_at_80", "selective_risk_at_90"]
    rows = []
    for keys, block in long.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n"] = int(len(block))
        row["seed_n"] = int(block["split_seed"].nunique()) if "split_seed" in block else np.nan
        row["dataset_n"] = int(block["dataset_id"].nunique()) if "dataset_id" in block else np.nan
        for col in metric_cols:
            vals = pd.to_numeric(block[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--components", nargs="*", default=COMPONENTS)
    args = parser.parse_args()

    roots = expand_roots(args.roots)
    out = ensure_dir(args.out)
    all_rows = load_prediction_rows(roots, args.components)
    suites = sorted(all_rows["suite"].dropna().unique())
    case_rows = []
    model_cards = []
    for heldout in suites:
        dev = all_rows[(all_rows["suite"] != heldout) & (all_rows["split_role"].astype(str) == "test")].copy()
        test = all_rows[(all_rows["suite"] == heldout) & (all_rows["split_role"].astype(str) == "test")].copy()
        rank_cols = [f"rank_{c}" for c in args.components]
        x_dev = dev[rank_cols].to_numpy(dtype=float)
        y_dev = dev["failure"].to_numpy(dtype=int)
        weights, intercept = fit_nonnegative_logistic(x_dev, y_dev)
        model_cards.append(
            {
                "heldout_suite": heldout,
                "development_suites": sorted(set(suites) - {heldout}),
                "components": args.components,
                "fusion_rule": "nonnegative logistic over within-case rank-normalized component scores",
                "calibration": "intercept and component weights fit only on development-suite test rows; heldout suite labels unused",
                "weights": {c: float(w) for c, w in zip(args.components, weights)},
                "intercept": float(intercept),
                "n_development_rows": int(len(dev)),
                "n_heldout_rows": int(len(test)),
            }
        )
        x_test = test[rank_cols].to_numpy(dtype=float)
        test = test.copy()
        test["frozen_loso_mara_rf"] = sigmoid(intercept + x_test @ weights)
        test["frozen_loso_equal_rank_rf"] = sigmoid(intercept + np.mean(x_test, axis=1) * max(float(np.sum(weights)), 1e-6))
        methods = ["frozen_loso_mara_rf", "frozen_loso_equal_rank_rf", *[m for m in COMPARATORS if m in test.columns]]
        for keys, block in test.groupby(["suite", "dataset_id", "split", "split_seed", "run_dir"], dropna=False):
            y = block["failure"].to_numpy(dtype=int)
            for method in methods:
                if method not in block:
                    continue
                row = {c: v for c, v in zip(["suite", "dataset_id", "split", "split_seed", "run_dir"], keys)}
                row["method"] = method
                row.update(metric_row(y, block[method].to_numpy(dtype=float)))
                case_rows.append(row)
    Path(out / "frozen_loso_model_cards.json").write_text(json.dumps(model_cards, indent=2), encoding="utf-8")
    long = pd.DataFrame(case_rows)
    long.to_csv(out / "frozen_loso_metrics_long.csv", index=False)
    by_suite = summarize_metrics(long, ["suite", "method"])
    overall = summarize_metrics(long, ["method"])
    by_suite.to_csv(out / "frozen_loso_by_suite.csv", index=False)
    overall.to_csv(out / "frozen_loso_overall.csv", index=False)
    try:
        by_suite.sort_values(["suite", "failure_auroc_mean"], ascending=[True, False]).to_markdown(out / "table_frozen_loso_by_suite.md", index=False)
        overall.sort_values("failure_auroc_mean", ascending=False).to_markdown(out / "table_frozen_loso_overall.md", index=False)
    except Exception:
        pass
    print(f"Wrote frozen LOSO rank-fusion results for {len(suites)} heldout suites to {out}")


if __name__ == "__main__":
    main()
