#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mara_public_suite as mara  # noqa: E402


def tanimoto_matrix(query_bits: np.ndarray, ref_bits: np.ndarray) -> np.ndarray:
    q = query_bits.astype(np.float32, copy=False)
    r = ref_bits.astype(np.float32, copy=False)
    inter = q @ r.T
    union = q.sum(axis=1)[:, None] + r.sum(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def split_validation_pool(
    dataset: pd.DataFrame,
    val_idx: np.ndarray,
    split_name: str,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the original validation fold into label-visible calibration and hidden acquisition pools."""
    rng = np.random.default_rng(split_seed + 2)
    if split_name == "scaffold":
        groups: dict[str, list[int]] = {}
        for idx in val_idx:
            groups.setdefault(str(dataset.iloc[int(idx)]["scaffold"]), []).append(int(idx))
        grouped = list(groups.values())
        rng.shuffle(grouped)
        grouped.sort(key=len, reverse=True)
        buckets: dict[str, list[int]] = {"calibration": [], "acquisition": []}
        target = len(val_idx) / 2.0
        for members in grouped:
            chosen = min(buckets, key=lambda name: len(buckets[name]) - target)
            buckets[chosen].extend(members)
        calibration_idx = np.asarray(sorted(buckets["calibration"]), dtype=int)
        acquisition_idx = np.asarray(sorted(buckets["acquisition"]), dtype=int)
    else:
        shuffled = rng.permutation(val_idx)
        cut = len(shuffled) // 2
        calibration_idx = np.asarray(sorted(shuffled[:cut]), dtype=int)
        acquisition_idx = np.asarray(sorted(shuffled[cut:]), dtype=int)
    if min(len(calibration_idx), len(acquisition_idx)) < 32:
        raise RuntimeError(
            f"calibration/acquisition split too small: {len(calibration_idx)}, {len(acquisition_idx)}"
        )
    return calibration_idx, acquisition_idx


def prepare_bundle(
    bundle: mara.DatasetBundle,
    split_name: str,
    split_seed: int,
    n_bits: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = bundle.frame.copy().reset_index(drop=True)
    dataset["mol"] = dataset["smiles"].map(mara.mol_from_smiles)
    dataset = dataset[dataset["mol"].notna()].reset_index(drop=True)
    dataset["row_id"] = np.arange(len(dataset), dtype=int)
    dataset["scaffold"] = [mara.scaffold_smiles(mol, smi) for mol, smi in zip(dataset["mol"], dataset["smiles"])]
    if split_name == "random":
        split_role = mara.make_random_split(dataset, bundle.task_type, split_seed)
    elif split_name == "scaffold":
        split_role = mara.make_scaffold_split(dataset, split_seed)
    else:
        raise ValueError(split_name)
    dataset["split_role"] = split_role.to_numpy()
    train_idx = np.flatnonzero(dataset["split_role"].to_numpy() == "proper_train")
    val_idx = np.flatnonzero(dataset["split_role"].to_numpy() == "validation")
    test_idx = np.flatnonzero(dataset["split_role"].to_numpy() == "test")
    calibration_idx, acquisition_idx = split_validation_pool(dataset, val_idx, split_name, split_seed)
    dataset["action_role"] = dataset["split_role"]
    dataset.loc[calibration_idx, "action_role"] = "calibration"
    dataset.loc[acquisition_idx, "action_role"] = "acquisition"
    x_bits = mara.ecfp_matrix(dataset["mol"].tolist(), n_bits=n_bits)
    return dataset.drop(columns=["mol"]), x_bits, train_idx, calibration_idx, acquisition_idx, test_idx


def axis_selector(
    axis_name: str,
    test_pos: int,
    pool_idx: np.ndarray,
    sim_row: np.ndarray,
    axis_frame: pd.DataFrame,
    rng: np.random.Generator,
) -> int:
    pool_frame = axis_frame.iloc[pool_idx]
    if axis_name == "A1_chemistry":
        return int(pool_idx[int(np.argmax(sim_row))])
    if axis_name == "A4_frontier":
        target = float(axis_frame.iloc[test_pos]["a4_mahalanobis_frontier"])
        diff = np.abs(pool_frame["a4_mahalanobis_frontier"].to_numpy(dtype=float) - target)
        return int(pool_idx[int(np.argmin(diff))])
    if axis_name == "A5_model_conflict":
        pool_conflict = (
            pool_frame.get("a5_ensemble_std", pd.Series(0.0, index=pool_frame.index)).to_numpy(dtype=float)
            + pool_frame.get("a5_entropy", pd.Series(0.0, index=pool_frame.index)).to_numpy(dtype=float)
            + pool_frame.get("a5_margin_risk", pd.Series(0.0, index=pool_frame.index)).to_numpy(dtype=float)
        )
        score = 0.5 * sim_row + 0.5 * (pool_conflict / (np.nanmax(pool_conflict) + 1e-12))
        return int(pool_idx[int(np.argmax(score))])
    return int(rng.choice(pool_idx))


def select_support(
    policy: str,
    selected_test_idx: np.ndarray,
    pool_idx: np.ndarray,
    sim: np.ndarray,
    axis_frame: pd.DataFrame,
    budget: int,
    rng: np.random.Generator,
) -> list[int]:
    chosen: list[int] = []
    used: set[int] = set()
    pool_frame = axis_frame.iloc[pool_idx]
    available_axes = ["A1_chemistry", "A4_frontier", "A5_model_conflict"]

    def mara_top_axis(test_pos: int) -> str:
        contrib_cols = [c for c in axis_frame.columns if c.startswith("contrib_A")]
        if not contrib_cols:
            return "A1_chemistry"
        return axis_frame.iloc[test_pos][contrib_cols].astype(float).idxmax().replace("contrib_", "")

    def mismatched_axis(axis_name: str) -> str:
        if axis_name == "A1_chemistry":
            return "A5_model_conflict"
        if axis_name == "A5_model_conflict":
            return "A1_chemistry"
        return "A1_chemistry"

    if policy == "uncertainty_pool":
        score = (
            pool_frame.get("a5_ensemble_std", pd.Series(0.0, index=pool_frame.index)).to_numpy(dtype=float)
            + pool_frame.get("a5_entropy", pd.Series(0.0, index=pool_frame.index)).to_numpy(dtype=float)
            + pool_frame.get("a5_margin_risk", pd.Series(0.0, index=pool_frame.index)).to_numpy(dtype=float)
        )
        order = np.argsort(-score)
        return [int(pool_idx[i]) for i in order[:budget]]
    for local_i, test_pos in enumerate(selected_test_idx):
        if policy == "mara_axis":
            axis_name = mara_top_axis(int(test_pos))
        elif policy == "mismatched_axis":
            axis_name = mismatched_axis(mara_top_axis(int(test_pos)))
        elif policy == "random_axis":
            axis_name = str(rng.choice(available_axes))
        elif policy == "nearest_neighbor":
            axis_name = "A1_chemistry"
        else:
            raise ValueError(policy)
        picked = axis_selector(axis_name, int(test_pos), pool_idx, sim[local_i], axis_frame, rng)
        if picked not in used:
            chosen.append(picked)
            used.add(picked)
        if len(chosen) >= budget:
            break
    if len(chosen) < budget:
        extras = [int(i) for i in rng.permutation(pool_idx) if int(i) not in used]
        chosen.extend(extras[: budget - len(chosen)])
    return chosen[:budget]


def evaluate_predictions(task_type: str, y: np.ndarray, pred: mara.BasePredictions, threshold: float | None) -> dict:
    if task_type == "classification":
        p = np.clip(pred.proba_mean, 1e-6, 1.0 - 1e-6)
        err = ((p >= 0.5).astype(float) != y).astype(float)
        return {
            "error_rate": float(err.mean()),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
        }
    abs_err = np.abs(y - pred.pred_mean)
    out = {
        "mae": float(mean_absolute_error(y, pred.pred_mean)),
        "mean_abs_error": float(abs_err.mean()),
    }
    if threshold is not None:
        out["failure_rate"] = float((abs_err > threshold).mean())
    return out


def fit_selection_mara(
    dataset: pd.DataFrame,
    x_bits: np.ndarray,
    train_idx: np.ndarray,
    calibration_idx: np.ndarray,
    bundle: mara.DatasetBundle,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[mara.BasePredictions, pd.DataFrame, float | None]:
    """Fit the frozen predictor and MARA head without acquisition-pool labels."""
    y_all = dataset["y"].to_numpy(dtype=float)
    base = mara.train_base_ensemble(x_bits, y_all, train_idx, bundle.task_type, args.seeds, args.workers)
    axis = mara.build_axis_features(
        dataset,
        x_bits,
        train_idx,
        dataset["split_role"],
        bundle.task_type,
        args.split_seed,
    )
    axis = mara.add_model_conflict_features(axis, base, bundle.task_type)
    if bundle.task_type == "classification":
        p_cal = np.clip(base.proba_mean[calibration_idx], 1e-6, 1.0 - 1e-6)
        failure_cal = ((p_cal >= 0.5).astype(float) != y_all[calibration_idx]).astype(int)
        threshold = None
        definition = {
            "task_type": bundle.task_type,
            "failure_event": "predicted_label_mismatch",
            "threshold_source": "calibration labels only",
            "calibration_failure_rate": float(failure_cal.mean()),
        }
    else:
        error_cal = np.abs(y_all[calibration_idx] - base.pred_mean[calibration_idx])
        threshold = float(np.quantile(error_cal, 0.8))
        failure_cal = (error_cal > threshold).astype(int)
        definition = {
            "task_type": bundle.task_type,
            "failure_event": "absolute_error_above_calibration_80th_percentile",
            "threshold": threshold,
            "threshold_source": "calibration residuals only",
            "calibration_failure_rate": float(failure_cal.mean()),
        }
    axis["failure"] = np.nan
    axis.loc[calibration_idx, "failure"] = failure_cal
    (out_dir / "failure_definition.yaml").write_text(
        yaml.safe_dump(definition, sort_keys=True), encoding="utf-8"
    )
    feature_cols, available_axes = mara.feature_columns_for_available_axes(axis, train_idx)
    scaler = mara.QuantileMinMaxScaler().fit(axis.iloc[train_idx][feature_cols].to_numpy(dtype=float))
    x_scaled = scaler.transform(axis[feature_cols].to_numpy(dtype=float))
    axis_idx = {
        axis_name: [feature_cols.index(col) for col in cols if col in feature_cols]
        for axis_name, cols in available_axes.items()
    }
    x_mara, mara_feature_cols, mara_axis_idx, interaction_alloc, _ = mara.build_mara_design(
        x_base=x_scaled,
        feature_cols=feature_cols,
        axis_idx=axis_idx,
        use_interactions=False,
    )
    y_cal = axis.iloc[calibration_idx]["failure"].to_numpy(dtype=int)
    head = mara.fit_nonnegative_logistic(x_mara[calibration_idx], y_cal, mara_feature_cols)
    axis["risk_mara"] = np.clip(head.predict_proba(x_mara)[:, 1], 1e-6, 1.0 - 1e-6)
    if isinstance(head, mara.NonNegativeLogisticRisk):
        raw_contrib = mara.decomposed_axis_contributions(head, x_mara, mara_axis_idx, interaction_alloc)
        contrib = mara.anchored_axis_contributions(
            x_scaled=x_scaled,
            raw_contrib=raw_contrib,
            risk_prob=axis["risk_mara"].to_numpy(dtype=float),
            model=head,
            axis_idx=axis_idx,
            alpha=0.90,
        )
        axis = pd.concat([axis, contrib], axis=1)
    else:
        for axis_name in available_axes:
            axis[f"contrib_{axis_name}"] = 0.0
    return base, axis, threshold


def run_one(
    bundle: mara.DatasetBundle,
    split_name: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict]:
    run_id = f"{bundle.dataset_id}_{split_name}_seed{args.split_seed}"
    dataset, x_bits, train_idx, calibration_idx, acquisition_idx, test_idx = prepare_bundle(
        bundle, split_name, args.split_seed, args.n_bits
    )
    (out_dir / run_id).mkdir(parents=True, exist_ok=True)
    base, axis, threshold = fit_selection_mara(
        dataset, x_bits, train_idx, calibration_idx, bundle, args, out_dir / run_id
    )

    test_order = test_idx[np.argsort(-axis.iloc[test_idx]["risk_mara"].to_numpy(dtype=float))]
    selected_test_idx = np.asarray(test_order[: args.top_test], dtype=int)
    if len(selected_test_idx) == 0:
        return []
    sim = tanimoto_matrix(x_bits[selected_test_idx], x_bits[acquisition_idx])
    y_all = dataset["y"].to_numpy(dtype=float)
    y_focus = y_all[selected_test_idx]

    base_focus = evaluate_predictions(
        bundle.task_type,
        y_focus,
        mara.BasePredictions(
            pred_mean=base.pred_mean[selected_test_idx],
            pred_std=base.pred_std[selected_test_idx],
            proba_mean=None if base.proba_mean is None else base.proba_mean[selected_test_idx],
            per_seed_predictions=[],
        ),
        threshold,
    )
    records = []
    policy_repeats = {
        "mara_axis": 1,
        "mismatched_axis": 1,
        "nearest_neighbor": 1,
        "uncertainty_pool": 1,
        "random_axis": args.random_repeats,
    }
    for policy, repeats in policy_repeats.items():
        for repeat in range(repeats):
            seed_offset = sum(ord(c) for c in f"{run_id}:{policy}") + repeat * 1009
            rng = np.random.default_rng(args.split_seed + 202 + seed_offset)
            support_idx = select_support(
                policy, selected_test_idx, acquisition_idx, sim, axis, args.budget, rng
            )
            aug_idx = np.unique(np.concatenate([train_idx, np.array(support_idx, dtype=int)]))
            aug = mara.train_base_ensemble(x_bits, y_all, aug_idx, bundle.task_type, args.seeds, args.workers)
            aug_focus_pred = mara.BasePredictions(
                pred_mean=aug.pred_mean[selected_test_idx],
                pred_std=aug.pred_std[selected_test_idx],
                proba_mean=None if aug.proba_mean is None else aug.proba_mean[selected_test_idx],
                per_seed_predictions=[],
            )
            aug_focus = evaluate_predictions(bundle.task_type, y_focus, aug_focus_pred, threshold)
            rec = {
                "dataset_id": bundle.dataset_id,
                "split": split_name,
                "policy": policy,
                "repeat": repeat,
                "top_test": int(len(selected_test_idx)),
                "budget": int(len(support_idx)),
                "support_unique": int(len(set(support_idx))),
                "task_type": bundle.task_type,
                "n_proper_train": int(len(train_idx)),
                "n_calibration": int(len(calibration_idx)),
                "n_acquisition": int(len(acquisition_idx)),
                "n_test": int(len(test_idx)),
            }
            for key, value in base_focus.items():
                rec[f"baseline_{key}"] = value
            for key, value in aug_focus.items():
                rec[f"augmented_{key}"] = value
                if f"baseline_{key}" in rec:
                    rec[f"delta_{key}"] = rec[f"baseline_{key}"] - value
            records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["tdc:Caco2_Wang", "tdc:HIA_Hou", "moleculeace:auto:3"])
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--external-root", default="external")
    parser.add_argument("--top-test", type=int, default=64)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--out", default="artifacts/support_intervention_v0")
    args = parser.parse_args()

    bundles = mara.parse_datasets(args.datasets, Path("data"), Path(args.external_root))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    failures = []
    for bundle in bundles:
        for split in args.splits:
            try:
                all_records.extend(run_one(bundle, split, args, out_dir))
            except Exception as exc:
                failures.append({"dataset_id": bundle.dataset_id, "split": split, "error": repr(exc)})
                print(f"[ERROR] {failures[-1]}")
    frame = pd.DataFrame(all_records)
    frame.to_csv(out_dir / "support_intervention.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({"failures": failures, "n_records": len(all_records)}, indent=2) + "\n", encoding="utf-8")
    if not frame.empty:
        metric_cols = [c for c in frame.columns if c.startswith("delta_")]
        summary = frame.groupby("policy")[metric_cols].agg(["mean", "std", "count"])
        print(summary.to_string())


if __name__ == "__main__":
    main()
