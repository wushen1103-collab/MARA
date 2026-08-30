#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tqdm import tqdm

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

from run_mara_public_suite import (
    AXIS_GROUPS,
    BasePredictions,
    DatasetBundle,
    NonNegativeLogisticRisk,
    QuantileMinMaxScaler,
    add_model_conflict_features,
    add_train_defined_stress_slices,
    anchored_axis_contributions,
    build_axis_features,
    build_mara_design,
    decomposed_axis_contributions,
    define_failure,
    ecfp_matrix,
    empirical_cdf_score,
    ensure_dir,
    feature_columns_for_available_axes,
    find_column,
    fit_isotonic_score,
    fit_nonnegative_logistic,
    fit_rf_error_predictor,
    fit_scalar_logistic,
    isotonic_predict,
    make_group_ood_split,
    make_random_split,
    make_scaffold_split,
    make_temporal_split,
    metric_table,
    mol_from_smiles,
    parse_datasets,
    rank_average_score,
    rate_match_score,
    scaffold_smiles,
    stress_localization_table,
    tanimoto_knn_label_risk,
    train_base_ensemble,
    write_json,
)

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


PRETRAINED_MODELS = {
    "chemberta_mlm": "DeepChem/ChemBERTa-77M-MLM",
    "chemberta_mtr": "DeepChem/ChemBERTa-77M-MTR",
}


def split_dataset(dataset: pd.DataFrame, task_type: str, split_name: str, seed: int) -> pd.Series:
    if split_name == "random":
        return make_random_split(dataset, task_type, seed)
    if split_name == "scaffold":
        return make_scaffold_split(dataset, seed)
    if split_name in {"assay", "source"}:
        group_col = find_column(dataset.columns, ["assay_id", "assay_chembl_id", "source", "source_id", "domain_id", "doc_id"])
        if group_col is None:
            raise RuntimeError("assay/source split requires assay_id/source/doc metadata")
        return make_group_ood_split(dataset, group_col, seed)
    if split_name == "target":
        group_col = find_column(dataset.columns, ["target_chembl_id", "target_id", "uniprot_accession", "uniprot", "target"])
        if group_col is None:
            raise RuntimeError("target split requires target_id/target metadata")
        return make_group_ood_split(dataset, group_col, seed)
    if split_name == "temporal":
        return make_temporal_split(dataset)
    if split_name in {"official", "drugood"} and "preset_split_role" in dataset.columns:
        return dataset["preset_split_role"].copy()
    raise ValueError(f"unknown split: {split_name}")


def train_tabular_ensemble(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    task_type: str,
    seeds: list[int],
    workers: int,
    use_xgb: bool = True,
) -> BasePredictions:
    preds: list[np.ndarray] = []
    probas: list[np.ndarray] = []
    for seed in seeds:
        if use_xgb and xgb is not None:
            if task_type == "classification":
                model = xgb.XGBClassifier(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.035,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=max(1, workers),
                    random_state=seed,
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.035,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    objective="reg:squarederror",
                    tree_method="hist",
                    n_jobs=max(1, workers),
                    random_state=seed,
                )
        else:
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=400, n_jobs=max(1, workers), random_state=seed, class_weight="balanced_subsample")
            else:
                model = RandomForestRegressor(n_estimators=400, n_jobs=max(1, workers), random_state=seed)
        model.fit(features[train_idx], y[train_idx])
        if task_type == "classification":
            proba = model.predict_proba(features)[:, 1]
            probas.append(np.asarray(proba, dtype=float))
            preds.append((proba >= 0.5).astype(float))
        else:
            preds.append(np.asarray(model.predict(features), dtype=float))
    if task_type == "classification":
        proba_stack = np.vstack(probas)
        proba_mean = proba_stack.mean(axis=0)
        return BasePredictions(
            pred_mean=(proba_mean >= 0.5).astype(float),
            pred_std=proba_stack.std(axis=0),
            proba_mean=proba_mean,
            per_seed_predictions=probas,
        )
    pred_stack = np.vstack(preds)
    return BasePredictions(
        pred_mean=pred_stack.mean(axis=0),
        pred_std=pred_stack.std(axis=0),
        proba_mean=None,
        per_seed_predictions=preds,
    )


def smiles_hash(smiles: list[str]) -> str:
    digest = hashlib.sha256()
    for smi in smiles:
        digest.update(smi.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def chemberta_embeddings(
    smiles: list[str],
    dataset_id: str,
    args: argparse.Namespace,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = PRETRAINED_MODELS.get(args.pretrained_model, args.pretrained_model)
    model_slug = model_name.replace("/", "__").replace(":", "_")
    cache_dir = ensure_dir(Path(args.pretrained_feature_cache) / model_slug)
    cache_path = cache_dir / f"{dataset_id}_{len(smiles)}_{smiles_hash(smiles)}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["embeddings"].astype(np.float32)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=args.local_files_only).eval().to(device)
    chunks = []
    for start in tqdm(range(0, len(smiles), args.pretrained_batch_size), desc=f"{args.pretrained_model} embeddings"):
        batch_smiles = smiles[start : start + args.pretrained_batch_size]
        batch = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=args.pretrained_max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
            emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        chunks.append(emb.detach().cpu().numpy().astype(np.float32))
    embeddings = np.vstack(chunks)
    np.savez_compressed(cache_path, embeddings=embeddings, smiles=np.asarray(smiles, dtype=object), model=model_name)
    return embeddings


def write_chemprop_csv(frame: pd.DataFrame, path: Path, include_target: bool = True) -> None:
    out = pd.DataFrame({"smiles": frame["smiles"].astype(str)})
    if include_target:
        out["target"] = pd.to_numeric(frame["y"], errors="coerce")
    out.to_csv(path, index=False)


def read_chemprop_predictions(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    numeric_cols = [c for c in frame.columns if c.lower() != "smiles" and pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric_cols:
        raise RuntimeError(f"No numeric prediction column in {path}")
    return frame[numeric_cols[-1]].to_numpy(dtype=float)


def chemprop_command_env(args: argparse.Namespace, gpu_id: int | None) -> dict[str, str]:
    env = os.environ.copy()
    if args.chemprop_bin_dir:
        env["PATH"] = f"{args.chemprop_bin_dir}:{env.get('PATH', '')}"
    compat = args.chemprop_compat_path
    pythonpath = env.get("PYTHONPATH", "")
    paths = [p for p in [compat, str(Path.cwd())] if p]
    env["PYTHONPATH"] = ":".join(paths + ([pythonpath] if pythonpath else []))
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    return env


def train_chemprop_ensemble(
    dataset: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    task_type: str,
    seeds: list[int],
    out_dir: Path,
    args: argparse.Namespace,
) -> BasePredictions:
    work = ensure_dir(out_dir / "chemprop_work")
    train_csv = work / "proper_train.csv"
    val_csv = work / "validation.csv"
    all_csv = work / "all.csv"
    write_chemprop_csv(dataset.iloc[train_idx], train_csv)
    write_chemprop_csv(dataset.iloc[val_idx], val_csv)
    write_chemprop_csv(dataset, all_csv)
    dataset_type = "classification" if task_type == "classification" else "regression"
    metric = "auc" if task_type == "classification" else "rmse"
    predictions = []
    for i, seed in enumerate(seeds):
        model_dir = work / f"model_seed{seed}"
        if model_dir.exists() and not args.reuse_chemprop:
            shutil.rmtree(model_dir)
        preds_path = work / f"preds_seed{seed}.csv"
        gpu_id = None
        if args.gpu_ids:
            gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
        env = chemprop_command_env(args, gpu_id)
        if not args.reuse_chemprop or not preds_path.exists():
            command = [
                "chemprop_train",
                "--data_path",
                str(train_csv),
                "--dataset_type",
                dataset_type,
                "--save_dir",
                str(model_dir),
                "--separate_val_path",
                str(val_csv),
                "--epochs",
                str(args.chemprop_epochs),
                "--seed",
                str(seed),
                "--smiles_columns",
                "smiles",
                "--target_columns",
                "target",
                "--metric",
                metric,
                "--quiet",
            ]
            if gpu_id is not None:
                command.extend(["--gpu", "0"])
            proc = subprocess.run(command, cwd=Path.cwd(), env=env, capture_output=True, text=True, timeout=args.chemprop_timeout)
            if proc.returncode != 0:
                (work / f"train_seed{seed}.stderr.txt").write_text(proc.stderr[-8000:], encoding="utf-8")
                (work / f"train_seed{seed}.stdout.txt").write_text(proc.stdout[-4000:], encoding="utf-8")
                raise RuntimeError(f"chemprop_train failed for seed {seed}; see {work}")
            pred_cmd = [
                "chemprop_predict",
                "--test_path",
                str(all_csv),
                "--checkpoint_dir",
                str(model_dir),
                "--preds_path",
                str(preds_path),
                "--smiles_columns",
                "smiles",
            ]
            if gpu_id is not None:
                pred_cmd.extend(["--gpu", "0"])
            pred_proc = subprocess.run(pred_cmd, cwd=Path.cwd(), env=env, capture_output=True, text=True, timeout=args.chemprop_timeout)
            if pred_proc.returncode != 0:
                (work / f"predict_seed{seed}.stderr.txt").write_text(pred_proc.stderr[-8000:], encoding="utf-8")
                (work / f"predict_seed{seed}.stdout.txt").write_text(pred_proc.stdout[-4000:], encoding="utf-8")
                raise RuntimeError(f"chemprop_predict failed for seed {seed}; see {work}")
        predictions.append(read_chemprop_predictions(preds_path))
    pred_stack = np.vstack(predictions)
    if task_type == "classification":
        proba_mean = np.clip(pred_stack.mean(axis=0), 1e-6, 1 - 1e-6)
        return BasePredictions(
            pred_mean=(proba_mean >= 0.5).astype(float),
            pred_std=pred_stack.std(axis=0),
            proba_mean=proba_mean,
            per_seed_predictions=[row for row in pred_stack],
        )
    return BasePredictions(
        pred_mean=pred_stack.mean(axis=0),
        pred_std=pred_stack.std(axis=0),
        proba_mean=None,
        per_seed_predictions=[row for row in pred_stack],
    )


def train_base_predictions(
    predictor: str,
    dataset: pd.DataFrame,
    x_bits: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    task_type: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> BasePredictions:
    if predictor == "ecfp_xgb":
        return train_base_ensemble(x_bits, y, train_idx, task_type, args.seeds, args.workers)
    if predictor == "chemberta_xgb":
        emb = chemberta_embeddings(dataset["smiles"].astype(str).tolist(), str(dataset["dataset_id"].iloc[0]), args)
        return train_tabular_ensemble(emb, y, train_idx, task_type, args.seeds, args.workers, use_xgb=True)
    if predictor == "chemprop_dmpnn":
        return train_chemprop_ensemble(dataset, y, train_idx, val_idx, task_type, args.seeds, out_dir, args)
    raise ValueError(f"unknown predictor: {predictor}")


def run_one_multibackbone(bundle: DatasetBundle, split_name: str, predictor: str, args: argparse.Namespace, run_root: Path) -> Path:
    dataset = bundle.frame.copy().reset_index(drop=True)
    dataset["mol"] = dataset["smiles"].map(mol_from_smiles)
    dataset = dataset[dataset["mol"].notna()].reset_index(drop=True)
    dataset["row_id"] = np.arange(len(dataset), dtype=int)
    dataset["scaffold"] = [scaffold_smiles(mol, smi) for mol, smi in zip(dataset["mol"], dataset["smiles"])]
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.sample(n=args.max_samples, random_state=args.split_seed).reset_index(drop=True)
        dataset["row_id"] = np.arange(len(dataset), dtype=int)
    split_role = split_dataset(dataset, bundle.task_type, split_name, args.split_seed)
    dataset["split_role"] = split_role.to_numpy()
    train_idx = np.flatnonzero(dataset["split_role"].to_numpy() == "proper_train")
    val_idx = np.flatnonzero(dataset["split_role"].to_numpy() == "validation")
    test_idx = np.flatnonzero(dataset["split_role"].to_numpy() == "test")
    if min(len(train_idx), len(val_idx), len(test_idx)) < 20:
        raise RuntimeError(f"split too small for {bundle.dataset_id}/{split_name}: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")

    run_id = f"{predictor}_{bundle.dataset_id}_{split_name}_seed{args.split_seed}"
    out_dir = ensure_dir(run_root / run_id)
    if args.skip_existing and (out_dir / "metrics.csv").exists() and (out_dir / "predictions.csv").exists():
        print(f"[SKIP] {run_id} -> {out_dir}")
        return out_dir
    write_json(
        out_dir / "dataset_manifest.json",
        {
            **bundle.metadata,
            "dataset_id": bundle.dataset_id,
            "split": split_name,
            "base_predictor": predictor,
            "n_after_rdkit": int(len(dataset)),
            "n_proper_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "seeds": args.seeds,
        },
    )

    x_bits = ecfp_matrix(dataset["mol"].tolist(), n_bits=args.n_bits)
    base = train_base_predictions(
        predictor=predictor,
        dataset=dataset.drop(columns=["mol"]).copy(),
        x_bits=x_bits,
        y=dataset["y"].to_numpy(dtype=float),
        train_idx=train_idx,
        val_idx=val_idx,
        task_type=bundle.task_type,
        out_dir=out_dir,
        args=args,
    )
    axis = build_axis_features(dataset.drop(columns=["mol"]), x_bits, train_idx, dataset["split_role"], bundle.task_type, args.split_seed)
    axis = add_model_conflict_features(axis, base, bundle.task_type)
    axis = define_failure(axis, bundle.task_type, val_idx, out_dir)

    feature_cols, available_axes = feature_columns_for_available_axes(axis, train_idx)
    if args.drop_axes:
        drop_axes = set(args.drop_axes)
        available_axes = {axis_name: cols for axis_name, cols in available_axes.items() if axis_name not in drop_axes}
        feature_cols = [col for cols in available_axes.values() for col in cols]
    if not feature_cols:
        raise RuntimeError(f"no usable axis features for {run_id}")

    scaler = QuantileMinMaxScaler().fit(axis.iloc[train_idx][feature_cols].to_numpy(dtype=float))
    x_scaled = scaler.transform(axis[feature_cols].to_numpy(dtype=float))
    y_cal = axis.iloc[val_idx]["failure"].to_numpy(dtype=int)
    axis_idx = {
        axis_name: [feature_cols.index(col) for col in cols if col in feature_cols]
        for axis_name, cols in available_axes.items()
    }
    x_mara, mara_feature_cols, mara_axis_idx, interaction_alloc, _ = build_mara_design(
        x_base=x_scaled,
        feature_cols=feature_cols,
        axis_idx=axis_idx,
        use_interactions=bool(args.use_interactions and not args.no_interactions),
    )
    mara = fit_nonnegative_logistic(x_mara[val_idx], y_cal, mara_feature_cols, balanced=args.balanced_mara)
    if isinstance(mara, NonNegativeLogisticRisk):
        mara_raw = mara.predict_proba(x_mara)[:, 1]
        axis["risk_mara"] = np.clip(mara_raw, 1e-6, 1.0 - 1e-6)
        mara_iso = fit_isotonic_score(mara_raw[val_idx], y_cal)
        axis["risk_mara_isotonic"] = np.clip(isotonic_predict(mara_iso, mara_raw), 1e-6, 1.0 - 1e-6)
        raw_contrib = decomposed_axis_contributions(mara, x_mara, mara_axis_idx, interaction_alloc)
        contrib = anchored_axis_contributions(
            x_scaled=x_scaled,
            raw_contrib=raw_contrib,
            risk_prob=axis["risk_mara"].to_numpy(dtype=float),
            model=mara,
            axis_idx=axis_idx,
            alpha=args.anchor_alpha,
        )
        axis = pd.concat([axis, contrib], axis=1)
        write_json(
            out_dir / "mara_model.json",
            {
                "base_predictor": predictor,
                "feature_columns": mara_feature_cols,
                "base_feature_columns": feature_cols,
                "available_axes": available_axes,
                "weights": {col: float(w) for col, w in zip(mara_feature_cols, mara.weights)},
                "intercept": mara.intercept,
                "axis_attribution_anchor_alpha": args.anchor_alpha,
                "use_interactions": bool(args.use_interactions and not args.no_interactions),
                "balanced_mara_loss": bool(args.balanced_mara),
                "drop_axes": sorted(args.drop_axes or []),
            },
        )
    else:
        axis["risk_mara"] = mara.predict_proba(x_mara)[:, 1]
        axis["risk_mara_isotonic"] = axis["risk_mara"]
        for axis_name in available_axes:
            axis[f"contrib_{axis_name}"] = 0.0

    scalar_full = fit_scalar_logistic(x_mara[val_idx], y_cal)
    axis["risk_scalar_full"] = scalar_full.predict_proba(x_mara)[:, 1]
    raw_scalar_full = axis["risk_scalar_full"].to_numpy(dtype=float)
    uq_cols = [c for c in ["a5_ensemble_std", "a5_entropy", "a5_margin_risk"] if c in feature_cols]
    if not uq_cols:
        uq_cols = feature_cols
    uq_idx = [feature_cols.index(c) for c in uq_cols]
    scalar_uq = fit_scalar_logistic(x_scaled[val_idx][:, uq_idx], y_cal)
    axis["risk_uncertainty_only"] = scalar_uq.predict_proba(x_scaled[:, uq_idx])[:, 1]

    def feature_score(cols: list[str]) -> np.ndarray:
        idx = [feature_cols.index(c) for c in cols if c in feature_cols]
        return np.max(x_scaled[:, idx], axis=1) if idx else np.zeros(x_scaled.shape[0], dtype=float)

    axis["risk_knn_tanimoto"] = feature_score(["a1_tanimoto_distance"])
    axis["risk_applicability_domain"] = feature_score(["a1_tanimoto_distance", "a1_low_density", "a1_low_analog_support", "a1_scaffold_unseen"])
    axis["risk_mahalanobis"] = feature_score(["a4_mahalanobis_frontier", "a4_pca_frontier"])
    axis["risk_ensemble_variance"] = feature_score(["a5_ensemble_std", "a5_entropy", "a5_margin_risk"])
    raw_uq = axis["risk_ensemble_variance"].to_numpy(dtype=float)
    isotonic_uq = fit_isotonic_score(raw_uq[val_idx], y_cal)
    axis["risk_isotonic_uq"] = np.clip(isotonic_predict(isotonic_uq, raw_uq), 1e-6, 1.0 - 1e-6)
    axis["risk_conformal_uq"] = np.clip(empirical_cdf_score(raw_uq[val_idx], raw_uq), 1e-6, 1.0 - 1e-6)
    raw_tanimoto = axis["risk_knn_tanimoto"].to_numpy(dtype=float)
    tanimoto_cal = fit_nonnegative_logistic(raw_tanimoto[val_idx].reshape(-1, 1), y_cal, ["risk_knn_tanimoto"])
    axis["risk_calibrated_tanimoto"] = np.clip(tanimoto_cal.predict_proba(raw_tanimoto.reshape(-1, 1))[:, 1], 1e-6, 1.0 - 1e-6)
    raw_ad = axis["risk_applicability_domain"].to_numpy(dtype=float)
    ad_cal = fit_nonnegative_logistic(raw_ad[val_idx].reshape(-1, 1), y_cal, ["risk_applicability_domain"])
    axis["risk_calibrated_ad"] = np.clip(ad_cal.predict_proba(raw_ad.reshape(-1, 1))[:, 1], 1e-6, 1.0 - 1e-6)
    axis["risk_validation_knn_error"] = tanimoto_knn_label_risk(
        x_bits=x_bits,
        reference_idx=val_idx,
        y_reference=y_cal,
        query_row_ids=dataset["row_id"].to_numpy(dtype=int),
        reference_row_ids=dataset.iloc[val_idx]["row_id"].to_numpy(dtype=int),
        k=15,
    )
    rf_error = fit_rf_error_predictor(x_mara[val_idx], y_cal, args.split_seed)
    axis["risk_rf_error_predictor"] = rf_error.predict_proba(x_mara)[:, 1]
    rank_fusion_cols = ["risk_mara", "risk_scalar_full", "risk_validation_knn_error", "risk_calibrated_tanimoto", "risk_rf_error_predictor"]
    axis["risk_mara_rank_fusion"] = rate_match_score(rank_average_score(axis, rank_fusion_cols), y_cal, val_idx, power=1.0)

    axis = add_train_defined_stress_slices(axis, train_idx, out_dir)
    for axis_name in AXIS_GROUPS:
        axis[f"axis_available_{axis_name}"] = int(axis_name in available_axes)

    axis_manifest_cols = [
        "dataset_id",
        "row_id",
        "smiles",
        "y",
        "split_role",
        "scaffold",
        *[c for c in axis.columns if c.startswith(("a1_", "a2_", "a3_", "a4_", "a5_", "slice_", "axis_available_"))],
    ]
    axis[axis_manifest_cols].to_csv(out_dir / "axis_manifest.csv", index=False)
    pred_cols = [
        "dataset_id",
        "row_id",
        "smiles",
        "y",
        "split_role",
        "base_pred",
        "base_proba",
        "failure",
        "risk_uncertainty_only",
        "risk_knn_tanimoto",
        "risk_applicability_domain",
        "risk_mahalanobis",
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
        *[c for c in axis.columns if c.startswith("contrib_")],
    ]
    if "base_abs_error" in axis.columns:
        pred_cols.append("base_abs_error")
    if "base_loss" in axis.columns:
        pred_cols.append("base_loss")
    predictions = axis[pred_cols].copy()
    predictions.insert(0, "base_predictor", predictor)
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    risk_methods = [
        "risk_uncertainty_only",
        "risk_ensemble_variance",
        "risk_calibrated_tanimoto",
        "risk_validation_knn_error",
        "risk_rf_error_predictor",
        "risk_scalar_full",
        "risk_mara",
        "risk_mara_isotonic",
        "risk_mara_rank_fusion",
    ]
    metrics = metric_table(axis, test_idx, risk_methods, bundle.task_type)
    metrics.insert(0, "base_predictor", predictor)
    metrics.insert(0, "split_seed", args.split_seed)
    metrics.insert(0, "split", split_name)
    metrics.insert(0, "task_type", bundle.task_type)
    metrics.insert(0, "dataset_id", bundle.dataset_id)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    loc = stress_localization_table(axis, test_idx)
    if not loc.empty:
        loc.insert(0, "base_predictor", predictor)
        loc.insert(0, "split_seed", args.split_seed)
        loc.insert(0, "split", split_name)
        loc.insert(0, "dataset_id", bundle.dataset_id)
        loc.to_csv(out_dir / "stress_localization.csv", index=False)
    print(f"[DONE] {run_id} -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictors", nargs="+", default=["chemberta_xgb"])
    parser.add_argument("--datasets", nargs="+", default=["moleculeace:auto:3"])
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MARA_WORKERS", "16")))
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--anchor-alpha", type=float, default=0.9)
    parser.add_argument("--use-interactions", action="store_true")
    parser.add_argument("--no-interactions", action="store_true")
    parser.add_argument("--balanced-mara", action="store_true")
    parser.add_argument("--drop-axes", nargs="*", default=[])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--external-root", default="external")
    parser.add_argument("--artifacts-root", default="artifacts/multibackbone_v1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[])
    parser.add_argument("--pretrained-model", default="chemberta_mlm")
    parser.add_argument("--pretrained-feature-cache", default="artifacts/pretrained_feature_cache")
    parser.add_argument("--pretrained-batch-size", type=int, default=64)
    parser.add_argument("--pretrained-max-length", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--chemprop-bin-dir", default="", help="Optional directory containing Chemprop CLI executables")
    parser.add_argument("--chemprop-compat-path", default="", help="Optional compatibility-module directory for Chemprop")
    parser.add_argument("--chemprop-epochs", type=int, default=8)
    parser.add_argument("--chemprop-timeout", type=int, default=3600)
    parser.add_argument("--reuse-chemprop", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    data_root = ensure_dir(args.data_root)
    external_root = ensure_dir(args.external_root)
    run_root = ensure_dir(args.artifacts_root)
    ensure_dir("logs")
    started = time.time()
    bundles = parse_datasets(args.datasets, data_root, external_root)
    print(f"Loaded datasets: {[b.dataset_id for b in bundles]}")
    failures = []
    completed = []
    for predictor in args.predictors:
        for bundle in bundles:
            for split_name in args.splits:
                try:
                    completed.append(str(run_one_multibackbone(bundle, split_name, predictor, args, run_root)))
                except Exception as exc:
                    msg = {"base_predictor": predictor, "dataset_id": bundle.dataset_id, "split": split_name, "error": repr(exc)}
                    print(f"[ERROR] {msg}")
                    failures.append(msg)
    write_json(
        run_root / "suite_manifest.json",
        {
            "predictors": args.predictors,
            "datasets_requested": args.datasets,
            "splits": args.splits,
            "seeds": args.seeds,
            "split_seed": args.split_seed,
            "workers": args.workers,
            "max_samples": args.max_samples,
            "completed": completed,
            "failures": failures,
            "elapsed_seconds": time.time() - started,
        },
    )
    if failures and not completed:
        raise SystemExit("all runs failed")


if __name__ == "__main__":
    main()
