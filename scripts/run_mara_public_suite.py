#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")


AXIS_GROUPS = {
    "A1_chemistry": [
        "a1_tanimoto_distance",
        "a1_scaffold_unseen",
        "a1_low_density",
        "a1_low_analog_support",
        "a1_activity_cliff_proxy",
    ],
    "A2_assay_source": [
        "a2_assay_unseen",
        "a2_assay_low_frequency",
        "a2_doc_unseen",
        "a2_doc_low_frequency",
        "a2_assay_type_unseen",
        "a2_assay_type_low_frequency",
        "a2_target_unseen",
        "a2_target_low_frequency",
    ],
    "A3_label_reliability": [
        "a3_relation_censored",
        "a3_low_confidence",
        "a3_missing_confidence",
        "a3_unit_missing",
    ],
    "A4_frontier": [
        "a4_mahalanobis_frontier",
        "a4_pca_frontier",
        "a4_future_year_gap",
        "a4_nearest_year_gap",
        "a4_new_scaffold_flag",
    ],
    "A5_model_conflict": [
        "a5_ensemble_std",
        "a5_entropy",
        "a5_margin_risk",
    ],
}


@dataclass
class DatasetBundle:
    dataset_id: str
    source: str
    task_type: str
    frame: pd.DataFrame
    metadata: dict


@dataclass
class BasePredictions:
    pred_mean: np.ndarray
    pred_std: np.ndarray
    proba_mean: np.ndarray | None
    per_seed_predictions: list[np.ndarray]


class QuantileMinMaxScaler:
    def __init__(self, low_q: float = 0.01, high_q: float = 0.99):
        self.low_q = low_q
        self.high_q = high_q
        self.low_: np.ndarray | None = None
        self.high_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "QuantileMinMaxScaler":
        arr = np.asarray(x, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        self.low_ = np.nanquantile(arr, self.low_q, axis=0)
        self.high_ = np.nanquantile(arr, self.high_q, axis=0)
        same = np.isclose(self.high_, self.low_)
        self.high_[same] = self.low_[same] + 1.0
        self.low_ = np.where(np.isfinite(self.low_), self.low_, 0.0)
        self.high_ = np.where(np.isfinite(self.high_), self.high_, self.low_ + 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.low_ is None or self.high_ is None:
            raise RuntimeError("Scaler is not fit")
        arr = np.asarray(x, dtype=float)
        arr = np.where(np.isfinite(arr), arr, self.low_)
        out = (arr - self.low_) / (self.high_ - self.low_)
        return np.clip(out, 0.0, 1.0)


class ConstantRiskModel:
    def __init__(self, prob: float):
        self.prob = float(np.clip(prob, 1e-5, 1.0 - 1e-5))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(x.shape[0], self.prob, dtype=float)
        return np.column_stack([1.0 - p, p])


class NonNegativeLogisticRisk:
    def __init__(self, weights: np.ndarray, intercept: float, columns: list[str]):
        self.weights = weights.astype(float)
        self.intercept = float(intercept)
        self.columns = columns

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + x @ self.weights

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.decision_function(x)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        return np.column_stack([1.0 - p, p])

    def axis_contributions(self, x: np.ndarray, groups: dict[str, list[int]]) -> pd.DataFrame:
        weighted = np.maximum(x, 0.0) * np.maximum(self.weights, 0.0)
        out = {}
        for axis, idx in groups.items():
            out[f"contrib_{axis}"] = weighted[:, idx].sum(axis=1) if idx else np.zeros(x.shape[0])
        return pd.DataFrame(out)


def build_mara_design(
    x_base: np.ndarray,
    feature_cols: list[str],
    axis_idx: dict[str, list[int]],
    use_interactions: bool,
) -> tuple[np.ndarray, list[str], dict[str, list[int]], dict[int, tuple[str, str]], pd.DataFrame]:
    axis_names = list(axis_idx.keys())
    axis_scores = pd.DataFrame(index=np.arange(x_base.shape[0]))
    for axis_name in axis_names:
        idx = axis_idx.get(axis_name, [])
        axis_scores[axis_name] = np.max(x_base[:, idx], axis=1) if idx else 0.0
    if not use_interactions or len(axis_names) < 2:
        return x_base, feature_cols, axis_idx, {}, axis_scores

    blocks = [x_base]
    columns = list(feature_cols)
    groups = {axis_name: list(idx) for axis_name, idx in axis_idx.items()}
    interaction_alloc: dict[int, tuple[str, str]] = {}
    for i, axis_a in enumerate(axis_names):
        for axis_b in axis_names[i + 1 :]:
            score = (axis_scores[axis_a].to_numpy(dtype=float) * axis_scores[axis_b].to_numpy(dtype=float)).reshape(-1, 1)
            if float(np.nanstd(score)) <= 1e-12:
                continue
            col_name = f"interaction_{axis_a}__{axis_b}"
            col_idx = len(columns)
            columns.append(col_name)
            blocks.append(score)
            interaction_alloc[col_idx] = (axis_a, axis_b)
    x_design = np.hstack(blocks)
    return x_design, columns, groups, interaction_alloc, axis_scores


def decomposed_axis_contributions(
    model: NonNegativeLogisticRisk,
    x_design: np.ndarray,
    groups: dict[str, list[int]],
    interaction_alloc: dict[int, tuple[str, str]],
) -> pd.DataFrame:
    weighted = np.maximum(x_design, 0.0) * np.maximum(model.weights, 0.0)
    out = {f"contrib_{axis}": np.zeros(x_design.shape[0], dtype=float) for axis in groups}
    for axis, idx in groups.items():
        if idx:
            out[f"contrib_{axis}"] += weighted[:, idx].sum(axis=1)
    for col_idx, (axis_a, axis_b) in interaction_alloc.items():
        share = 0.5 * weighted[:, col_idx]
        out[f"contrib_{axis_a}"] += share
        out[f"contrib_{axis_b}"] += share
    return pd.DataFrame(out)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_metric(fn, default=np.nan):
    try:
        return float(fn())
    except Exception:
        return float(default)


def canonicalize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def scaffold_smiles(mol, smiles: str) -> str:
    if mol is None:
        return f"invalid::{smiles}"
    try:
        scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaff = ""
    return scaff if scaff else Chem.MolToSmiles(mol, canonical=True)


def ecfp_matrix(mols: list, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    x = np.zeros((len(mols), n_bits), dtype=np.uint8)
    for i, mol in enumerate(tqdm(mols, desc="ECFP4")):
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        x[i] = arr
    return x


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in columns:
        key = c.lower()
        if any(cand.lower() in key for cand in candidates):
            return c
    return None


def infer_y_column(frame: pd.DataFrame, smiles_col: str) -> str | None:
    preferred = [
        "Y",
        "y",
        "label",
        "Label",
        "target",
        "activity",
        "Activity",
        "pIC50",
        "pic50",
        "pChEMBL Value",
        "exp_mean [pEC50/pKi]",
        "exp_mean",
    ]
    for col in preferred:
        if col in frame.columns and col != smiles_col:
            vals = pd.to_numeric(frame[col], errors="coerce")
            if vals.notna().sum() >= max(30, int(0.5 * len(frame))):
                return col
    best = None
    best_count = -1
    for col in frame.columns:
        if col == smiles_col:
            continue
        key = col.lower()
        if any(skip in key for skip in ["id", "smiles", "scaffold", "split", "fold"]):
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        count = vals.notna().sum()
        if count > best_count and count >= max(30, int(0.5 * len(frame))):
            best = col
            best_count = count
    return best


def standardize_frame(raw: pd.DataFrame, dataset_id: str, source: str) -> DatasetBundle | None:
    smiles_col = find_column(raw.columns, ["smiles", "SMILES", "Drug", "drug", "mol", "molecule"])
    if smiles_col is None:
        return None
    y_col = infer_y_column(raw, smiles_col)
    if y_col is None:
        return None
    frame = raw.copy()
    frame["smiles"] = frame[smiles_col].map(canonicalize_smiles)
    frame["y"] = pd.to_numeric(frame[y_col], errors="coerce")
    frame = frame[frame["smiles"].notna() & frame["y"].notna()].copy()
    frame = frame.drop_duplicates(subset=["smiles", "y"]).reset_index(drop=True)
    if len(frame) < 80:
        return None
    unique = sorted(pd.Series(frame["y"]).dropna().unique().tolist())
    binary_like = len(unique) <= 2 and set(float(v) for v in unique).issubset({0.0, 1.0})
    task_type = "classification" if binary_like else "regression"
    keep = ["smiles", "y"]
    metadata_cols = []
    for col in frame.columns:
        key = col.lower()
        if col in keep or col in (smiles_col, y_col):
            continue
        if col == "preset_split_role" or any(
            token in key
            for token in ["assay", "source", "year", "date", "relation", "confidence", "comment", "cliff", "target", "doc", "unit"]
        ):
            metadata_cols.append(col)
    out = frame[keep + metadata_cols].copy()
    out["dataset_id"] = dataset_id
    out["row_id"] = np.arange(len(out), dtype=int)
    meta = {
        "source": source,
        "smiles_column": smiles_col,
        "target_column": y_col,
        "n_rows": int(len(out)),
        "task_type": task_type,
        "metadata_columns": metadata_cols,
    }
    return DatasetBundle(dataset_id=dataset_id, source=source, task_type=task_type, frame=out, metadata=meta)


def load_tdc_dataset(name: str, data_root: Path) -> DatasetBundle:
    from tdc.single_pred import ADME, Tox

    errors = []
    for cls in (ADME, Tox):
        try:
            obj = cls(name=name, path=str(data_root / "tdc"))
            raw = obj.get_data()
            bundle = standardize_frame(raw, f"tdc_{name}", f"TDC:{cls.__name__}")
            if bundle is None:
                raise RuntimeError("could not standardize TDC frame")
            return bundle
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError(f"Could not load TDC dataset {name}: {' | '.join(errors)}")


def clone_or_update_moleculeace(repo_dir: Path) -> Path:
    if (repo_dir / ".git").exists():
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/molML/MoleculeACE.git", str(repo_dir)],
        check=True,
    )
    return repo_dir


def load_moleculeace_auto(limit: int, repo_dir: Path) -> list[DatasetBundle]:
    try:
        clone_or_update_moleculeace(repo_dir)
    except Exception as exc:
        print(f"[WARN] MoleculeACE clone failed: {exc}")
        return []
    bundles: list[DatasetBundle] = []
    for csv_path in sorted(repo_dir.rglob("*.csv")):
        if len(bundles) >= limit:
            break
        try:
            raw = pd.read_csv(csv_path)
        except Exception:
            continue
        dataset_id = "moleculeace_" + csv_path.stem.replace(" ", "_").replace("/", "_")
        bundle = standardize_frame(raw, dataset_id, f"MoleculeACE:{csv_path.relative_to(repo_dir)}")
        if bundle is None:
            continue
        bundle.metadata["source_path"] = str(csv_path)
        bundles.append(bundle)
    return bundles


def load_drugood_json(path: Path) -> DatasetBundle:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    split_obj = obj.get("split", obj)
    if not isinstance(split_obj, dict):
        raise RuntimeError(f"DrugOOD JSON has no split mapping: {path}")

    def role_for_split(name: str) -> str:
        key = name.lower()
        if "train" in key:
            return "proper_train"
        if "val" in key or "valid" in key or "dev" in key:
            return "validation"
        if "ood" in key and "test" in key:
            return "test"
        if key.endswith("test") or "test" in key:
            return "test"
        return "unused"

    rows = []
    for split_name, records in split_obj.items():
        if not isinstance(records, list):
            continue
        role = role_for_split(split_name)
        for item in records:
            if not isinstance(item, dict):
                continue
            smiles = item.get("smiles") or item.get("SMILES") or item.get("drug")
            label = item.get("cls_label", item.get("label", item.get("y", item.get("reg_label"))))
            if smiles is None or label is None:
                continue
            row = {
                "smiles": canonicalize_smiles(str(smiles)),
                "y": label,
                "preset_split_role": role,
                "drugood_split": split_name,
            }
            domain = item.get("domain_id", item.get("domain", item.get("group")))
            if domain is not None:
                row["domain_id"] = domain
                row["source_id"] = domain
                if "assay" in path.stem.lower():
                    row["assay_id"] = domain
            for key in [
                "assay_id",
                "assay_chembl_id",
                "target",
                "target_id",
                "target_chembl_id",
                "scaffold",
                "relation",
                "standard_relation",
                "confidence_score",
                "standard_units",
                "year",
                "publication_year",
            ]:
                if key in item and key not in row:
                    row[key] = item[key]
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"DrugOOD JSON yielded no usable rows: {path}")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame = frame[frame["smiles"].notna() & frame["y"].notna()].copy()
    frame = frame.drop_duplicates(subset=["smiles", "y", "preset_split_role"]).reset_index(drop=True)
    if len(frame) < 80:
        raise RuntimeError(f"DrugOOD JSON too small after cleaning: {path} ({len(frame)} rows)")
    if not {"proper_train", "validation", "test"}.issubset(set(frame["preset_split_role"])):
        train_mask = frame["preset_split_role"] == "proper_train"
        if train_mask.sum() >= 40 and (frame["preset_split_role"] == "validation").sum() == 0:
            train_ids = frame.index[train_mask].to_numpy()
            train_keep, val = train_test_split(train_ids, test_size=0.2, random_state=20260811)
            frame.loc[val, "preset_split_role"] = "validation"
            frame.loc[train_keep, "preset_split_role"] = "proper_train"
    if not {"proper_train", "validation", "test"}.issubset(set(frame["preset_split_role"])):
        counts = frame["preset_split_role"].value_counts().to_dict()
        raise RuntimeError(f"DrugOOD split lacks train/validation/test roles in {path}: {counts}")
    unique = sorted(pd.Series(frame["y"]).dropna().unique().tolist())
    binary_like = len(unique) <= 2 and set(float(v) for v in unique).issubset({0.0, 1.0})
    task_type = "classification" if binary_like else "regression"
    frame["dataset_id"] = "drugood_" + path.stem.replace(" ", "_")
    frame["row_id"] = np.arange(len(frame), dtype=int)
    meta = {
        "source": f"DrugOOD:{path}",
        "source_path": str(path),
        "smiles_column": "smiles",
        "target_column": "cls_label/label/y/reg_label",
        "n_rows": int(len(frame)),
        "task_type": task_type,
        "metadata_columns": [c for c in frame.columns if c not in {"smiles", "y", "dataset_id", "row_id"}],
    }
    return DatasetBundle(dataset_id=str(frame["dataset_id"].iloc[0]), source="DrugOOD", task_type=task_type, frame=frame, metadata=meta)


def make_random_split(frame: pd.DataFrame, task_type: str, seed: int) -> pd.Series:
    idx = np.arange(len(frame))
    strat = frame["y"] if task_type == "classification" and frame["y"].nunique() == 2 else None
    train_val, test = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    strat_tv = frame.iloc[train_val]["y"] if strat is not None else None
    train, val = train_test_split(train_val, test_size=0.25, random_state=seed + 1, stratify=strat_tv)
    role = pd.Series("test", index=frame.index)
    role.iloc[train] = "proper_train"
    role.iloc[val] = "validation"
    role.iloc[test] = "test"
    return role


def make_scaffold_split(frame: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    groups = frame.groupby("scaffold", sort=False).indices
    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    buckets = {"proper_train": [], "validation": [], "test": []}
    targets = {"proper_train": 0.6 * len(frame), "validation": 0.2 * len(frame), "test": 0.2 * len(frame)}
    counts = {k: 0 for k in buckets}
    for _, members in ordered:
        members = np.asarray(members)
        if len(members) <= 1:
            rng.shuffle(members)
        deficits = {k: targets[k] - counts[k] for k in buckets}
        chosen = max(deficits, key=deficits.get)
        buckets[chosen].extend(members.tolist())
        counts[chosen] += len(members)
    role = pd.Series("test", index=frame.index)
    for key, members in buckets.items():
        role.iloc[members] = key
    return role


def make_group_ood_split(frame: pd.DataFrame, group_col: str, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    groups = frame.groupby(group_col, sort=False).indices
    ordered = list(groups.items())
    rng.shuffle(ordered)
    ordered = sorted(ordered, key=lambda kv: len(kv[1]), reverse=True)
    buckets = {"proper_train": [], "validation": [], "test": []}
    targets = {"proper_train": 0.6 * len(frame), "validation": 0.2 * len(frame), "test": 0.2 * len(frame)}
    counts = {k: 0 for k in buckets}
    for _, members in ordered:
        members = np.asarray(members)
        deficits = {k: targets[k] - counts[k] for k in buckets}
        chosen = max(deficits, key=deficits.get)
        buckets[chosen].extend(members.tolist())
        counts[chosen] += len(members)
    role = pd.Series("test", index=frame.index)
    for key, members in buckets.items():
        role.iloc[members] = key
    return role


def make_temporal_split(frame: pd.DataFrame) -> pd.Series:
    year_col = find_column(frame.columns, ["publication_year", "curation_year", "year"])
    if year_col is None:
        raise RuntimeError("temporal split requires publication_year/curation_year/year metadata")
    year = pd.to_numeric(frame[year_col], errors="coerce")
    if year.notna().sum() < max(30, int(0.5 * len(frame))):
        raise RuntimeError("temporal split has too few valid year values")
    ordered = frame.assign(_year=year.fillna(year.median())).sort_values(["_year", "row_id"]).index.to_numpy()
    n_train = int(round(0.6 * len(ordered)))
    n_val = int(round(0.2 * len(ordered)))
    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    role = pd.Series("test", index=frame.index)
    role.loc[train] = "proper_train"
    role.loc[val] = "validation"
    role.loc[test] = "test"
    return role


def tanimoto_support_features(
    x_bits: np.ndarray,
    train_idx: np.ndarray,
    row_ids: np.ndarray,
    k: int = 5,
    analog_threshold: float = 0.4,
    chunk_size: int = 512,
) -> pd.DataFrame:
    x = x_bits.astype(np.float32, copy=False)
    train_x = x[train_idx]
    train_sum = train_x.sum(axis=1)
    row_to_train_pos = {int(row_ids[i]): pos for pos, i in enumerate(train_idx)}
    out_max = np.zeros(x.shape[0], dtype=float)
    out_topk_mean = np.zeros(x.shape[0], dtype=float)
    out_analog_count = np.zeros(x.shape[0], dtype=float)
    for start in tqdm(range(0, x.shape[0], chunk_size), desc="Tanimoto support"):
        end = min(x.shape[0], start + chunk_size)
        xb = x[start:end]
        inter = xb @ train_x.T
        union = xb.sum(axis=1)[:, None] + train_sum[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        for local, row_id in enumerate(row_ids[start:end]):
            pos = row_to_train_pos.get(int(row_id))
            if pos is not None:
                sim[local, pos] = -1.0
        sim = np.maximum(sim, 0.0)
        if sim.shape[1] == 0:
            continue
        kk = min(k, sim.shape[1])
        part = np.partition(sim, kth=sim.shape[1] - kk, axis=1)[:, -kk:]
        out_max[start:end] = part.max(axis=1)
        out_topk_mean[start:end] = part.mean(axis=1)
        out_analog_count[start:end] = (sim >= analog_threshold).sum(axis=1)
    return pd.DataFrame(
        {
            "a1_nn_tanimoto": out_max,
            "a1_tanimoto_distance": 1.0 - out_max,
            "a1_low_density": 1.0 - out_topk_mean,
            "a1_low_analog_support": 1.0 / np.sqrt(1.0 + out_analog_count),
            "a1_analog_count": out_analog_count,
        }
    )


def tanimoto_knn_label_risk(
    x_bits: np.ndarray,
    reference_idx: np.ndarray,
    y_reference: np.ndarray,
    query_row_ids: np.ndarray,
    reference_row_ids: np.ndarray,
    k: int = 15,
    chunk_size: int = 512,
) -> np.ndarray:
    x = x_bits.astype(np.float32, copy=False)
    ref_x = x[reference_idx]
    ref_sum = ref_x.sum(axis=1)
    ref_y = np.asarray(y_reference, dtype=float)
    default = float(np.mean(ref_y)) if len(ref_y) else 0.5
    out = np.full(x.shape[0], default, dtype=float)
    ref_row_to_pos = {int(row_id): pos for pos, row_id in enumerate(reference_row_ids)}
    for start in range(0, x.shape[0], chunk_size):
        end = min(x.shape[0], start + chunk_size)
        xb = x[start:end]
        inter = xb @ ref_x.T
        union = xb.sum(axis=1)[:, None] + ref_sum[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        for local, row_id in enumerate(query_row_ids[start:end]):
            pos = ref_row_to_pos.get(int(row_id))
            if pos is not None:
                sim[local, pos] = -1.0
        sim = np.maximum(sim, 0.0)
        if sim.shape[1] == 0:
            continue
        kk = min(k, sim.shape[1])
        top_pos = np.argpartition(sim, kth=sim.shape[1] - kk, axis=1)[:, -kk:]
        top_sim = np.take_along_axis(sim, top_pos, axis=1)
        top_y = ref_y[top_pos]
        weights = top_sim + 1e-6
        out[start:end] = (weights * top_y).sum(axis=1) / weights.sum(axis=1)
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def frontier_features(x_bits: np.ndarray, train_idx: np.ndarray, seed: int) -> pd.DataFrame:
    x = x_bits.astype(np.float32, copy=False)
    n_comp = max(2, min(64, len(train_idx) - 1, x.shape[1]))
    if n_comp < 2:
        return pd.DataFrame({"a4_mahalanobis_frontier": np.zeros(len(x)), "a4_pca_frontier": np.zeros(len(x))})
    pca = PCA(n_components=n_comp, random_state=seed)
    z_train = pca.fit_transform(x[train_idx])
    z = pca.transform(x)
    mean = z_train.mean(axis=0)
    cov = np.cov(z_train, rowvar=False)
    cov = np.atleast_2d(cov) + np.eye(n_comp) * 1e-4
    inv_cov = np.linalg.pinv(cov)
    centered = z - mean
    maha = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inv_cov, centered), 0.0))
    pca_frontier = np.linalg.norm(centered, axis=1)
    return pd.DataFrame({"a4_mahalanobis_frontier": maha, "a4_pca_frontier": pca_frontier})


def activity_cliff_proxy(y: np.ndarray, sim_features: pd.DataFrame, task_type: str) -> np.ndarray:
    if task_type == "classification":
        return np.zeros_like(y, dtype=float)
    # Without labels from test in axis features, this proxy is used only for MoleculeACE slice construction
    # when an explicit cliff column is unavailable. It is not included in risk training for held-out rows.
    return np.zeros_like(y, dtype=float)


def train_base_ensemble(
    x_bits: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    task_type: str,
    seeds: list[int],
    workers: int,
) -> BasePredictions:
    preds: list[np.ndarray] = []
    probas: list[np.ndarray] = []
    for seed in seeds:
        if xgb is not None:
            if task_type == "classification":
                model = xgb.XGBClassifier(
                    n_estimators=350,
                    max_depth=5,
                    learning_rate=0.03,
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
                    n_estimators=450,
                    max_depth=5,
                    learning_rate=0.03,
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
                model = RandomForestClassifier(n_estimators=500, n_jobs=max(1, workers), random_state=seed)
            else:
                model = RandomForestRegressor(n_estimators=500, n_jobs=max(1, workers), random_state=seed)
        model.fit(x_bits[train_idx], y[train_idx])
        if task_type == "classification":
            proba = model.predict_proba(x_bits)[:, 1]
            probas.append(proba.astype(float))
            preds.append((proba >= 0.5).astype(float))
        else:
            pred = model.predict(x_bits)
            preds.append(np.asarray(pred, dtype=float))
    if task_type == "classification":
        proba_mean = np.vstack(probas).mean(axis=0)
        pred_mean = (proba_mean >= 0.5).astype(float)
        pred_std = np.vstack(probas).std(axis=0)
        return BasePredictions(pred_mean=pred_mean, pred_std=pred_std, proba_mean=proba_mean, per_seed_predictions=probas)
    pred_stack = np.vstack(preds)
    return BasePredictions(
        pred_mean=pred_stack.mean(axis=0),
        pred_std=pred_stack.std(axis=0),
        proba_mean=None,
        per_seed_predictions=preds,
    )


def add_model_conflict_features(frame: pd.DataFrame, base: BasePredictions, task_type: str) -> pd.DataFrame:
    out = frame.copy()
    out["base_pred"] = base.pred_mean
    out["a5_ensemble_std"] = base.pred_std
    if task_type == "classification":
        p = np.clip(base.proba_mean, 1e-6, 1.0 - 1e-6)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)
        margin_risk = 1.0 - np.abs(p - 0.5) * 2.0
        out["base_proba"] = p
        out["a5_entropy"] = entropy
        out["a5_margin_risk"] = margin_risk
    else:
        out["base_proba"] = np.nan
        out["a5_entropy"] = 0.0
        out["a5_margin_risk"] = 0.0
    return out


def build_axis_features(
    frame: pd.DataFrame,
    x_bits: np.ndarray,
    train_idx: np.ndarray,
    split_role: pd.Series,
    task_type: str,
    seed: int,
) -> pd.DataFrame:
    row_ids = frame["row_id"].to_numpy()
    sim = tanimoto_support_features(x_bits, train_idx=train_idx, row_ids=row_ids)
    frontier = frontier_features(x_bits, train_idx=train_idx, seed=seed)
    out = pd.concat([frame.reset_index(drop=True), sim, frontier], axis=1)
    train_scaffolds = set(out.loc[split_role.to_numpy() == "proper_train", "scaffold"].astype(str))
    out["a1_scaffold_unseen"] = (~out["scaffold"].astype(str).isin(train_scaffolds)).astype(float)
    explicit_cliff_col = find_column(out.columns, ["cliff_mol", "cliff", "activity_cliff"])
    if explicit_cliff_col is not None:
        out["a1_activity_cliff_proxy"] = pd.to_numeric(out[explicit_cliff_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        out["a1_activity_cliff_proxy"] = activity_cliff_proxy(out["y"].to_numpy(), sim, task_type)
    out["a2_missing_mask"] = 1.0
    out["a3_missing_mask"] = 1.0
    train_mask = split_role.to_numpy() == "proper_train"

    def metadata_col(candidates: list[str]) -> str | None:
        return find_column(out.columns, candidates)

    def categorical_support(col: str | None, prefix: str) -> None:
        if col is None:
            out[f"{prefix}_unseen"] = 0.0
            out[f"{prefix}_low_frequency"] = 0.0
            return
        train_vals = out.loc[train_mask, col].astype(str)
        counts = train_vals.value_counts(dropna=False)
        vals = out[col].astype(str)
        freq = vals.map(counts).fillna(0).astype(float)
        out[f"{prefix}_unseen"] = (freq <= 0).astype(float)
        out[f"{prefix}_low_frequency"] = 1.0 / np.sqrt(1.0 + freq)

    categorical_support(metadata_col(["assay_id", "assay_chembl_id", "assay"]), "a2_assay")
    categorical_support(metadata_col(["doc_id", "document_id", "source_doc", "source", "source_id", "domain_id"]), "a2_doc")
    categorical_support(metadata_col(["assay_type", "endpoint_type"]), "a2_assay_type")
    categorical_support(metadata_col(["target_chembl_id", "target_id", "target"]), "a2_target")

    relation_col = metadata_col(["standard_relation", "relation"])
    if relation_col is not None:
        relation = out[relation_col].astype(str).str.strip()
        out["a3_relation_censored"] = (~relation.isin(["=", "==", "nan", "None", ""])).astype(float)
    else:
        out["a3_relation_censored"] = 0.0
    conf_col = metadata_col(["confidence_score", "confidence"])
    if conf_col is not None:
        conf = pd.to_numeric(out[conf_col], errors="coerce")
        train_conf = pd.to_numeric(out.loc[train_mask, conf_col], errors="coerce")
        cmin = float(np.nanmin(train_conf)) if train_conf.notna().any() else 0.0
        cmax = float(np.nanmax(train_conf)) if train_conf.notna().any() else 1.0
        if abs(cmax - cmin) <= 1e-12:
            out["a3_low_confidence"] = 0.0
        else:
            out["a3_low_confidence"] = 1.0 - ((conf - cmin) / (cmax - cmin)).clip(0.0, 1.0)
        out["a3_missing_confidence"] = conf.isna().astype(float)
    else:
        out["a3_low_confidence"] = 0.0
        out["a3_missing_confidence"] = 0.0
    unit_col = metadata_col(["standard_units", "unit"])
    if unit_col is not None:
        out["a3_unit_missing"] = (
            out[unit_col].isna() | (out[unit_col].astype(str).str.len() == 0)
        ).astype(float)
    else:
        out["a3_unit_missing"] = 0.0

    year_col = metadata_col(["publication_year", "curation_year", "year"])
    if year_col is not None:
        year = pd.to_numeric(out[year_col], errors="coerce")
        train_year = pd.to_numeric(out.loc[train_mask, year_col], errors="coerce").dropna()
        if len(train_year):
            max_year = float(train_year.max())
            min_year = float(train_year.min())
            scale = max(1.0, max_year - min_year)
            unique_years = np.sort(train_year.unique().astype(float))
            vals = year.fillna(max_year).to_numpy(dtype=float)
            out["a4_future_year_gap"] = np.maximum(vals - max_year, 0.0) / scale
            nearest = np.zeros(len(vals), dtype=float)
            for i, value in enumerate(vals):
                nearest[i] = float(np.min(np.abs(unique_years - value))) / scale
            out["a4_nearest_year_gap"] = nearest
        else:
            out["a4_future_year_gap"] = 0.0
            out["a4_nearest_year_gap"] = 0.0
    else:
        out["a4_future_year_gap"] = 0.0
        out["a4_nearest_year_gap"] = 0.0
    first_scaffold_col = metadata_col(["is_first_seen_scaffold"])
    if first_scaffold_col is not None:
        out["a4_new_scaffold_flag"] = pd.to_numeric(out[first_scaffold_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        out["a4_new_scaffold_flag"] = 0.0
    return out


def define_failure(frame: pd.DataFrame, task_type: str, val_idx: np.ndarray, out_dir: Path) -> pd.DataFrame:
    out = frame.copy()
    y = out["y"].to_numpy(dtype=float)
    if task_type == "classification":
        p = np.clip(out["base_proba"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        pred_label = (p >= 0.5).astype(float)
        out["base_loss"] = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        out["failure"] = (pred_label != y).astype(int)
        definition = {
            "task_type": task_type,
            "failure_event": "predicted_label_mismatch",
            "threshold_source": "validation labels; no test residual in threshold",
            "validation_failure_rate": float(out.iloc[val_idx]["failure"].mean()),
        }
    else:
        err = np.abs(y - out["base_pred"].to_numpy(dtype=float))
        threshold = float(np.quantile(err[val_idx], 0.8))
        out["base_abs_error"] = err
        out["failure"] = (err > threshold).astype(int)
        definition = {
            "task_type": task_type,
            "failure_event": "absolute_error_above_validation_80th_percentile",
            "threshold": threshold,
            "threshold_source": "validation residuals only",
            "validation_failure_rate": float(out.iloc[val_idx]["failure"].mean()),
        }
    (out_dir / "failure_definition.yaml").write_text(yaml.safe_dump(definition, sort_keys=True), encoding="utf-8")
    return out


def feature_columns_for_available_axes(frame: pd.DataFrame, train_idx: np.ndarray) -> tuple[list[str], dict[str, list[str]]]:
    available: dict[str, list[str]] = {}
    for axis, cols in AXIS_GROUPS.items():
        keep = []
        for col in cols:
            if col not in frame.columns:
                continue
            vals = pd.to_numeric(frame.iloc[train_idx][col], errors="coerce").to_numpy(dtype=float)
            if np.nanstd(vals) > 1e-8:
                keep.append(col)
        if keep:
            available[axis] = keep
    columns = [col for cols in available.values() for col in cols]
    return columns, available


def fit_nonnegative_logistic(
    x: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    l2: float = 1e-3,
    balanced: bool = False,
) -> NonNegativeLogisticRisk | ConstantRiskModel:
    y = y.astype(float)
    if len(np.unique(y)) < 2:
        return ConstantRiskModel(float(np.mean(y)))
    n_features = x.shape[1]
    if balanced:
        pos = float(np.mean(y))
        sample_weight = np.where(y > 0.5, 0.5 / max(pos, 1e-6), 0.5 / max(1.0 - pos, 1e-6))
        sample_weight = sample_weight / np.mean(sample_weight)
    else:
        sample_weight = np.ones_like(y, dtype=float)

    def objective(theta: np.ndarray):
        w = theta[:n_features]
        b = theta[-1]
        z = b + x @ w
        per_sample = np.logaddexp(0.0, z) - y * z
        loss = np.mean(sample_weight * per_sample) + l2 * np.sum(w * w)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        residual = sample_weight * (p - y)
        grad_w = x.T @ residual / len(y) + 2.0 * l2 * w
        grad_b = np.array([np.mean(residual)])
        grad = np.concatenate([grad_w, grad_b])
        return loss, grad

    init_p = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
    init = np.zeros(n_features + 1, dtype=float)
    init[-1] = math.log(init_p / (1.0 - init_p))
    bounds = [(0.0, None)] * n_features + [(None, None)]
    result = minimize(lambda th: objective(th), init, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    if not result.success:
        print(f"[WARN] nonnegative logistic optimizer: {result.message}")
    theta = result.x
    return NonNegativeLogisticRisk(theta[:n_features], theta[-1], columns)


def fit_scalar_logistic(x: np.ndarray, y: np.ndarray):
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return ConstantRiskModel(float(np.mean(y)))
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    model.fit(x, y)
    return model


def fit_rf_error_predictor(x: np.ndarray, y: np.ndarray, seed: int):
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return ConstantRiskModel(float(np.mean(y)))
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def fit_isotonic_score(score: np.ndarray, y: np.ndarray):
    y = y.astype(float)
    if len(np.unique(y)) < 2 or float(np.nanstd(score)) <= 1e-12:
        return ConstantRiskModel(float(np.mean(y)))
    model = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1.0 - 1e-5)
    model.fit(np.asarray(score, dtype=float), y)
    return model


def isotonic_predict(model, score: np.ndarray) -> np.ndarray:
    if isinstance(model, ConstantRiskModel):
        return model.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]
    return np.asarray(model.predict(np.asarray(score, dtype=float)), dtype=float)


def empirical_cdf_score(reference_score: np.ndarray, score: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_score, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return np.full(len(score), 0.5, dtype=float)
    ref = np.sort(ref)
    vals = np.asarray(score, dtype=float)
    vals = np.where(np.isfinite(vals), vals, np.nanmedian(ref))
    return np.searchsorted(ref, vals, side="right") / len(ref)


def rate_match_score(score: np.ndarray, y_cal: np.ndarray, cal_idx: np.ndarray, power: float = 2.0) -> np.ndarray:
    raw = np.asarray(score, dtype=float)
    finite = np.isfinite(raw)
    fill = float(np.nanmedian(raw[finite])) if np.any(finite) else 0.0
    raw = np.where(finite, raw, fill)
    shaped = np.power(np.clip(raw, 1e-6, 1.0), power)
    target_rate = float(np.mean(y_cal)) if len(y_cal) else 0.5
    cal_mean = float(np.mean(shaped[cal_idx])) if len(cal_idx) else float(np.mean(shaped))
    scale = target_rate / max(cal_mean, 1e-6)
    return np.clip(shaped * scale, 1e-6, 1.0 - 1e-6)


def rank_average_score(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    ranks = []
    for col in columns:
        score = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(score)
        fill = float(np.nanmedian(score[finite])) if np.any(finite) else 0.0
        score = np.where(finite, score, fill)
        ranks.append((pd.Series(score).rank(method="average").to_numpy(dtype=float) - 1.0) / max(1, len(score) - 1))
    return np.mean(ranks, axis=0)


def expected_calibration_error(y: np.ndarray, risk: np.ndarray, bins: int = 10) -> float:
    y = y.astype(float)
    risk = np.asarray(risk, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (risk >= lo) & (risk < hi if hi < 1.0 else risk <= hi)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(float(y[mask].mean()) - float(risk[mask].mean()))
    return float(ece)


def risk_coverage(y: np.ndarray, risk: np.ndarray) -> tuple[float, dict[str, float]]:
    order = np.argsort(risk)
    coverages = np.linspace(0.1, 1.0, 91)
    curve = []
    points: dict[str, float] = {}
    for cov in coverages:
        n = max(1, int(round(cov * len(y))))
        val = float(np.mean(y[order[:n]]))
        curve.append(val)
    auc = float(np.trapezoid(curve, coverages) / (coverages[-1] - coverages[0]))
    for cov in [0.70, 0.80, 0.90, 0.95]:
        n = max(1, int(round(cov * len(y))))
        points[f"selective_risk_at_{int(cov * 100)}"] = float(np.mean(y[order[:n]]))
    return auc, points


def stress_slices(frame: pd.DataFrame, test_idx: np.ndarray) -> dict[str, np.ndarray]:
    test = frame.iloc[test_idx]
    predefined = {}
    for name in [
        "chemistry_stress",
        "assay_source_stress",
        "label_reliability_stress",
        "frontier_stress",
        "model_conflict_stress",
        "activity_cliff_stress",
    ]:
        col = f"slice_{name}"
        if col in test.columns:
            predefined[name] = test[col].to_numpy(dtype=bool)
    if predefined:
        return predefined

    def qmask(col: str, q: float = 0.8) -> np.ndarray:
        vals = test[col].to_numpy(dtype=float)
        thr = float(np.nanquantile(vals, q)) if np.isfinite(vals).any() else np.inf
        return vals >= thr

    chemistry = (test["a1_scaffold_unseen"].to_numpy(dtype=float) > 0.5) | qmask("a1_tanimoto_distance")
    frontier = qmask("a4_mahalanobis_frontier")
    conflict = qmask("a5_ensemble_std") | qmask("a5_entropy") | qmask("a5_margin_risk")
    cliff = test.get("a1_activity_cliff_proxy", pd.Series(0.0, index=test.index)).to_numpy(dtype=float) > 0.5
    return {
        "chemistry_stress": chemistry,
        "assay_source_stress": np.zeros(len(test), dtype=bool),
        "label_reliability_stress": np.zeros(len(test), dtype=bool),
        "frontier_stress": frontier,
        "model_conflict_stress": conflict,
        "activity_cliff_stress": cliff,
    }


def add_train_defined_stress_slices(frame: pd.DataFrame, train_idx: np.ndarray, out_dir: Path) -> pd.DataFrame:
    out = frame.copy()
    train = out.iloc[train_idx]

    def threshold(col: str, q: float = 0.8) -> float:
        vals = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return float("inf")
        if float(np.std(vals)) <= 1e-10:
            return float("inf")
        return float(np.quantile(vals, q))

    thresholds = {
        "a1_tanimoto_distance_q80_train": threshold("a1_tanimoto_distance"),
        "a1_low_density_q80_train": threshold("a1_low_density"),
        "a2_assay_low_frequency_q80_train": threshold("a2_assay_low_frequency"),
        "a2_doc_low_frequency_q80_train": threshold("a2_doc_low_frequency"),
        "a2_assay_type_low_frequency_q80_train": threshold("a2_assay_type_low_frequency"),
        "a2_target_low_frequency_q80_train": threshold("a2_target_low_frequency"),
        "a3_low_confidence_q80_train": threshold("a3_low_confidence"),
        "a4_mahalanobis_frontier_q80_train": threshold("a4_mahalanobis_frontier"),
        "a4_pca_frontier_q80_train": threshold("a4_pca_frontier"),
        "a5_ensemble_std_q80_train": threshold("a5_ensemble_std"),
        "a5_entropy_q80_train": threshold("a5_entropy"),
        "a5_margin_risk_q80_train": threshold("a5_margin_risk"),
    }
    out["slice_chemistry_stress"] = (
        (out["a1_scaffold_unseen"].to_numpy(dtype=float) > 0.5)
        | (out["a1_tanimoto_distance"].to_numpy(dtype=float) >= thresholds["a1_tanimoto_distance_q80_train"])
        | (out["a1_low_density"].to_numpy(dtype=float) >= thresholds["a1_low_density_q80_train"])
    ).astype(int)
    out["slice_assay_source_stress"] = (
        (out["a2_assay_unseen"].to_numpy(dtype=float) > 0.5)
        | (out["a2_doc_unseen"].to_numpy(dtype=float) > 0.5)
        | (out["a2_assay_type_unseen"].to_numpy(dtype=float) > 0.5)
        | (out["a2_target_unseen"].to_numpy(dtype=float) > 0.5)
        | (out["a2_assay_low_frequency"].to_numpy(dtype=float) >= thresholds["a2_assay_low_frequency_q80_train"])
        | (out["a2_doc_low_frequency"].to_numpy(dtype=float) >= thresholds["a2_doc_low_frequency_q80_train"])
        | (out["a2_assay_type_low_frequency"].to_numpy(dtype=float) >= thresholds["a2_assay_type_low_frequency_q80_train"])
        | (out["a2_target_low_frequency"].to_numpy(dtype=float) >= thresholds["a2_target_low_frequency_q80_train"])
    ).astype(int)
    out["slice_label_reliability_stress"] = (
        (out["a3_relation_censored"].to_numpy(dtype=float) > 0.5)
        | (out["a3_missing_confidence"].to_numpy(dtype=float) > 0.5)
        | (out["a3_unit_missing"].to_numpy(dtype=float) > 0.5)
        | (out["a3_low_confidence"].to_numpy(dtype=float) >= thresholds["a3_low_confidence_q80_train"])
    ).astype(int)
    out["slice_frontier_stress"] = (
        (out["a4_mahalanobis_frontier"].to_numpy(dtype=float) >= thresholds["a4_mahalanobis_frontier_q80_train"])
        | (out["a4_pca_frontier"].to_numpy(dtype=float) >= thresholds["a4_pca_frontier_q80_train"])
    ).astype(int)
    out["slice_model_conflict_stress"] = (
        (out["a5_ensemble_std"].to_numpy(dtype=float) >= thresholds["a5_ensemble_std_q80_train"])
        | (out["a5_entropy"].to_numpy(dtype=float) >= thresholds["a5_entropy_q80_train"])
        | (out["a5_margin_risk"].to_numpy(dtype=float) >= thresholds["a5_margin_risk_q80_train"])
    ).astype(int)
    out["slice_activity_cliff_stress"] = (out["a1_activity_cliff_proxy"].to_numpy(dtype=float) > 0.5).astype(int)

    def ratio_score(col: str, thr_key: str) -> np.ndarray:
        thr = thresholds[thr_key]
        if not np.isfinite(thr) or abs(thr) <= 1e-12:
            return np.zeros(len(out), dtype=float)
        vals = out[col].to_numpy(dtype=float)
        return np.maximum(vals / thr, 0.0)

    out["slice_score_chemistry_stress"] = np.maximum.reduce(
        [
            out["a1_scaffold_unseen"].to_numpy(dtype=float),
            ratio_score("a1_tanimoto_distance", "a1_tanimoto_distance_q80_train"),
            ratio_score("a1_low_density", "a1_low_density_q80_train"),
            out["a1_activity_cliff_proxy"].to_numpy(dtype=float),
        ]
    )
    out["slice_score_assay_source_stress"] = np.maximum.reduce(
        [
            out["a2_assay_unseen"].to_numpy(dtype=float),
            out["a2_doc_unseen"].to_numpy(dtype=float),
            out["a2_assay_type_unseen"].to_numpy(dtype=float),
            out["a2_target_unseen"].to_numpy(dtype=float),
            ratio_score("a2_assay_low_frequency", "a2_assay_low_frequency_q80_train"),
            ratio_score("a2_doc_low_frequency", "a2_doc_low_frequency_q80_train"),
            ratio_score("a2_assay_type_low_frequency", "a2_assay_type_low_frequency_q80_train"),
            ratio_score("a2_target_low_frequency", "a2_target_low_frequency_q80_train"),
        ]
    )
    out["slice_score_label_reliability_stress"] = np.maximum.reduce(
        [
            out["a3_relation_censored"].to_numpy(dtype=float),
            out["a3_missing_confidence"].to_numpy(dtype=float),
            out["a3_unit_missing"].to_numpy(dtype=float),
            ratio_score("a3_low_confidence", "a3_low_confidence_q80_train"),
        ]
    )
    out["slice_score_frontier_stress"] = np.maximum(
        ratio_score("a4_mahalanobis_frontier", "a4_mahalanobis_frontier_q80_train"),
        ratio_score("a4_pca_frontier", "a4_pca_frontier_q80_train"),
    )
    out["slice_score_model_conflict_stress"] = np.maximum.reduce(
        [
            ratio_score("a5_ensemble_std", "a5_ensemble_std_q80_train"),
            ratio_score("a5_entropy", "a5_entropy_q80_train"),
            ratio_score("a5_margin_risk", "a5_margin_risk_q80_train"),
        ]
    )
    out["slice_score_activity_cliff_stress"] = out["a1_activity_cliff_proxy"].to_numpy(dtype=float)
    definition = {
        "source": "proper_train axis feature quantiles only; no test label or test residual",
        "quantile": 0.8,
        "thresholds": thresholds,
        "slice_columns": [
            "slice_chemistry_stress",
            "slice_assay_source_stress",
            "slice_label_reliability_stress",
            "slice_frontier_stress",
            "slice_model_conflict_stress",
            "slice_activity_cliff_stress",
        ],
    }
    (out_dir / "stress_slice_definition.yaml").write_text(yaml.safe_dump(definition, sort_keys=True), encoding="utf-8")
    return out


def anchored_axis_contributions(
    x_scaled: np.ndarray,
    raw_contrib: pd.DataFrame,
    risk_prob: np.ndarray,
    model: NonNegativeLogisticRisk,
    axis_idx: dict[str, list[int]],
    alpha: float,
) -> pd.DataFrame:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return raw_contrib
    anchor = pd.DataFrame(index=raw_contrib.index)
    for axis_name, idx in axis_idx.items():
        if idx:
            anchor[f"contrib_{axis_name}"] = np.max(x_scaled[:, idx], axis=1)
        else:
            anchor[f"contrib_{axis_name}"] = 0.0
    for col in raw_contrib.columns:
        if col not in anchor:
            anchor[col] = 0.0
    anchor = anchor[raw_contrib.columns]
    blended = (1.0 - alpha) * raw_contrib.to_numpy(dtype=float) + alpha * anchor.to_numpy(dtype=float)
    denom = blended.sum(axis=1, keepdims=True)
    denom = np.where(denom > 1e-12, denom, 1.0)
    props = blended / denom
    logit = np.log(np.clip(risk_prob, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - risk_prob, 1e-6, 1.0))
    risk_mass = np.maximum(logit - model.intercept, 0.0)
    anchored = props * risk_mass[:, None]
    return pd.DataFrame(anchored, columns=raw_contrib.columns, index=raw_contrib.index)


def metric_table(frame: pd.DataFrame, test_idx: np.ndarray, risk_cols: list[str], task_type: str) -> pd.DataFrame:
    test = frame.iloc[test_idx].copy()
    y_fail = test["failure"].to_numpy(dtype=int)
    rows = []
    slices = stress_slices(frame, test_idx)
    for method in risk_cols:
        risk = np.clip(test[method].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        rc_auc, rc_points = risk_coverage(y_fail, risk)
        slice_risks = []
        for mask in slices.values():
            if np.any(mask):
                slice_risks.append(float(y_fail[mask].mean()))
        overall = float(y_fail.mean())
        row = {
            "method": method,
            "failure_rate": overall,
            "failure_auroc": safe_metric(lambda: roc_auc_score(y_fail, risk)),
            "failure_auprc": safe_metric(lambda: average_precision_score(y_fail, risk)),
            "risk_nll": safe_metric(lambda: log_loss(y_fail, risk, labels=[0, 1])),
            "risk_brier": safe_metric(lambda: brier_score_loss(y_fail, risk)),
            "risk_ece_10bin": expected_calibration_error(y_fail, risk),
            "risk_coverage_auc": rc_auc,
            "worst_slice_risk": max(slice_risks) if slice_risks else np.nan,
            "coverage_gap_worst_minus_overall": (max(slice_risks) - overall) if slice_risks else np.nan,
        }
        row.update(rc_points)
        if task_type == "regression":
            row["base_mae"] = float(mean_absolute_error(test["y"], test["base_pred"]))
            row["base_rmse"] = float(np.sqrt(mean_squared_error(test["y"], test["base_pred"])))
        else:
            pred = (test["base_proba"].to_numpy(float) >= 0.5).astype(int)
            row["base_accuracy"] = float(np.mean(pred == test["y"].to_numpy(int)))
        rows.append(row)
    return pd.DataFrame(rows)


def stress_localization_table(frame: pd.DataFrame, test_idx: np.ndarray) -> pd.DataFrame:
    test = frame.iloc[test_idx].copy().reset_index(drop=True)
    contrib_cols = [c for c in test.columns if c.startswith("contrib_A")]
    if not contrib_cols:
        return pd.DataFrame()
    axis_labels = [c.replace("contrib_", "") for c in contrib_cols]
    pred_axis = np.array(axis_labels)[np.argmax(test[contrib_cols].to_numpy(dtype=float), axis=1)]
    slices = stress_slices(frame, test_idx)
    records = []
    slice_to_axis = {
        "chemistry_stress": "A1_chemistry",
        "assay_source_stress": "A2_assay_source",
        "label_reliability_stress": "A3_label_reliability",
        "activity_cliff_stress": "A1_chemistry",
        "frontier_stress": "A4_frontier",
        "model_conflict_stress": "A5_model_conflict",
    }
    score_cols = {
        "chemistry_stress": "slice_score_chemistry_stress",
        "assay_source_stress": "slice_score_assay_source_stress",
        "label_reliability_stress": "slice_score_label_reliability_stress",
        "activity_cliff_stress": "slice_score_activity_cliff_stress",
        "frontier_stress": "slice_score_frontier_stress",
        "model_conflict_stress": "slice_score_model_conflict_stress",
    }
    score_frame = pd.DataFrame(
        {
            name: test[col].to_numpy(dtype=float) if col in test.columns else mask.astype(float)
            for name, col in score_cols.items()
            for mask in [slices.get(name, np.zeros(len(test), dtype=bool))]
        }
    )
    any_stress = np.zeros(len(test), dtype=bool)
    for mask in slices.values():
        any_stress |= mask
    dominant_slice = score_frame.idxmax(axis=1).to_numpy()
    y_true = np.array([slice_to_axis[name] for name in dominant_slice])
    y_pred = pred_axis
    eval_mask = any_stress
    overlap_count = np.zeros(len(test), dtype=int)
    for mask in slices.values():
        overlap_count += mask.astype(int)
    for slice_name in slice_to_axis:
        mask = eval_mask & (dominant_slice == slice_name)
        true_axis = slice_to_axis[slice_name]
        if not np.any(mask):
            records.append({"slice": slice_name, "n": 0, "top_axis_accuracy": np.nan, "true_axis": true_axis})
            continue
        acc = float(np.mean(y_pred[mask] == true_axis))
        records.append(
            {
                "slice": slice_name,
                "n": int(mask.sum()),
                "top_axis_accuracy": acc,
                "true_axis": true_axis,
                "mean_overlap_count": float(overlap_count[mask].mean()),
            }
        )
    y_true_eval = y_true[eval_mask].tolist()
    y_pred_eval = y_pred[eval_mask].tolist()
    present_axis_labels = sorted(set(y_true_eval))
    macro_f1 = safe_metric(lambda: f1_score(y_true_eval, y_pred_eval, labels=present_axis_labels, average="macro")) if present_axis_labels else np.nan
    macro_f1_all_axes = safe_metric(
        lambda: f1_score(y_true_eval, y_pred_eval, labels=sorted(set(slice_to_axis.values())), average="macro")
    ) if y_true_eval else np.nan
    records.append(
        {
            "slice": "macro",
            "n": int(len(y_true_eval)),
            "top_axis_accuracy": np.nan,
            "macro_f1": macro_f1,
            "macro_f1_all_axes": macro_f1_all_axes,
            "mean_overlap_count": float(overlap_count[eval_mask].mean()) if np.any(eval_mask) else np.nan,
        }
    )
    return pd.DataFrame(records)


def run_one(bundle: DatasetBundle, split_name: str, args: argparse.Namespace, run_root: Path) -> Path:
    dataset = bundle.frame.copy().reset_index(drop=True)
    dataset["mol"] = dataset["smiles"].map(mol_from_smiles)
    dataset = dataset[dataset["mol"].notna()].reset_index(drop=True)
    dataset["row_id"] = np.arange(len(dataset), dtype=int)
    dataset["scaffold"] = [scaffold_smiles(mol, smi) for mol, smi in zip(dataset["mol"], dataset["smiles"])]
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.sample(n=args.max_samples, random_state=args.split_seed).reset_index(drop=True)
        dataset["row_id"] = np.arange(len(dataset), dtype=int)
    if split_name == "random":
        split_role = make_random_split(dataset, bundle.task_type, args.split_seed)
    elif split_name == "scaffold":
        split_role = make_scaffold_split(dataset, args.split_seed)
    elif split_name in {"assay", "source"}:
        group_col = find_column(dataset.columns, ["assay_id", "assay_chembl_id", "source", "source_id", "domain_id", "doc_id"])
        if group_col is None:
            raise RuntimeError("assay/source split requires assay_id/source/doc metadata")
        split_role = make_group_ood_split(dataset, group_col, args.split_seed)
    elif split_name == "target":
        group_col = find_column(dataset.columns, ["target_chembl_id", "target_id", "uniprot_accession", "uniprot", "target"])
        if group_col is None:
            raise RuntimeError("target split requires target_id/target metadata")
        split_role = make_group_ood_split(dataset, group_col, args.split_seed)
    elif split_name == "temporal":
        split_role = make_temporal_split(dataset)
    elif split_name in {"official", "drugood"} and "preset_split_role" in dataset.columns:
        dataset = dataset[dataset["preset_split_role"].isin(["proper_train", "validation", "test"])].reset_index(drop=True)
        dataset["row_id"] = np.arange(len(dataset), dtype=int)
        split_role = dataset["preset_split_role"].copy()
    else:
        raise ValueError(f"unknown split: {split_name}")
    dataset["split_role"] = split_role.to_numpy()
    train_idx = np.flatnonzero(dataset["split_role"].to_numpy() == "proper_train")
    val_idx = np.flatnonzero(dataset["split_role"].to_numpy() == "validation")
    test_idx = np.flatnonzero(dataset["split_role"].to_numpy() == "test")
    if min(len(train_idx), len(val_idx), len(test_idx)) < 20:
        raise RuntimeError(f"split too small for {bundle.dataset_id}/{split_name}: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")

    run_id = f"{bundle.dataset_id}_{split_name}_seed{args.split_seed}"
    out_dir = ensure_dir(run_root / run_id)
    write_json(
        out_dir / "dataset_manifest.json",
        {
            **bundle.metadata,
            "dataset_id": bundle.dataset_id,
            "split": split_name,
            "n_after_rdkit": int(len(dataset)),
            "n_proper_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "seeds": args.seeds,
        },
    )

    x_bits = ecfp_matrix(dataset["mol"].tolist(), n_bits=args.n_bits)
    base = train_base_ensemble(
        x_bits=x_bits,
        y=dataset["y"].to_numpy(dtype=float),
        train_idx=train_idx,
        task_type=bundle.task_type,
        seeds=args.seeds,
        workers=args.workers,
    )
    axis = build_axis_features(dataset.drop(columns=["mol"]), x_bits, train_idx, dataset["split_role"], bundle.task_type, args.split_seed)
    axis = add_model_conflict_features(axis, base, bundle.task_type)
    axis = define_failure(axis, bundle.task_type, val_idx, out_dir)

    feature_cols, available_axes = feature_columns_for_available_axes(axis, train_idx)
    drop_axes = set(args.drop_axes or [])
    if drop_axes:
        available_axes = {axis_name: cols for axis_name, cols in available_axes.items() if axis_name not in drop_axes}
        feature_cols = [col for cols in available_axes.values() for col in cols]
    if not feature_cols:
        raise RuntimeError(f"no usable axis features for {run_id}")
    scaler = QuantileMinMaxScaler().fit(axis.iloc[train_idx][feature_cols].to_numpy(dtype=float))
    x_scaled = scaler.transform(axis[feature_cols].to_numpy(dtype=float))
    if args.shuffle_axis_features:
        rng = np.random.default_rng(args.split_seed + 991)
        x_work = x_scaled.copy()
        for col_idx in range(x_work.shape[1]):
            rng.shuffle(x_work[:, col_idx])
    else:
        x_work = x_scaled
    y_cal = axis.iloc[val_idx]["failure"].to_numpy(dtype=int)

    axis_idx = {
        axis_name: [feature_cols.index(col) for col in cols if col in feature_cols]
        for axis_name, cols in available_axes.items()
    }
    x_mara, mara_feature_cols, mara_axis_idx, interaction_alloc, axis_scores = build_mara_design(
        x_base=x_work,
        feature_cols=feature_cols,
        axis_idx=axis_idx,
        use_interactions=bool(args.use_interactions and not args.no_interactions),
    )
    mara = fit_nonnegative_logistic(x_mara[val_idx], y_cal, mara_feature_cols, balanced=args.balanced_mara)
    if isinstance(mara, NonNegativeLogisticRisk):
        mara_raw = mara.predict_proba(x_mara)[:, 1]
        mara_iso = fit_isotonic_score(mara_raw[val_idx], y_cal)
        axis["risk_mara"] = np.clip(mara_raw, 1e-6, 1.0 - 1e-6)
        axis["risk_mara_isotonic"] = np.clip(isotonic_predict(mara_iso, mara_raw), 1e-6, 1.0 - 1e-6)
        raw_contrib = decomposed_axis_contributions(mara, x_mara, mara_axis_idx, interaction_alloc)
        contrib = anchored_axis_contributions(
            x_scaled=x_work,
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
                "feature_columns": mara_feature_cols,
                "base_feature_columns": feature_cols,
                "available_axes": available_axes,
                "weights": {col: float(w) for col, w in zip(mara_feature_cols, mara.weights)},
                "intercept": mara.intercept,
                "constraint": "nonnegative feature weights on train-only min-max oriented risk features",
                "axis_attribution_anchor_alpha": args.anchor_alpha,
                "axis_attribution_mode": "risk-proportional blend of learned nonnegative contributions and train-only axis-anchor scores",
                "risk_calibration": "risk_mara is raw nonnegative logistic risk; risk_mara_isotonic is an optional validation isotonic variant",
                "use_interactions": bool(args.use_interactions and not args.no_interactions),
                "balanced_mara_loss": bool(args.balanced_mara),
                "interaction_allocations": {mara_feature_cols[k]: list(v) for k, v in interaction_alloc.items()},
                "drop_axes": sorted(drop_axes),
                "shuffle_axis_features": bool(args.shuffle_axis_features),
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
    axis["risk_mara_dual_guard_rate"] = rate_match_score(
        np.maximum(axis["risk_mara"].to_numpy(dtype=float), raw_scalar_full), y_cal, val_idx, power=2.0
    )
    axis["risk_mara_dual_blend_rate"] = rate_match_score(
        0.5 * axis["risk_mara"].to_numpy(dtype=float) + 0.5 * raw_scalar_full, y_cal, val_idx, power=2.0
    )
    uq_cols = [c for c in ["a5_ensemble_std", "a5_entropy", "a5_margin_risk"] if c in feature_cols]
    if not uq_cols:
        uq_cols = feature_cols
    uq_idx = [feature_cols.index(c) for c in uq_cols]
    scalar_uq = fit_scalar_logistic(x_work[val_idx][:, uq_idx], y_cal)
    axis["risk_uncertainty_only"] = scalar_uq.predict_proba(x_work[:, uq_idx])[:, 1]

    def feature_score(cols: list[str]) -> np.ndarray:
        idx = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not idx:
            return np.zeros(x_work.shape[0], dtype=float)
        return np.max(x_work[:, idx], axis=1)

    axis["risk_knn_tanimoto"] = feature_score(["a1_tanimoto_distance"])
    axis["risk_applicability_domain"] = feature_score(
        ["a1_tanimoto_distance", "a1_low_density", "a1_low_analog_support", "a1_scaffold_unseen"]
    )
    axis["risk_mahalanobis"] = feature_score(["a4_mahalanobis_frontier", "a4_pca_frontier"])
    axis["risk_ensemble_variance"] = feature_score(["a5_ensemble_std", "a5_entropy", "a5_margin_risk"])
    raw_uq = axis["risk_ensemble_variance"].to_numpy(dtype=float)
    isotonic_uq = fit_isotonic_score(raw_uq[val_idx], y_cal)
    axis["risk_isotonic_uq"] = np.clip(isotonic_predict(isotonic_uq, raw_uq), 1e-6, 1.0 - 1e-6)
    axis["risk_conformal_uq"] = np.clip(empirical_cdf_score(raw_uq[val_idx], raw_uq), 1e-6, 1.0 - 1e-6)
    raw_tanimoto = axis["risk_knn_tanimoto"].to_numpy(dtype=float)
    tanimoto_cal = fit_nonnegative_logistic(raw_tanimoto[val_idx].reshape(-1, 1), y_cal, ["risk_knn_tanimoto"])
    axis["risk_calibrated_tanimoto"] = np.clip(
        tanimoto_cal.predict_proba(raw_tanimoto.reshape(-1, 1))[:, 1], 1e-6, 1.0 - 1e-6
    )
    raw_ad = axis["risk_applicability_domain"].to_numpy(dtype=float)
    ad_cal = fit_nonnegative_logistic(raw_ad[val_idx].reshape(-1, 1), y_cal, ["risk_applicability_domain"])
    axis["risk_calibrated_ad"] = np.clip(ad_cal.predict_proba(raw_ad.reshape(-1, 1))[:, 1], 1e-6, 1.0 - 1e-6)
    raw_mara_guard = np.maximum(axis["risk_mara"].to_numpy(dtype=float), raw_ad)
    axis["risk_mara_ad_guard_rate"] = rate_match_score(raw_mara_guard, y_cal, val_idx, power=2.0)
    mara_guard_cal = fit_nonnegative_logistic(raw_mara_guard[val_idx].reshape(-1, 1), y_cal, ["risk_mara_ad_guard"])
    axis["risk_mara_ad_guard"] = np.clip(
        mara_guard_cal.predict_proba(raw_mara_guard.reshape(-1, 1))[:, 1], 1e-6, 1.0 - 1e-6
    )
    mara_guard_iso = fit_isotonic_score(raw_mara_guard[val_idx], y_cal)
    axis["risk_mara_ad_guard_isotonic"] = np.clip(isotonic_predict(mara_guard_iso, raw_mara_guard), 1e-6, 1.0 - 1e-6)
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
    rank_fusion_cols = [
        "risk_mara",
        "risk_scalar_full",
        "risk_validation_knn_error",
        "risk_calibrated_tanimoto",
        "risk_rf_error_predictor",
    ]
    axis["risk_mara_rank_fusion"] = rate_match_score(
        rank_average_score(axis, rank_fusion_cols), y_cal, val_idx, power=1.0
    )

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
        *[c for c in axis.columns if c.startswith("a1_") or c.startswith("a2_") or c.startswith("a3_") or c.startswith("a4_") or c.startswith("a5_")],
        *[c for c in axis.columns if c.startswith("slice_")],
        *[c for c in axis.columns if c.startswith("axis_available_")],
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
        "risk_mara_dual_guard_rate",
        "risk_mara_dual_blend_rate",
        "risk_mara_ad_guard_rate",
        "risk_mara_ad_guard",
        "risk_mara_ad_guard_isotonic",
        "risk_mara_rank_fusion",
        *[c for c in axis.columns if c.startswith("contrib_")],
    ]
    if "base_abs_error" in axis.columns:
        pred_cols.append("base_abs_error")
    if "base_loss" in axis.columns:
        pred_cols.append("base_loss")
    axis[pred_cols].to_csv(out_dir / "predictions.csv", index=False)

    risk_methods = [
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
        "risk_mara_dual_guard_rate",
        "risk_mara_dual_blend_rate",
        "risk_mara_ad_guard_rate",
        "risk_mara_ad_guard",
        "risk_mara_ad_guard_isotonic",
        "risk_mara_rank_fusion",
    ]
    metrics = metric_table(axis, test_idx, risk_methods, bundle.task_type)
    metrics.insert(0, "split", split_name)
    metrics.insert(0, "task_type", bundle.task_type)
    metrics.insert(0, "dataset_id", bundle.dataset_id)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    loc = stress_localization_table(axis, test_idx)
    if not loc.empty:
        loc.insert(0, "split", split_name)
        loc.insert(0, "dataset_id", bundle.dataset_id)
        loc.to_csv(out_dir / "stress_localization.csv", index=False)

    print(f"[DONE] {run_id} -> {out_dir}")
    return out_dir


def parse_datasets(specs: list[str], data_root: Path, external_root: Path) -> list[DatasetBundle]:
    bundles: list[DatasetBundle] = []
    for spec in specs:
        if spec.startswith("tdc:"):
            name = spec.split(":", 1)[1]
            bundles.append(load_tdc_dataset(name, data_root))
        elif spec.startswith("moleculeace:auto"):
            parts = spec.split(":")
            limit = int(parts[2]) if len(parts) >= 3 and parts[2] else 3
            bundles.extend(load_moleculeace_auto(limit, external_root / "MoleculeACE"))
        elif spec.startswith("drugood:auto"):
            parts = spec.split(":")
            limit = int(parts[2]) if len(parts) >= 3 and parts[2] else 12
            candidates = sorted((external_root / "drugood_official").rglob("*.json"))
            candidates = [p for p in candidates if not p.name.lower().startswith("._")]
            for path in candidates[:limit]:
                try:
                    bundles.append(load_drugood_json(path))
                except Exception as exc:
                    print(f"[WARN] skip DrugOOD JSON {path}: {exc}")
        elif spec.startswith("drugood:"):
            pattern = spec.split(":", 1)[1]
            paths = [Path(p) for p in glob.glob(pattern)] if any(ch in pattern for ch in "*?[") else [Path(pattern)]
            for path in sorted(paths):
                bundles.append(load_drugood_json(path))
        else:
            path = Path(spec)
            if path.suffix.lower() == ".json":
                bundles.append(load_drugood_json(path))
            else:
                raw = pd.read_csv(path, sep=None, engine="python")
                bundle = standardize_frame(raw, path.stem, f"csv:{path}")
                if bundle is None:
                    raise RuntimeError(f"could not standardize CSV dataset: {path}")
                bundles.append(bundle)
    dedup: dict[str, DatasetBundle] = {}
    for bundle in bundles:
        dedup[bundle.dataset_id] = bundle
    return list(dedup.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["tdc:Caco2_Wang", "tdc:HIA_Hou", "moleculeace:auto:3"])
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MARA_WORKERS", "64")))
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--anchor-alpha", type=float, default=0.0)
    parser.add_argument("--use-interactions", action="store_true")
    parser.add_argument("--no-interactions", action="store_true", help="Backward-compatible alias for the default no-interaction setting")
    parser.add_argument("--balanced-mara", action="store_true")
    parser.add_argument("--drop-axes", nargs="*", default=[])
    parser.add_argument("--shuffle-axis-features", action="store_true")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--external-root", default="external")
    parser.add_argument("--artifacts-root", default="artifacts/public_v0")
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
    for bundle in bundles:
        for split_name in args.splits:
            try:
                completed.append(str(run_one(bundle, split_name, args, run_root)))
            except Exception as exc:
                msg = {"dataset_id": bundle.dataset_id, "split": split_name, "error": repr(exc)}
                print(f"[ERROR] {msg}")
                failures.append(msg)
    write_json(
        run_root / "suite_manifest.json",
        {
            "datasets_requested": args.datasets,
            "splits": args.splits,
            "seeds": args.seeds,
            "split_seed": args.split_seed,
            "workers": args.workers,
            "completed": completed,
            "failures": failures,
            "elapsed_seconds": time.time() - started,
        },
    )
    if failures and not completed:
        raise SystemExit("all runs failed")


if __name__ == "__main__":
    main()
