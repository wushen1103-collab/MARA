#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import subprocess
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    jaccard_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


AXIS_GROUPS = {
    "A1_support": ["a1_nn_distance", "a1_low_density", "a1_low_analog_support"],
    "A2_source_domain": ["a2_source_unseen", "a2_source_low_frequency", "a2_centroid_distance"],
    "A3_supervision": ["a3_expected_noise"],
    "A4_frontier": ["a4_mahalanobis_frontier", "a4_pca_frontier", "a4_future_time_gap"],
    "A5_model_conflict": ["a5_disagreement", "a5_entropy", "a5_margin_risk"],
}
AXES = list(AXIS_GROUPS)


@dataclass
class DomainData:
    dataset: str
    protocol: str
    x_views: list[np.ndarray]
    x_support: np.ndarray
    y: np.ndarray
    source: np.ndarray
    year: np.ndarray
    split_role: np.ndarray
    supervision_risk: np.ndarray
    metadata: dict


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def download_with_curl(urls: list[str], destination: Path, minimum_bytes: int = 1024 * 1024) -> None:
    ensure_dir(destination.parent)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for url in urls:
        result = subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "5",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--max-time",
                "1800",
                "-o",
                str(temporary),
                url,
            ],
            check=False,
        )
        if result.returncode == 0 and temporary.exists() and temporary.stat().st_size >= minimum_bytes:
            temporary.replace(destination)
            return
    raise RuntimeError(f"failed to download {destination.name} from all configured mirrors")


def safe_metric(fn) -> float:
    try:
        return float(fn())
    except Exception:
        return float("nan")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def empirical_rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)


def expected_calibration_error(y: np.ndarray, risk: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (risk >= lo) & (risk < hi if hi < 1.0 else risk <= hi)
        if np.any(mask):
            out += float(mask.mean()) * abs(float(y[mask].mean()) - float(risk[mask].mean()))
    return out


def risk_coverage_auc(y: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(risk, kind="stable")
    cumulative = np.cumsum(y[order], dtype=float) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1, dtype=float) / len(y)
    return float(np.trapezoid(cumulative, coverage))


def fit_calibrator(raw: np.ndarray, y: np.ndarray):
    raw = np.asarray(raw, dtype=float).reshape(-1, 1)
    if len(np.unique(y)) < 2 or float(np.std(raw)) <= 1e-12:
        return ("constant", float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)))
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(raw, y)
    return ("logistic", model)


def apply_calibrator(model, raw: np.ndarray) -> np.ndarray:
    if model[0] == "constant":
        return np.full(len(raw), model[1], dtype=float)
    return np.clip(model[1].predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1], 1e-6, 1.0 - 1e-6)


def fit_nonnegative_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    if len(np.unique(y)) < 2:
        return np.zeros(x.shape[1]), float(np.log((np.mean(y) + 1e-5) / (1.0 - np.mean(y) + 1e-5)))
    y = y.astype(float)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    sample_weight = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = theta[:-1], theta[-1]
        p = sigmoid(x @ w + b)
        eps = 1e-8
        loss = -np.mean(sample_weight * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
        loss += 1e-3 * float(np.dot(w, w))
        residual = sample_weight * (p - y) / len(y)
        grad = np.r_[x.T @ residual + 2e-3 * w, residual.sum()]
        return float(loss), grad

    init = np.zeros(x.shape[1] + 1, dtype=float)
    init[-1] = float(np.log((np.mean(y) + 1e-3) / (1.0 - np.mean(y) + 1e-3)))
    result = minimize(objective, init, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * x.shape[1] + [(None, None)])
    return result.x[:-1], float(result.x[-1])


def quantile_scale_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.nanquantile(x, 0.01, axis=0)
    hi = np.nanquantile(x, 0.99, axis=0)
    lo = np.where(np.isfinite(lo), lo, 0.0)
    hi = np.where(np.isfinite(hi), hi, lo + 1.0)
    hi[np.isclose(lo, hi)] = lo[np.isclose(lo, hi)] + 1.0
    return lo, hi


def quantile_scale(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    arr = np.where(np.isfinite(x), x, lo)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def hnsw_support(x: np.ndarray, train_idx: np.ndarray, workers: int, k: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(x[train_idx])
    z = scaler.transform(x).astype(np.float32)
    if z.shape[1] > 32:
        pca = PCA(n_components=32, svd_solver="randomized", random_state=2026)
        pca.fit(z[train_idx])
        z = pca.transform(z).astype(np.float32)
    index = hnswlib.Index(space="l2", dim=z.shape[1])
    index.init_index(max_elements=len(train_idx), ef_construction=160, M=24, random_seed=2026)
    index.add_items(z[train_idx], train_idx.astype(np.int64), num_threads=workers)
    index.set_ef(max(80, k + 10))
    labels, distances = index.knn_query(z, k=min(k + 1, len(train_idx)), num_threads=workers)
    nearest = np.zeros(len(z), dtype=float)
    mean_k = np.zeros(len(z), dtype=float)
    analog_count = np.zeros(len(z), dtype=float)
    train_set = set(train_idx.tolist())
    distance_scale = max(float(np.sqrt(np.quantile(distances, 0.90))), 1e-6)
    for i in range(len(z)):
        d = np.sqrt(np.maximum(distances[i], 0.0))
        if i in train_set:
            d = d[labels[i] != i]
        d = d[:k]
        if not len(d):
            continue
        nearest[i] = d[0] / distance_scale
        mean_k[i] = float(np.mean(d)) / distance_scale
        analog_count[i] = float(np.sum(d <= distance_scale))
    low_analog = 1.0 / np.sqrt(1.0 + analog_count)
    return nearest, mean_k, low_analog


def frontier_features(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(x[train_idx])
    z0 = scaler.transform(x).astype(np.float32)
    n_comp = min(16, z0.shape[1], max(2, len(train_idx) - 1))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2026)
    pca.fit(z0[train_idx])
    z = pca.transform(z0)
    mean = z[train_idx].mean(axis=0)
    std = np.maximum(z[train_idx].std(axis=0), 1e-4)
    maha = np.sqrt(np.mean(((z - mean) / std) ** 2, axis=1))
    recon = np.linalg.norm(z0 - pca.inverse_transform(z), axis=1) / math.sqrt(z0.shape[1])
    return maha, recon


def source_features(x: np.ndarray, source: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = source.astype(str)
    values, counts = np.unique(source[train_idx], return_counts=True)
    count_map = dict(zip(values.tolist(), counts.tolist()))
    freq = np.asarray([count_map.get(v, 0) for v in source], dtype=float)
    unseen = (freq == 0).astype(float)
    low_frequency = 1.0 / np.sqrt(1.0 + freq)
    scaler = StandardScaler().fit(x[train_idx])
    z = scaler.transform(x).astype(np.float32)
    if z.shape[1] > 24:
        pca = PCA(n_components=24, svd_solver="randomized", random_state=2026).fit(z[train_idx])
        z = pca.transform(z).astype(np.float32)
    centroids = np.vstack([z[train_idx][source[train_idx] == value].mean(axis=0) for value in values])
    centroid_distance = np.full(len(z), np.inf, dtype=float)
    for start in range(0, len(z), 4096):
        block = z[start : start + 4096]
        d2 = ((block[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
        centroid_distance[start : start + len(block)] = np.sqrt(d2.min(axis=1))
    return unseen, low_frequency, centroid_distance


def build_static_axes(data: DomainData, train_idx: np.ndarray, workers: int) -> pd.DataFrame:
    nn_distance, low_density, low_analog = hnsw_support(data.x_support, train_idx, workers)
    source_unseen, source_low_frequency, centroid_distance = source_features(data.x_support, data.source, train_idx)
    maha, pca_frontier = frontier_features(data.x_support, train_idx)
    train_year = data.year[train_idx].astype(float)
    year_scale = max(float(np.max(train_year) - np.min(train_year)), 1.0)
    future_gap = np.maximum(data.year.astype(float) - float(np.max(train_year)), 0.0) / year_scale
    return pd.DataFrame(
        {
            "a1_nn_distance": nn_distance,
            "a1_low_density": low_density,
            "a1_low_analog_support": low_analog,
            "a2_source_unseen": source_unseen,
            "a2_source_low_frequency": source_low_frequency,
            "a2_centroid_distance": centroid_distance,
            "a3_expected_noise": data.supervision_risk,
            "a4_mahalanobis_frontier": maha,
            "a4_pca_frontier": pca_frontier,
            "a4_future_time_gap": future_gap,
        }
    )


def corrupt_labels(y: np.ndarray, train_idx: np.ndarray, risk: np.ndarray, strength: float, seed: int) -> tuple[np.ndarray, int]:
    out = y.copy()
    rng = np.random.default_rng(seed)
    rates = np.clip(strength * (0.15 + 0.85 * risk[train_idx]), 0.0, 0.49)
    corrupt = rng.random(len(train_idx)) < rates
    idx = train_idx[corrupt]
    classes = int(np.max(y)) + 1
    if classes == 2:
        out[idx] = 1 - out[idx]
    else:
        offset = rng.integers(1, classes, size=len(idx))
        out[idx] = (out[idx] + offset) % classes
    return out, int(corrupt.sum())


def align_proba(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full((len(proba), n_classes), 1e-8, dtype=float)
    out[:, classes.astype(int)] = proba
    out /= out.sum(axis=1, keepdims=True)
    return out


def train_acs_ensemble(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, seed: int, workers: int) -> np.ndarray:
    models = []
    if xgb is not None:
        models.append(
            xgb.XGBClassifier(
                n_estimators=220,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=workers,
                random_state=seed,
            )
        )
    models.extend(
        [
            ExtraTreesClassifier(n_estimators=240, min_samples_leaf=2, max_features=0.8, n_jobs=workers, random_state=seed + 17),
            RandomForestClassifier(n_estimators=220, min_samples_leaf=2, max_features=0.8, n_jobs=workers, random_state=seed + 31),
        ]
    )
    predictions = []
    for model in models:
        model.fit(x[train_idx], y[train_idx])
        predictions.append(model.predict_proba(x))
    return np.stack(predictions, axis=0)


def train_graph_ensemble(views: list[np.ndarray], y: np.ndarray, train_idx: np.ndarray, seed: int, workers: int) -> np.ndarray:
    n_classes = int(np.max(y)) + 1
    predictions = []
    for view_id, view in enumerate(views):
        scaler = StandardScaler().fit(view[train_idx])
        model = LogisticRegression(
            C=0.1,
            solver="lbfgs",
            max_iter=160,
            tol=1e-4,
            random_state=seed + view_id * 101,
        )
        rng = np.random.default_rng(seed + view_id * 101)
        bootstrap_weight = rng.exponential(scale=1.0, size=len(train_idx))
        bootstrap_weight /= bootstrap_weight.mean()
        model.fit(scaler.transform(view[train_idx]), y[train_idx], sample_weight=bootstrap_weight)
        decision = np.asarray(model.decision_function(scaler.transform(view)), dtype=float)
        decision = np.nan_to_num(decision, nan=0.0, posinf=50.0, neginf=-50.0)
        decision -= decision.max(axis=1, keepdims=True)
        exp_score = np.exp(np.clip(decision, -50.0, 0.0))
        proba = exp_score / np.maximum(exp_score.sum(axis=1, keepdims=True), 1e-12)
        predictions.append(align_proba(proba, model.classes_, n_classes))
    return np.stack(predictions, axis=0)


def load_arxiv_edge_index(data_root: Path, n_nodes: int):
    import torch
    from torch_geometric.utils import to_undirected

    raw_root = find_arxiv_root(data_root)
    with gzip.open(raw_root / "raw" / "edge.csv.gz", "rt") as handle:
        edge = np.loadtxt(handle, delimiter=",", dtype=np.int64)
    edge_index = torch.from_numpy(edge.T).long()
    return to_undirected(edge_index, num_nodes=n_nodes)


def train_modern_graph_ensemble(
    data: DomainData,
    train_idx: np.ndarray,
    noisy_y: np.ndarray,
    seed: int,
    backbone: str,
    data_root: Path,
    device: str,
    epochs: int,
    members: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.nn import GCNConv, SAGEConv

    class GraphModel(torch.nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
            super().__init__()
            layer = GCNConv if backbone == "gcn" else SAGEConv
            self.conv1 = layer(in_channels, hidden_channels)
            self.conv2 = layer(hidden_channels, out_channels)

        def forward(self, features, edges):
            hidden = self.conv1(features, edges)
            hidden = functional.relu(hidden)
            hidden = functional.dropout(hidden, p=0.5, training=self.training)
            return self.conv2(hidden, edges)

    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    features = torch.from_numpy(np.array(data.x_views[0], dtype=np.float32, copy=True)).to(target_device)
    labels = torch.from_numpy(np.asarray(noisy_y, dtype=np.int64)).to(target_device)
    train_tensor = torch.from_numpy(np.asarray(train_idx, dtype=np.int64)).to(target_device)
    edge_index = load_arxiv_edge_index(data_root, len(data.y)).to(target_device)
    n_classes = int(np.max(data.y)) + 1
    predictions = []
    for member in range(members):
        member_seed = seed + member * 1009
        torch.manual_seed(member_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(member_seed)
        model = GraphModel(features.shape[1], 128, n_classes).to(target_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, edge_index)
            loss = functional.cross_entropy(logits[train_tensor], labels[train_tensor])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = functional.softmax(model(features, edge_index), dim=1).cpu().numpy()
        predictions.append(probability)
        del model, optimizer
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    return np.stack(predictions, axis=0)


def add_conflict_axes(
    axis: pd.DataFrame,
    model_proba: np.ndarray,
    primary_model_index: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    out = axis.copy()
    mean_proba = model_proba.mean(axis=0) if primary_model_index is None else model_proba[primary_model_index]
    n_classes = mean_proba.shape[1]
    entropy = -(mean_proba * np.log(np.clip(mean_proba, 1e-9, 1.0))).sum(axis=1) / math.log(n_classes)
    disagreement = np.mean(np.var(model_proba, axis=0), axis=1)
    sorted_p = np.sort(mean_proba, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    out["a5_disagreement"] = disagreement
    out["a5_entropy"] = entropy
    out["a5_margin_risk"] = 1.0 - margin
    return out, mean_proba, np.argmax(mean_proba, axis=1)


def validation_knn_error(x: np.ndarray, val_idx: np.ndarray, failure: np.ndarray, workers: int, k: int = 25) -> np.ndarray:
    scaler = StandardScaler().fit(x[val_idx])
    z = scaler.transform(x).astype(np.float32)
    dim = z.shape[1]
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=len(val_idx), ef_construction=120, M=20, random_seed=2027)
    index.add_items(z[val_idx], val_idx.astype(np.int64), num_threads=workers)
    index.set_ef(max(80, k + 10))
    labels, distances = index.knn_query(z, k=min(k + 1, len(val_idx)), num_threads=workers)
    out = np.zeros(len(x), dtype=float)
    val_set = set(val_idx.tolist())
    for i in range(len(x)):
        lab, dist = labels[i], np.sqrt(np.maximum(distances[i], 0.0))
        if i in val_set:
            keep = lab != i
            lab, dist = lab[keep], dist[keep]
        lab, dist = lab[:k], dist[:k]
        weight = 1.0 / (dist + 1e-3)
        out[i] = float(np.sum(weight * failure[lab]) / np.sum(weight)) if len(lab) else float(failure[val_idx].mean())
    return out


def attribution_metrics(true_causes: np.ndarray, contribution: np.ndarray) -> dict:
    valid = true_causes.sum(axis=1) > 0
    if not np.any(valid):
        return {"macro_f1": np.nan, "micro_f1": np.nan, "jaccard": np.nan, "exact_match": np.nan, "precision_at_k": np.nan, "recall_at_k": np.nan}
    truth = true_causes[valid].astype(int)
    score = contribution[valid]
    prediction = np.zeros_like(truth)
    precisions, recalls = [], []
    for i in range(len(truth)):
        k = int(truth[i].sum())
        top = np.argsort(-score[i])[:k]
        prediction[i, top] = 1
        hit = int(truth[i, top].sum())
        precisions.append(hit / max(k, 1))
        recalls.append(hit / max(int(truth[i].sum()), 1))
    return {
        "macro_f1": safe_metric(lambda: f1_score(truth, prediction, average="macro", zero_division=0)),
        "micro_f1": safe_metric(lambda: f1_score(truth, prediction, average="micro", zero_division=0)),
        "jaccard": safe_metric(lambda: jaccard_score(truth, prediction, average="samples", zero_division=0)),
        "exact_match": float(np.mean(np.all(truth == prediction, axis=1))),
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
    }


def evaluate_seed(
    data: DomainData,
    seed: int,
    workers: int,
    noise_strength: float,
    out_dir: Path,
    graph_backbone: str = "sgc",
    graph_data_root: Path | None = None,
    graph_device: str = "cuda:0",
    graph_epochs: int = 80,
    graph_ensemble_members: int = 3,
) -> tuple[pd.DataFrame, dict]:
    train_idx = np.flatnonzero(data.split_role == "proper_train")
    val_idx = np.flatnonzero(data.split_role == "validation")
    test_idx = np.flatnonzero(data.split_role == "test")
    static_axis = build_static_axes(data, train_idx, workers)
    noisy_y, corruption_n = corrupt_labels(data.y, train_idx, data.supervision_risk, noise_strength, seed)
    if data.dataset.startswith("ACS"):
        model_proba = train_acs_ensemble(data.x_views[0], noisy_y, train_idx, seed, workers)
    elif graph_backbone == "sgc":
        model_proba = train_graph_ensemble(data.x_views, noisy_y, train_idx, seed, workers)
    else:
        if graph_data_root is None:
            raise ValueError("graph_data_root is required for a modern graph backbone")
        model_proba = train_modern_graph_ensemble(
            data,
            train_idx,
            noisy_y,
            seed,
            graph_backbone,
            graph_data_root,
            graph_device,
            graph_epochs,
            graph_ensemble_members,
        )
    primary_model_index = -1 if data.dataset == "OGBN-Arxiv" and graph_backbone == "sgc" else None
    axis, mean_proba, prediction = add_conflict_axes(static_axis, model_proba, primary_model_index=primary_model_index)
    failure = (prediction != data.y).astype(int)

    feature_cols = [c for axis_name in AXES for c in AXIS_GROUPS[axis_name]]
    raw_x = axis[feature_cols].to_numpy(dtype=float)
    lo, hi = quantile_scale_fit(raw_x[train_idx])
    x_scaled = quantile_scale(raw_x, lo, hi)
    axis_scores = np.column_stack(
        [np.max(x_scaled[:, [feature_cols.index(c) for c in AXIS_GROUPS[name]]], axis=1) for name in AXES]
    )
    val_y = failure[val_idx]
    weights, intercept = fit_nonnegative_logistic(axis_scores[val_idx], val_y)
    mara_raw = sigmoid(axis_scores @ weights + intercept)
    mara = apply_calibrator(fit_calibrator(mara_raw[val_idx], val_y), mara_raw)
    uncertainty_raw = np.maximum(axis["a5_entropy"].to_numpy(), axis["a5_margin_risk"].to_numpy())
    support_raw = np.maximum(axis["a1_nn_distance"].to_numpy(), axis["a1_low_density"].to_numpy())
    frontier_raw = np.maximum(axis["a4_mahalanobis_frontier"].to_numpy(), axis["a4_pca_frontier"].to_numpy())
    scalar_raw = axis_scores.mean(axis=1)
    val_knn_raw = validation_knn_error(axis_scores, val_idx, failure, workers)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=workers, class_weight="balanced", random_state=seed)
    rf.fit(axis_scores[val_idx], val_y)
    rf_raw = rf.predict_proba(axis_scores)[:, 1]
    methods = {
        "max_softmax_risk": apply_calibrator(
            fit_calibrator((1.0 - mean_proba.max(axis=1))[val_idx], val_y), 1.0 - mean_proba.max(axis=1)
        ),
        "predictive_entropy": apply_calibrator(
            fit_calibrator(axis["a5_entropy"].to_numpy(dtype=float)[val_idx], val_y),
            axis["a5_entropy"].to_numpy(dtype=float),
        ),
        "ensemble_disagreement": apply_calibrator(
            fit_calibrator(axis["a5_disagreement"].to_numpy(dtype=float)[val_idx], val_y),
            axis["a5_disagreement"].to_numpy(dtype=float),
        ),
        "uncertainty_composite": apply_calibrator(fit_calibrator(uncertainty_raw[val_idx], val_y), uncertainty_raw),
        "representation_support": apply_calibrator(fit_calibrator(support_raw[val_idx], val_y), support_raw),
        "distribution_frontier": apply_calibrator(fit_calibrator(frontier_raw[val_idx], val_y), frontier_raw),
        "scalar_axis_mean": apply_calibrator(fit_calibrator(scalar_raw[val_idx], val_y), scalar_raw),
        "validation_knn_error": apply_calibrator(fit_calibrator(val_knn_raw[val_idx], val_y), val_knn_raw),
        "rf_error_predictor": apply_calibrator(fit_calibrator(rf_raw[val_idx], val_y), rf_raw),
        "mara_nonnegative": mara,
    }
    fusion_raw = np.mean(np.column_stack([empirical_rank(methods[name]) for name in ["mara_nonnegative", "scalar_axis_mean", "validation_knn_error", "rf_error_predictor", "uncertainty_composite"]]), axis=1)
    methods["mara_rank_fusion"] = apply_calibrator(fit_calibrator(fusion_raw[val_idx], val_y), fusion_raw)

    rows = []
    for name, risk in methods.items():
        y_test, r_test = failure[test_idx], risk[test_idx]
        rows.append(
            {
                "dataset": data.dataset,
                "protocol": data.protocol,
                "seed": seed,
                "base_predictor": graph_backbone if data.dataset == "OGBN-Arxiv" else "tabular_ensemble",
                "method": name,
                "n_train": len(train_idx),
                "n_validation": len(val_idx),
                "n_test": len(test_idx),
                "corrupted_train_labels": corruption_n,
                "base_accuracy": float(np.mean(prediction[test_idx] == data.y[test_idx])),
                "failure_rate": float(y_test.mean()),
                "failure_auroc": safe_metric(lambda: roc_auc_score(y_test, r_test)),
                "failure_auprc": safe_metric(lambda: average_precision_score(y_test, r_test)),
                "risk_brier": safe_metric(lambda: brier_score_loss(y_test, r_test)),
                "risk_nll": safe_metric(lambda: log_loss(y_test, r_test, labels=[0, 1])),
                "risk_ece": expected_calibration_error(y_test, r_test),
                "aurc": risk_coverage_auc(y_test, r_test),
            }
        )

    learned = np.maximum(axis_scores * weights[None, :], 0.0)
    attribution_anchor_alpha = 0.90
    anchored = (1.0 - attribution_anchor_alpha) * learned + attribution_anchor_alpha * axis_scores
    denom = np.maximum(anchored.sum(axis=1, keepdims=True), 1e-12)
    contribution = anchored / denom
    train_threshold = np.quantile(axis_scores[train_idx], 0.90, axis=0)
    operational_causes = axis_scores > train_threshold[None, :]
    attribution = attribution_metrics(operational_causes[test_idx], contribution[test_idx])
    attribution.update(
        {
            "dataset": data.dataset,
            "protocol": data.protocol,
            "seed": seed,
            "base_predictor": graph_backbone if data.dataset == "OGBN-Arxiv" else "tabular_ensemble",
            "label_type": "train-defined operational stress proxy (not causal ground truth)",
            "failed_test_n": int(failure[test_idx].sum()),
            "compound_attribution_rate": float(np.mean((contribution[test_idx] >= 0.15).sum(axis=1) >= 2)),
        }
    )

    seed_out = ensure_dir(out_dir / f"seed{seed}")
    pred_frame = pd.DataFrame(
        {
            "row_id": test_idx,
            "y": data.y[test_idx],
            "prediction": prediction[test_idx],
            "failure": failure[test_idx],
            "source": data.source[test_idx],
            "year": data.year[test_idx],
            **{name: values[test_idx] for name, values in methods.items()},
            **{f"axis_score_{name}": axis_scores[test_idx, i] for i, name in enumerate(AXES)},
            **{f"contrib_{name}": contribution[test_idx, i] for i, name in enumerate(AXES)},
        }
    )
    pred_frame.to_csv(seed_out / "test_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(rows).to_csv(seed_out / "metrics.csv", index=False)
    write_json(
        seed_out / "mara_model.json",
        {
            "axis_names": AXES,
            "feature_columns": feature_cols,
            "axis_groups": AXIS_GROUPS,
            "weights": dict(zip(AXES, weights.tolist())),
            "intercept": intercept,
            "attribution_anchor_alpha": attribution_anchor_alpha,
            "base_prediction_policy": "2-hop SGC multinomial softmax with Bayesian-bootstrap training weights" if data.dataset == "OGBN-Arxiv" else "equal ensemble mean",
            "noise_strength": noise_strength,
            "corrupted_train_labels": corruption_n,
        },
    )
    write_json(
        seed_out / "data_manifest.json",
        {
            "dataset": data.dataset,
            "protocol": data.protocol,
            "n_total": int(len(data.y)),
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "class_count": int(np.max(data.y) + 1),
            "source_count": int(len(np.unique(data.source))),
            "year_min": int(np.min(data.year)),
            "year_max": int(np.max(data.year)),
            "transductive_graph_features": bool(data.dataset == "OGBN-Arxiv"),
            **data.metadata,
        },
    )
    return pd.DataFrame(rows), attribution


def summarize_runs(metrics: pd.DataFrame, attribution: pd.DataFrame, out: Path) -> None:
    metrics.to_csv(out / "metrics_long.csv", index=False)
    metric_cols = ["base_accuracy", "failure_rate", "failure_auroc", "failure_auprc", "risk_brier", "risk_nll", "risk_ece", "aurc"]
    summary = metrics.groupby(["dataset", "protocol", "method"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary.columns]
    summary.to_csv(out / "metrics_mean_std.csv", index=False)
    attribution.to_csv(out / "attribution_operational_long.csv", index=False)
    attr_cols = ["macro_f1", "micro_f1", "jaccard", "exact_match", "precision_at_k", "recall_at_k", "compound_attribution_rate"]
    attr_summary = attribution.groupby(["dataset", "protocol"], as_index=False)[attr_cols].agg(["mean", "std"])
    attr_summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in attr_summary.columns]
    attr_summary.to_csv(out / "attribution_operational_mean_std.csv", index=False)
    try:
        summary.to_markdown(out / "table_metrics_mean_std.md", index=False)
        attr_summary.to_markdown(out / "table_attribution_operational.md", index=False)
    except Exception:
        pass


def supervision_score(x: np.ndarray) -> np.ndarray:
    col = x[:, min(5, x.shape[1] - 1)].astype(float)
    med = float(np.median(col))
    scale = max(float(np.std(col)), 1e-6)
    return sigmoid((col - med) / scale)


ACS_HF_FILES = [
    "test-00000-of-00001-724549452f5f5dcb.parquet",
    "train-00000-of-00002-a71537175a4688ae.parquet",
    "train-00001-of-00002-b62ecdf1723bbfaf.parquet",
]


def prepare_acs_hf(data_root: Path) -> list[Path]:
    mirror_root = ensure_dir(data_root / "hf_mirror")
    paths = []
    for filename in ACS_HF_FILES:
        destination = mirror_root / filename
        if not destination.exists() or destination.stat().st_size < 10 * 1024 * 1024:
            relative = f"datasets/birkhoffg/folktables-acs-income/resolve/main/data/{filename}"
            download_with_curl(
                [f"https://hf-mirror.com/{relative}", f"https://huggingface.co/{relative}"],
                destination,
                minimum_bytes=10 * 1024 * 1024,
            )
        paths.append(destination)
    return paths


def load_acs_cell(data_root: Path, year: int, state: str, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    cache = ensure_dir(data_root / "processed") / f"acs_income_{year}_{state}.npz"
    if cache.exists():
        data = np.load(cache)
        x, y = data["x"], data["y"]
    else:
        import pyarrow.dataset as ds

        files = prepare_acs_hf(data_root)
        feature_cols = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
        dataset = ds.dataset([str(path) for path in files], format="parquet")
        table = dataset.to_table(
            columns=[*feature_cols, "PINCP", "STATE", "YEAR"],
            filter=(ds.field("YEAR") == int(year)) & (ds.field("STATE") == state),
        )
        frame = table.to_pandas()
        if frame.empty:
            raise RuntimeError(f"ACS mirror contains no rows for {state}/{year}")
        x = frame[feature_cols].fillna(-1).to_numpy(dtype=np.float32)
        raw_y = pd.to_numeric(frame["PINCP"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = raw_y.astype(np.int64) if len(np.unique(raw_y)) <= 2 else (raw_y > 50000).astype(np.int64)
        np.savez_compressed(cache, x=x, y=y)
    if max_samples and len(y) > max_samples:
        state_code = sum(ord(c) for c in state)
        rng = np.random.default_rng(year * 1000 + state_code)
        idx = np.sort(rng.choice(len(y), size=max_samples, replace=False))
        x, y = x[idx], y[idx]
    return x, y


def load_acs_protocol(data_root: Path, protocol: str, max_samples: int) -> DomainData:
    definitions = {
        "state_shift": [(2018, "CA", "proper_train"), (2018, "TX", "validation"), (2018, "MI", "test")],
        "temporal_shift": [(2014, "CA", "proper_train"), (2016, "CA", "validation"), (2018, "CA", "test")],
        "state_time_compound": [(2014, "CA", "proper_train"), (2016, "TX", "validation"), (2018, "MI", "test")],
    }
    if protocol not in definitions:
        raise ValueError(f"unknown ACS protocol: {protocol}")
    xs, ys, sources, years, roles = [], [], [], [], []
    for year, state, role in definitions[protocol]:
        x, y = load_acs_cell(data_root, year, state, max_samples)
        xs.append(x)
        ys.append(y)
        sources.append(np.full(len(y), state))
        years.append(np.full(len(y), year, dtype=int))
        roles.append(np.full(len(y), role))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    return DomainData(
        dataset="ACSIncome",
        protocol=protocol,
        x_views=[x],
        x_support=x,
        y=y,
        source=np.concatenate(sources),
        year=np.concatenate(years),
        split_role=np.concatenate(roles),
        supervision_risk=supervision_score(x),
        metadata={
            "cells": definitions[protocol],
            "max_samples_per_cell": max_samples,
            "task_definition": "Folktables ACSIncome",
            "distribution_mirror": "birkhoffg/folktables-acs-income via hf-mirror.com with Hugging Face fallback",
            "features": ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"],
        },
    )


def find_arxiv_root(root: Path) -> Path:
    candidates = [root / "arxiv", root / "ogbn_arxiv" / "arxiv", root]
    for candidate in candidates:
        if (candidate / "raw" / "node-feat.csv.gz").exists():
            return candidate
    raise FileNotFoundError("OGBN-Arxiv raw files were not found")


def prepare_arxiv(data_root: Path, workers: int) -> Path:
    processed = ensure_dir(data_root / "processed")
    required = [processed / name for name in ["x0.npy", "x1.npy", "x2.npy", "y.npy", "year.npy", "degree.npy", "source.npy", "split_role.npy"]]
    if all(path.exists() for path in required):
        return processed
    archive = data_root / "arxiv.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        download_with_curl(
            [
                "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
                "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
            ],
            archive,
            minimum_bytes=50 * 1024 * 1024,
        )
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"downloaded OGBN-Arxiv archive is not a valid zip: {archive}")
    extract_root = ensure_dir(data_root / "ogbn_arxiv")
    try:
        raw_root = find_arxiv_root(data_root)
    except FileNotFoundError:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extract_root)
        raw_root = find_arxiv_root(data_root)

    def load_csv_gz(path: Path, dtype) -> np.ndarray:
        with gzip.open(path, "rt") as handle:
            return np.loadtxt(handle, delimiter=",", dtype=dtype)

    x0 = load_csv_gz(raw_root / "raw" / "node-feat.csv.gz", np.float32)
    edge = load_csv_gz(raw_root / "raw" / "edge.csv.gz", np.int64)
    y = load_csv_gz(raw_root / "raw" / "node-label.csv.gz", np.int64).reshape(-1)
    year = load_csv_gz(raw_root / "raw" / "node_year.csv.gz", np.int64).reshape(-1)
    n = len(y)
    adjacency = sp.coo_matrix((np.ones(len(edge), dtype=np.float32), (edge[:, 0], edge[:, 1])), shape=(n, n)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency = adjacency + sp.eye(n, dtype=np.float32, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    norm = sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)
    x1 = np.asarray(norm @ x0, dtype=np.float32)
    x2 = np.asarray(norm @ x1, dtype=np.float32)
    official_train = load_csv_gz(raw_root / "split" / "time" / "train.csv.gz", np.int64).reshape(-1)
    official_valid = load_csv_gz(raw_root / "split" / "time" / "valid.csv.gz", np.int64).reshape(-1)
    official_test = load_csv_gz(raw_root / "split" / "time" / "test.csv.gz", np.int64).reshape(-1)
    role = np.full(n, "", dtype="<U12")
    role[official_train] = "proper_train"
    role[official_valid] = "validation"
    role[official_test] = "test"
    kmeans = MiniBatchKMeans(n_clusters=12, batch_size=8192, n_init=5, random_state=2026)
    kmeans.fit(x2[official_train])
    source = kmeans.predict(x2).astype(np.int16)
    for path, arr in zip(required, [x0, x1, x2, y, year, degree, source, role]):
        np.save(path, arr)
    write_json(
        processed / "manifest.json",
        {
            "dataset": "ogbn-arxiv",
            "n_nodes": n,
            "n_edges_directed_raw": int(len(edge)),
            "feature_dim": int(x0.shape[1]),
            "classes": int(np.max(y) + 1),
            "split": "official time split: train <=2017, validation 2018, test >=2019",
            "graph_views": ["raw text embedding", "one-hop SGC", "two-hop SGC"],
            "source_domain": "12 unsupervised clusters on train SGC2 embeddings; labels excluded",
        },
    )
    return processed


def degree_supervision_score(degree: np.ndarray) -> np.ndarray:
    rank = pd.Series(degree).rank(method="average", pct=True).to_numpy(dtype=float)
    return 1.0 - rank


def load_arxiv_protocol(data_root: Path, protocol: str, seed: int, workers: int) -> DomainData:
    processed = prepare_arxiv(data_root, workers)
    x0 = np.load(processed / "x0.npy", mmap_mode="r")
    x1 = np.load(processed / "x1.npy", mmap_mode="r")
    x2 = np.load(processed / "x2.npy", mmap_mode="r")
    y = np.load(processed / "y.npy")
    year = np.load(processed / "year.npy")
    degree = np.load(processed / "degree.npy")
    source = np.load(processed / "source.npy")
    official_role = np.load(processed / "split_role.npy")
    if protocol == "official_temporal":
        role = official_role
    elif protocol == "structural_group_ood":
        groups = np.unique(source)
        rng = np.random.default_rng(seed)
        ordered = rng.permutation(groups)
        train_groups, val_groups, test_groups = ordered[:8], ordered[8:10], ordered[10:]
        role = np.full(len(y), "", dtype="<U12")
        role[np.isin(source, train_groups)] = "proper_train"
        role[np.isin(source, val_groups)] = "validation"
        role[np.isin(source, test_groups)] = "test"
    else:
        raise ValueError(f"unknown OGBN-Arxiv protocol: {protocol}")
    return DomainData(
        dataset="OGBN-Arxiv",
        protocol=protocol,
        x_views=[x0, x1, x2],
        x_support=x2,
        y=y,
        source=source.astype(str),
        year=year,
        split_role=role,
        supervision_risk=degree_supervision_score(degree),
        metadata={"official_protocol": protocol == "official_temporal", "seed_dependent_group_split": protocol == "structural_group_ood"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MARA cross-domain validation on ACSIncome and OGBN-Arxiv")
    parser.add_argument("--domain", choices=["acs", "arxiv"], required=True)
    parser.add_argument("--protocols", nargs="+")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260831, 20260832, 20260833, 20260834, 20260835])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--noise-strength", type=float, default=0.25)
    parser.add_argument("--max-samples-per-cell", type=int, default=30000)
    parser.add_argument("--graph-backbone", choices=["sgc", "gcn", "sage"], default="sgc")
    parser.add_argument("--graph-device", default="cuda:0")
    parser.add_argument("--graph-epochs", type=int, default=80)
    parser.add_argument("--graph-ensemble-members", type=int, default=3)
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, 64))
    out = ensure_dir(args.out)
    protocols = args.protocols or (["state_shift", "temporal_shift", "state_time_compound"] if args.domain == "acs" else ["official_temporal", "structural_group_ood"])
    all_metrics, all_attribution = [], []
    started = time.time()
    for protocol in protocols:
        for seed in args.seeds:
            print(f"[RUN] domain={args.domain} protocol={protocol} seed={seed}", flush=True)
            if args.domain == "acs":
                data = load_acs_protocol(Path(args.data_root), protocol, args.max_samples_per_cell)
            else:
                data = load_arxiv_protocol(Path(args.data_root), protocol, seed, args.workers)
            run_out = ensure_dir(out / data.dataset / protocol)
            metrics, attribution = evaluate_seed(
                data,
                seed,
                args.workers,
                args.noise_strength,
                run_out,
                graph_backbone=args.graph_backbone,
                graph_data_root=Path(args.data_root),
                graph_device=args.graph_device,
                graph_epochs=args.graph_epochs,
                graph_ensemble_members=args.graph_ensemble_members,
            )
            all_metrics.append(metrics)
            all_attribution.append(attribution)
            print(f"[DONE] {data.dataset}/{protocol}/seed{seed}", flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    attribution = pd.DataFrame(all_attribution)
    summarize_runs(metrics, attribution, out)
    write_json(
        out / "run_manifest.json",
        {
            "domain": args.domain,
            "protocols": protocols,
            "seeds": args.seeds,
            "workers": args.workers,
            "noise_strength": args.noise_strength,
            "graph_backbone": args.graph_backbone,
            "graph_device": args.graph_device,
            "graph_epochs": args.graph_epochs,
            "graph_ensemble_members": args.graph_ensemble_members,
            "elapsed_seconds": time.time() - started,
            "cause_label_policy": "Known corruption strength is an intervention; real-domain attribution uses train-defined operational stress proxies only.",
            "attribution_anchor_alpha": 0.90,
        },
    )
    print(f"Wrote cross-domain results to {out}")


if __name__ == "__main__":
    main()
