#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import hnswlib
import numpy as np
import pandas as pd
import psutil
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0**2)


def fit_nonnegative(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    y = y.astype(float)

    def objective(theta: np.ndarray):
        w, b = theta[:-1], theta[-1]
        p = sigmoid(x @ w + b)
        loss = -np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8)) + 1e-3 * np.dot(w, w)
        residual = (p - y) / len(y)
        return float(loss), np.r_[x.T @ residual + 2e-3 * w, residual.sum()]

    init = np.zeros(x.shape[1] + 1)
    init[-1] = float(np.log((np.mean(y) + 1e-3) / (1 - np.mean(y) + 1e-3)))
    result = minimize(objective, init, jac=True, method="L-BFGS-B", bounds=[(0, None)] * x.shape[1] + [(None, None)])
    return result.x[:-1], float(result.x[-1])


def load_acs_template(template_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    files = sorted(template_root.glob("acs_income_*.npz"))
    if not files:
        return None
    xs, domains, years = [], [], []
    for file_id, path in enumerate(files):
        data = np.load(path)
        x = np.asarray(data["x"], dtype=np.float32)
        match = re.search(r"acs_income_(\d{4})_([A-Z]{2})", path.stem)
        year = int(match.group(1)) if match else 2018
        xs.append(x)
        domains.append(np.full(len(x), file_id, dtype=np.int16))
        years.append(np.full(len(x), year, dtype=np.int16))
    x = np.vstack(xs)
    scaler = StandardScaler().fit(x)
    return scaler.transform(x).astype(np.float32), np.concatenate(domains), np.concatenate(years)


def make_workload(n: int, seed: int, template_root: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    template = load_acs_template(template_root) if template_root else None
    if template is not None:
        base_x, base_domain, base_year = template
        idx = rng.choice(len(base_x), size=n, replace=n > len(base_x))
        x = base_x[idx] + rng.normal(0, 0.015, size=(n, base_x.shape[1])).astype(np.float32)
        return x.astype(np.float32), base_domain[idx], base_year[idx], "ACSIncome empirical bootstrap"
    dim, n_domains = 32, 20
    centers = rng.normal(0, 1.8, size=(n_domains, dim)).astype(np.float32)
    domain = rng.integers(0, n_domains, size=n, dtype=np.int16)
    x = centers[domain] + rng.normal(0, 1.0, size=(n, dim)).astype(np.float32)
    year = rng.integers(2014, 2020, size=n, dtype=np.int16)
    return x, domain, year, "generic Gaussian-mixture representation"


def hnsw_all(
    x: np.ndarray,
    train_n: int,
    workers: int,
    k: int,
    seed: int,
) -> tuple[np.ndarray, hnswlib.Index, float, float, float]:
    index = hnswlib.Index(space="l2", dim=x.shape[1])
    before = rss_mb()
    start = time.perf_counter()
    index.init_index(max_elements=train_n, ef_construction=120, M=16, random_seed=seed)
    index.add_items(x[:train_n], np.arange(train_n, dtype=np.int64), num_threads=workers)
    build_seconds = time.perf_counter() - start
    index.set_ef(max(64, k + 8))
    support = np.empty(len(x), dtype=np.float32)
    query_start = time.perf_counter()
    for start_idx in range(0, len(x), 50000):
        end_idx = min(len(x), start_idx + 50000)
        labels, distances = index.knn_query(x[start_idx:end_idx], k=min(k + 1, train_n), num_threads=workers)
        for local, global_idx in enumerate(range(start_idx, end_idx)):
            d = distances[local]
            lab = labels[local]
            if global_idx < train_n:
                d = d[lab != global_idx]
            support[global_idx] = float(np.sqrt(max(float(d[0]), 0.0))) if len(d) else 0.0
    query_seconds = time.perf_counter() - query_start
    memory_delta = max(rss_mb() - before, 0.0)
    return support, index, build_seconds, query_seconds, memory_delta


def exact_subset(
    x: np.ndarray,
    train_n: int,
    query_idx: np.ndarray,
    workers: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    before = rss_mb()
    start = time.perf_counter()
    model = NearestNeighbors(n_neighbors=min(k + 1, train_n), algorithm="brute", metric="euclidean", n_jobs=workers)
    model.fit(x[:train_n])
    fit_seconds = time.perf_counter() - start
    query_start = time.perf_counter()
    distances, labels = model.kneighbors(x[query_idx], return_distance=True)
    query_seconds = time.perf_counter() - query_start
    out_distance = np.zeros(len(query_idx), dtype=np.float32)
    out_neighbors = np.full((len(query_idx), k), -1, dtype=np.int64)
    for i, global_idx in enumerate(query_idx):
        d, lab = distances[i], labels[i]
        if global_idx < train_n:
            keep = lab != global_idx
            d, lab = d[keep], lab[keep]
        d, lab = d[:k], lab[:k]
        out_distance[i] = float(d[0]) if len(d) else 0.0
        out_neighbors[i, : len(lab)] = lab
    return out_distance, out_neighbors, fit_seconds, query_seconds, max(rss_mb() - before, 0.0)


def benchmark_single(args: argparse.Namespace) -> dict:
    n, seed = args.single_n, args.seed
    process_start = rss_mb()
    generate_start = time.perf_counter()
    template_root = Path(args.template_root) if args.template_root else None
    x, domain, year, workload = make_workload(n, seed, template_root)
    train_n, val_n = int(n * 0.70), int(n * 0.15)
    val_slice = slice(train_n, train_n + val_n)
    test_slice = slice(train_n + val_n, n)
    generation_seconds = time.perf_counter() - generate_start
    hnsw_distance, index, hnsw_build, hnsw_query, hnsw_memory = hnsw_all(x, train_n, args.workers, args.k, seed)

    feature_start = time.perf_counter()
    train_scale = max(float(np.quantile(hnsw_distance[:train_n], 0.90)), 1e-6)
    a1 = np.clip(hnsw_distance / train_scale, 0, 4)
    domain_values, domain_counts = np.unique(domain[:train_n], return_counts=True)
    count_map = dict(zip(domain_values.tolist(), domain_counts.tolist()))
    a2 = np.asarray([1.0 / np.sqrt(1.0 + count_map.get(int(value), 0)) for value in domain], dtype=np.float32)
    rng = np.random.default_rng(seed + 13)
    a3 = rng.beta(1.5, 6.0, size=n).astype(np.float32)
    center = x[:train_n].mean(axis=0)
    scale = np.maximum(x[:train_n].std(axis=0), 1e-4)
    frontier = np.sqrt(np.mean(((x - center) / scale) ** 2, axis=1))
    future = np.maximum(year.astype(float) - float(np.max(year[:train_n])), 0.0) / max(float(np.ptp(year[:train_n])), 1.0)
    a4 = np.maximum(frontier, future).astype(np.float32)
    a5 = np.clip(0.35 * a1 + rng.beta(2.0, 5.0, size=n), 0, 3).astype(np.float32)
    axes = np.column_stack([a1, a2, a3, a4, a5]).astype(np.float32)
    lo, hi = np.quantile(axes[:train_n], [0.01, 0.99], axis=0)
    hi[np.isclose(lo, hi)] = lo[np.isclose(lo, hi)] + 1.0
    axes = np.clip((axes - lo) / (hi - lo), 0, 1)
    p_failure = sigmoid(-4.0 + axes @ np.asarray([1.7, 1.3, 1.2, 1.6, 1.8]) + 0.8 * axes[:, 0] * axes[:, 3])
    failure = (rng.random(n) < p_failure).astype(int)
    axis_feature_seconds = time.perf_counter() - feature_start

    fit_start = time.perf_counter()
    weights, intercept = fit_nonnegative(axes[val_slice], failure[val_slice])
    mara_fit_seconds = time.perf_counter() - fit_start
    inference_start = time.perf_counter()
    test_risk = sigmoid(axes[test_slice] @ weights + intercept)
    inference_seconds = time.perf_counter() - inference_start
    test_auroc = float(roc_auc_score(failure[test_slice], test_risk))

    query_n = min(args.exact_query_count, n)
    query_idx = np.linspace(0, n - 1, query_n, dtype=np.int64)
    exact_distance, exact_neighbors, exact_fit, exact_query, exact_memory = exact_subset(x, train_n, query_idx, args.workers, args.k)
    approx_labels, approx_distances = index.knn_query(x[query_idx], k=min(args.k + 1, train_n), num_threads=args.workers)
    approx_distance = np.zeros(query_n, dtype=float)
    recall = []
    for i, global_idx in enumerate(query_idx):
        lab, dist = approx_labels[i], approx_distances[i]
        if global_idx < train_n:
            keep = lab != global_idx
            lab, dist = lab[keep], dist[keep]
        lab, dist = lab[: args.k], dist[: args.k]
        approx_distance[i] = math.sqrt(max(float(dist[0]), 0.0)) if len(dist) else 0.0
        exact_set = set(exact_neighbors[i][exact_neighbors[i] >= 0].tolist())
        recall.append(len(exact_set.intersection(lab.tolist())) / max(len(exact_set), 1))
    y_query = failure[query_idx]
    exact_support_auroc = float(roc_auc_score(y_query, exact_distance))
    approx_support_auroc = float(roc_auc_score(y_query, approx_distance))
    peak = max(rss_mb(), process_start + hnsw_memory, process_start + exact_memory)
    return {
        "workload": workload,
        "n": n,
        "dimension": int(x.shape[1]),
        "seed": seed,
        "workers": args.workers,
        "train_n": train_n,
        "validation_n": val_n,
        "test_n": n - train_n - val_n,
        "generation_seconds": generation_seconds,
        "hnsw_build_seconds": hnsw_build,
        "hnsw_query_all_seconds": hnsw_query,
        "axis_feature_seconds_excluding_neighbors": axis_feature_seconds,
        "axis_construction_seconds": hnsw_build + hnsw_query + axis_feature_seconds,
        "mara_fit_seconds": mara_fit_seconds,
        "mara_inference_seconds": inference_seconds,
        "mara_inference_microseconds_per_sample": inference_seconds * 1e6 / max(n - train_n - val_n, 1),
        "peak_rss_mb": peak,
        "gpu_memory_mb": 0.0,
        "test_failure_auroc_mara": test_auroc,
        "exact_query_count": query_n,
        "exact_fit_seconds": exact_fit,
        "exact_query_seconds": exact_query,
        "exact_query_microseconds_per_sample": exact_query * 1e6 / query_n,
        "hnsw_query_microseconds_per_sample": hnsw_query * 1e6 / n,
        "hnsw_recall_at_k": float(np.mean(recall)),
        "exact_support_auroc_subset": exact_support_auroc,
        "hnsw_support_auroc_subset": approx_support_auroc,
        "support_auroc_loss_exact_minus_hnsw": exact_support_auroc - approx_support_auroc,
        "exact_scope": "reference index fit on 70% of n; exact queries evaluated on a fixed subset",
        "approximate_scope": "HNSW index fit on 70% of n; all n records queried",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MARA exact/HNSW scalability benchmark")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10000, 50000, 100000, 250000, 500000, 1000000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeats-at-1m", type=int, default=3)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--exact-query-count", type=int, default=10000)
    parser.add_argument("--template-root")
    parser.add_argument("--single-n", type=int)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, 64))
    out = ensure_dir(args.out)
    if args.single_n:
        result = benchmark_single(args)
        path = out / f"scalability_n{args.single_n}_seed{args.seed}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        return

    rows = []
    for n in args.sizes:
        repeats = args.repeats_at_1m if n >= 1000000 else args.repeats
        for repeat in range(repeats):
            seed = 20260831 + repeat
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--out",
                str(out),
                "--single-n",
                str(n),
                "--seed",
                str(seed),
                "--workers",
                str(args.workers),
                "--k",
                str(args.k),
                "--exact-query-count",
                str(args.exact_query_count),
            ]
            if args.template_root:
                command.extend(["--template-root", args.template_root])
            print(f"[RUN] n={n} repeat={repeat + 1}/{repeats}", flush=True)
            subprocess.run(command, check=True)
            result_path = out / f"scalability_n{n}_seed{seed}.json"
            rows.append(json.loads(result_path.read_text()))
    long = pd.DataFrame(rows)
    long.to_csv(out / "scalability_long.csv", index=False)
    metric_cols = [c for c in long.columns if c.endswith("seconds") or c.endswith("mb") or "microseconds" in c or "auroc" in c or c == "hnsw_recall_at_k"]
    summary = long.groupby(["workload", "n", "dimension"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary.columns]
    summary.to_csv(out / "scalability_mean_std.csv", index=False)
    try:
        summary.to_markdown(out / "table_scalability_mean_std.md", index=False)
    except Exception:
        pass
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "sizes": args.sizes,
                "repeats_below_1m": args.repeats,
                "repeats_at_1m": args.repeats_at_1m,
                "workers": args.workers,
                "k": args.k,
                "exact_query_count": args.exact_query_count,
                "template_root": args.template_root,
                "exact_vs_approx_policy": "Exact kNN is measured on the same fixed query subset at every n; HNSW indexes 70% and queries all n.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote scalability benchmark to {out}")


if __name__ == "__main__":
    main()
