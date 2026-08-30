#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    jaccard_score,
    log_loss,
    roc_auc_score,
)


AXES = ["A1_support", "A2_source_domain", "A3_supervision", "A4_frontier", "A5_model_conflict"]
COMBOS = [
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (0, 1),
    (0, 3),
    (1, 3),
    (0, 1, 3),
    (0, 2, 4),
    (1, 2, 3),
    (0, 1, 2, 3, 4),
]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def safe_metric(fn) -> float:
    try:
        return float(fn())
    except Exception:
        return float("nan")


def expected_calibration_error(y: np.ndarray, risk: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (risk >= lo) & (risk < hi if hi < 1.0 else risk <= hi)
        if np.any(mask):
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(risk[mask].mean()))
    return total


def aurc(y: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(risk, kind="stable")
    selective_risk = np.cumsum(y[order], dtype=float) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1, dtype=float) / len(y)
    return float(np.trapezoid(selective_risk, coverage))


def fit_nonnegative(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    y = y.astype(float)
    if len(np.unique(y)) < 2:
        return np.zeros(x.shape[1]), float(np.log((np.mean(y) + 1e-5) / (1 - np.mean(y) + 1e-5)))
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    sw = np.where(y > 0.5, len(y) / (2 * pos), len(y) / (2 * neg))

    def objective(theta: np.ndarray):
        w, b = theta[:-1], theta[-1]
        p = sigmoid(x @ w + b)
        loss = -np.mean(sw * (y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))) + 1e-3 * np.dot(w, w)
        residual = sw * (p - y) / len(y)
        grad = np.r_[x.T @ residual + 2e-3 * w, residual.sum()]
        return float(loss), grad

    init = np.zeros(x.shape[1] + 1)
    init[-1] = float(np.log((np.mean(y) + 1e-3) / (1 - np.mean(y) + 1e-3)))
    result = minimize(objective, init, jac=True, method="L-BFGS-B", bounds=[(0, None)] * x.shape[1] + [(None, None)])
    return result.x[:-1], float(result.x[-1])


def calibrate(raw_val: np.ndarray, y_val: np.ndarray, raw: np.ndarray) -> np.ndarray:
    if len(np.unique(y_val)) < 2 or float(np.std(raw_val)) <= 1e-12:
        return np.full(len(raw), float(np.mean(y_val)))
    model = LogisticRegression(max_iter=1000).fit(raw_val.reshape(-1, 1), y_val)
    return np.clip(model.predict_proba(raw.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def failure_probability(x: np.ndarray) -> np.ndarray:
    weights = np.asarray([1.65, 1.45, 1.35, 1.60, 1.75])
    interaction = 0.95 * x[:, 0] * x[:, 1] + 0.85 * x[:, 0] * x[:, 3] + 0.90 * x[:, 2] * x[:, 4]
    return sigmoid(-4.4 + x @ weights + interaction)


def generate_block(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.beta(1.8, 7.5, size=(n, len(AXES)))
    cause = np.zeros((n, len(AXES)), dtype=int)
    combo_id = rng.integers(0, len(COMBOS), size=n)
    no_shift = rng.random(n) < 0.18
    intensity = rng.uniform(0.50, 1.00, size=n)
    labels = np.full(n, "none", dtype="<U160")
    for i, combo_idx in enumerate(combo_id):
        if no_shift[i]:
            continue
        combo = COMBOS[int(combo_idx)]
        cause[i, list(combo)] = 1
        x[i, list(combo)] = np.maximum(x[i, list(combo)], np.clip(intensity[i] + rng.normal(0, 0.035, len(combo)), 0, 1))
        labels[i] = "+".join(AXES[j] for j in combo)
    prob = failure_probability(x)
    failure = (rng.random(n) < prob).astype(int)
    return x, cause, failure, labels


def fit_axis_thresholds(score: np.ndarray, truth: np.ndarray) -> np.ndarray:
    thresholds = np.zeros(score.shape[1], dtype=float)
    grid = np.linspace(0.03, 0.65, 80)
    for j in range(score.shape[1]):
        values = [f1_score(truth[:, j], score[:, j] >= threshold, zero_division=0) for threshold in grid]
        thresholds[j] = grid[int(np.argmax(values))]
    return thresholds


def cause_metrics(truth: np.ndarray, score: np.ndarray, threshold: np.ndarray) -> dict:
    valid = truth.sum(axis=1) > 0
    truth = truth[valid]
    score = score[valid]
    prediction = (score >= threshold[None, :]).astype(int)
    p_at_k, r_at_k = [], []
    for row_truth, row_score in zip(truth, score):
        k = int(row_truth.sum())
        top = np.argsort(-row_score)[:k]
        hit = int(row_truth[top].sum())
        p_at_k.append(hit / max(k, 1))
        r_at_k.append(hit / max(int(row_truth.sum()), 1))
    return {
        "macro_f1": safe_metric(lambda: f1_score(truth, prediction, average="macro", zero_division=0)),
        "micro_f1": safe_metric(lambda: f1_score(truth, prediction, average="micro", zero_division=0)),
        "jaccard": safe_metric(lambda: jaccard_score(truth, prediction, average="samples", zero_division=0)),
        "exact_match": float(np.mean(np.all(truth == prediction, axis=1))),
        "precision_at_k": float(np.mean(p_at_k)),
        "recall_at_k": float(np.mean(r_at_k)),
    }


def risk_metrics(y: np.ndarray, risk: np.ndarray) -> dict:
    return {
        "failure_rate": float(y.mean()),
        "failure_auroc": safe_metric(lambda: roc_auc_score(y, risk)),
        "failure_auprc": safe_metric(lambda: average_precision_score(y, risk)),
        "risk_brier": safe_metric(lambda: brier_score_loss(y, risk)),
        "risk_nll": safe_metric(lambda: log_loss(y, risk, labels=[0, 1])),
        "risk_ece": expected_calibration_error(y, risk),
        "aurc": aurc(y, risk),
    }


def normalized_rf_occlusion(model: RandomForestClassifier, x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    base = model.predict_proba(x)[:, 1]
    score = np.zeros_like(x, dtype=float)
    for axis in range(x.shape[1]):
        occluded = x.copy()
        occluded[:, axis] = reference[axis]
        score[:, axis] = np.maximum(base - model.predict_proba(occluded)[:, 1], 0.0)
    return score / np.maximum(score.sum(axis=1, keepdims=True), 1e-12)


def compound_experiment(seed: int, n_train: int, n_validation: int, n_test: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    x_train, _, _, _ = generate_block(n_train, seed + 1)
    x_val, cause_val, y_val, _ = generate_block(n_validation, seed + 2)
    x_test, cause_test, y_test, combo_test = generate_block(n_test, seed + 3)
    lo = np.quantile(x_train, 0.01, axis=0)
    hi = np.quantile(x_train, 0.99, axis=0)
    hi[np.isclose(lo, hi)] = lo[np.isclose(lo, hi)] + 1.0
    val = np.clip((x_val - lo) / (hi - lo), 0, 1)
    test = np.clip((x_test - lo) / (hi - lo), 0, 1)
    w_mara, b_mara = fit_nonnegative(val, y_val)
    raw_mara_val = sigmoid(val @ w_mara + b_mara)
    raw_mara_test = sigmoid(test @ w_mara + b_mara)
    risk_mara = calibrate(raw_mara_val, y_val, raw_mara_test)
    scalar_val, scalar_test = val.mean(axis=1), test.mean(axis=1)
    risk_scalar = calibrate(scalar_val, y_val, scalar_test)
    max_val, max_test = val.max(axis=1), test.max(axis=1)
    risk_max = calibrate(max_val, y_val, max_test)
    rf = RandomForestClassifier(n_estimators=350, min_samples_leaf=3, n_jobs=16, random_state=seed, class_weight="balanced")
    rf.fit(val, y_val)
    rf_val, rf_test = rf.predict_proba(val)[:, 1], rf.predict_proba(test)[:, 1]
    rf_reference = np.median(val, axis=0)
    rf_occlusion_val = normalized_rf_occlusion(rf, val, rf_reference)
    rf_occlusion_test = normalized_rf_occlusion(rf, test, rf_reference)
    risk_rf = calibrate(rf_val, y_val, rf_test)
    rank_val = np.mean(np.column_stack([pd.Series(v).rank(pct=True).to_numpy() for v in [raw_mara_val, scalar_val, max_val, rf_val]]), axis=1)
    rank_test = np.mean(np.column_stack([pd.Series(v).rank(pct=True).to_numpy() for v in [raw_mara_test, scalar_test, max_test, rf_test]]), axis=1)
    risk_fusion = calibrate(rank_val, y_val, rank_test)
    risk_rows = []
    for method, risk in {
        "scalar_axis_mean": risk_scalar,
        "max_axis": risk_max,
        "rf_error_predictor": risk_rf,
        "mara_nonnegative": risk_mara,
        "mara_rank_fusion": risk_fusion,
    }.items():
        risk_rows.append({"seed": seed, "method": method, **risk_metrics(y_test, risk)})

    learned_val = np.maximum(val * w_mara[None, :], 0)
    learned_test = np.maximum(test * w_mara[None, :], 0)
    attribution_anchor_alpha = 0.90
    mara_contrib_val = (1.0 - attribution_anchor_alpha) * learned_val + attribution_anchor_alpha * val
    mara_contrib_test = (1.0 - attribution_anchor_alpha) * learned_test + attribution_anchor_alpha * test
    mara_contrib_val /= np.maximum(mara_contrib_val.sum(axis=1, keepdims=True), 1e-12)
    mara_contrib_test /= np.maximum(mara_contrib_test.sum(axis=1, keepdims=True), 1e-12)
    equal_val = val / np.maximum(val.sum(axis=1, keepdims=True), 1e-12)
    equal_test = test / np.maximum(test.sum(axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed + 99)
    random_val = rng.random(val.shape)
    random_test = rng.random(test.shape)
    random_val /= random_val.sum(axis=1, keepdims=True)
    random_test /= random_test.sum(axis=1, keepdims=True)
    supervised_val = np.zeros_like(val)
    supervised_test = np.zeros_like(test)
    for axis in range(len(AXES)):
        cause_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed + axis)
        cause_model.fit(val, cause_val[:, axis])
        supervised_val[:, axis] = cause_model.predict_proba(val)[:, 1]
        supervised_test[:, axis] = cause_model.predict_proba(test)[:, 1]
    cause_rows = []
    method_scores = {
        "mara_attribution": (mara_contrib_val, mara_contrib_test),
        "axis_severity": (equal_val, equal_test),
        "rf_occlusion_attribution": (rf_occlusion_val, rf_occlusion_test),
        "supervised_cause_upper_bound": (supervised_val, supervised_test),
        "random_attribution": (random_val, random_test),
    }
    thresholds = {}
    for method, (score_val, score_test) in method_scores.items():
        threshold = fit_axis_thresholds(score_val, cause_val)
        thresholds[method] = threshold.tolist()
        cause_rows.append({"seed": seed, "method": method, "combo": "ALL", "n": len(cause_test), **cause_metrics(cause_test, score_test, threshold)})
        for combo in sorted(np.unique(combo_test)):
            mask = combo_test == combo
            if combo == "none" or int(mask.sum()) < 20:
                continue
            cause_rows.append({"seed": seed, "method": method, "combo": combo, "n": int(mask.sum()), **cause_metrics(cause_test[mask], score_test[mask], threshold)})
    state = {
        "x_val": val,
        "cause_val": cause_val,
        "y_val": y_val,
        "x_test": test,
        "cause_test": cause_test,
        "y_test": y_test,
        "w_mara": w_mara,
        "b_mara": b_mara,
        "mara_contrib_test": mara_contrib_test,
        "rf_occlusion_test": rf_occlusion_test,
        "thresholds": thresholds,
    }
    return pd.DataFrame(risk_rows), pd.DataFrame(cause_rows), state


def apply_random_mask(x: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    missing = rng.random(x.shape) < rate
    all_missing = missing.all(axis=1)
    if np.any(all_missing):
        keep = rng.integers(0, x.shape[1], size=int(all_missing.sum()))
        missing[np.flatnonzero(all_missing), keep] = False
    return np.where(missing, 0.0, x), missing


def calibrated_mask_probability(score: np.ndarray, rate: float, slope: float = 2.0) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    score = (score - score.mean()) / max(float(score.std()), 1e-8)
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if float(sigmoid(mid + slope * score).mean()) < rate:
            lo = mid
        else:
            hi = mid
    return sigmoid((lo + hi) / 2.0 + slope * score)


def apply_informative_mask(x: np.ndarray, rate: float, seed: int, mechanism: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    if mechanism == "MAR":
        probability = np.column_stack(
            [calibrated_mask_probability(x[:, (axis + 1) % d], rate) for axis in range(d)]
        )
        missing = rng.random(x.shape) < probability
    elif mechanism == "MNAR":
        probability = np.column_stack([calibrated_mask_probability(x[:, axis], rate) for axis in range(d)])
        missing = rng.random(x.shape) < probability
    elif mechanism == "axis_dependent":
        axis_weight = np.asarray([1.60, 0.60, 1.25, 0.75, 0.80])
        probability = np.clip(rate * axis_weight / axis_weight.mean(), 0.0, 0.98)
        missing = rng.random(x.shape) < probability[None, :]
    elif mechanism == "adversarial":
        missing = np.zeros_like(x, dtype=bool)
        expected = rate * d
        fixed = int(np.floor(expected))
        extra = expected - fixed
        order = np.argsort(-x, axis=1)
        if fixed:
            missing[np.arange(n)[:, None], order[:, :fixed]] = True
        if extra > 0 and fixed < d:
            rows = np.flatnonzero(rng.random(n) < extra)
            missing[rows, order[rows, fixed]] = True
    else:
        raise ValueError(f"Unknown informative missingness mechanism: {mechanism}")
    all_missing = missing.all(axis=1)
    if np.any(all_missing):
        keep = np.argmin(x[all_missing], axis=1)
        missing[np.flatnonzero(all_missing), keep] = False
    return np.where(missing, 0.0, x), missing


def missing_evidence_experiment(seed: int, state: dict, rates: list[float], mechanism: str = "MCAR") -> pd.DataFrame:
    val, y_val = state["x_val"], state["y_val"]
    test, y_test = state["x_test"], state["y_test"]
    cause_test = state["cause_test"]
    full_w, full_b = state["w_mara"], state["b_mara"]
    rows = []
    for rate_idx, rate in enumerate(rates):
        mask_fn = apply_random_mask if mechanism == "MCAR" else lambda values, fraction, random_seed: apply_informative_mask(
            values, fraction, random_seed, mechanism
        )
        val_masked, missing_val = mask_fn(val, rate, seed + 1000 + rate_idx)
        test_masked, missing_test = mask_fn(test, rate, seed + 2000 + rate_idx)
        strategies = {}
        raw_val = sigmoid(val_masked @ full_w + full_b)
        raw_test = sigmoid(test_masked @ full_w + full_b)
        strategies["zero_imputation"] = (raw_val, raw_test, np.maximum(test_masked * full_w[None, :], 0))

        indicator_val = np.hstack([val_masked, missing_val.astype(float)])
        indicator_test = np.hstack([test_masked, missing_test.astype(float)])
        indicator_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(indicator_val, y_val)
        strategies["missing_indicator"] = (
            indicator_model.predict_proba(indicator_val)[:, 1],
            indicator_model.predict_proba(indicator_test)[:, 1],
            np.maximum(test_masked * indicator_model.coef_[0, :5][None, :], 0),
        )

        aug_x, aug_y = [], []
        for rep, dropout_rate in enumerate([0.1, 0.3, 0.5, 0.7]):
            masked, _ = apply_random_mask(val, dropout_rate, seed + 3000 + rate_idx * 10 + rep)
            aug_x.append(masked)
            aug_y.append(y_val)
        dropout_w, dropout_b = fit_nonnegative(np.vstack(aug_x), np.concatenate(aug_y))
        strategies["axis_dropout_training"] = (
            sigmoid(val_masked @ dropout_w + dropout_b),
            sigmoid(test_masked @ dropout_w + dropout_b),
            np.maximum(test_masked * dropout_w[None, :], 0),
        )

        observed_val = np.maximum((~missing_val).sum(axis=1), 1)
        observed_test = np.maximum((~missing_test).sum(axis=1), 1)
        modular_val_logit = full_b + (val_masked @ full_w) * (len(AXES) / observed_val)
        modular_test_logit = full_b + (test_masked @ full_w) * (len(AXES) / observed_test)
        strategies["modular_inference"] = (
            sigmoid(modular_val_logit),
            sigmoid(modular_test_logit),
            np.maximum(test_masked * full_w[None, :] * (len(AXES) / observed_test[:, None]), 0),
        )

        for strategy, (raw_val, raw_test, contrib) in strategies.items():
            risk = calibrate(raw_val, y_val, raw_test)
            normalized = contrib / np.maximum(contrib.sum(axis=1, keepdims=True), 1e-12)
            observable_truth = cause_test * (~missing_test)
            threshold = np.full(len(AXES), 0.20)
            attr_observable = cause_metrics(observable_truth, normalized, threshold)
            attr_all = cause_metrics(cause_test, normalized, threshold)
            rows.append(
                {
                    "seed": seed,
                    "missingness_mechanism": mechanism,
                    "mask_rate": rate,
                    "strategy": strategy,
                    **risk_metrics(y_test, risk),
                    "attribution_macro_f1_all_causes": attr_all["macro_f1"],
                    "attribution_micro_f1_all_causes": attr_all["micro_f1"],
                    "attribution_macro_f1_observable_causes": attr_observable["macro_f1"],
                    "attribution_micro_f1_observable_causes": attr_observable["micro_f1"],
                }
            )
    return pd.DataFrame(rows)


def remediation_experiment(seed: int, state: dict, budget: int = 2, strength: float = 0.45) -> pd.DataFrame:
    shifted = state["cause_test"].sum(axis=1) > 0
    x = state["x_test"][shifted]
    causes = state["cause_test"][shifted]
    mara_score = state["mara_contrib_test"][shifted]
    global_order = np.argsort(-state["w_mara"])
    actions = {}
    actions["no_remediation"] = np.zeros_like(x)
    actions["generic_remediation"] = np.full_like(x, budget * strength / len(AXES))
    oracle_cause = np.zeros_like(x)
    mara = np.zeros_like(x)
    raw_severity = np.zeros_like(x)
    rf_occlusion = np.zeros_like(x)
    rf_score = state["rf_occlusion_test"][shifted]
    generic_top = np.zeros_like(x)
    generic_top[:, global_order[:budget]] = strength
    for i in range(len(x)):
        true_idx = np.flatnonzero(causes[i])
        if len(true_idx):
            selected = true_idx[np.argsort(-x[i, true_idx])[:budget]]
            oracle_cause[i, selected] = budget * strength / len(selected)
        mara[i, np.argsort(-mara_score[i])[:budget]] = strength
        raw_severity[i, np.argsort(-x[i])[:budget]] = strength
        rf_occlusion[i, np.argsort(-rf_score[i])[:budget]] = strength
    actions["global_top_axis"] = generic_top
    actions["mara_guided"] = mara
    actions["raw_severity_guided"] = raw_severity
    actions["rf_occlusion_guided"] = rf_occlusion
    actions["oracle_cause_guided"] = oracle_cause
    candidate_actions = []
    for selected in itertools.combinations(range(len(AXES)), budget):
        action = np.zeros_like(x)
        action[:, list(selected)] = strength
        candidate_actions.append(action)
    candidate_prob = np.column_stack([failure_probability(np.clip(x - action, 0, 1)) for action in candidate_actions])
    best = np.argmin(candidate_prob, axis=1)
    oracle_counterfactual = np.zeros_like(x)
    for candidate_id, action in enumerate(candidate_actions):
        mask = best == candidate_id
        oracle_counterfactual[mask] = action[mask]
    actions["oracle_counterfactual"] = oracle_counterfactual
    base_prob = failure_probability(x)
    rows = []
    for method, action in actions.items():
        residual_x = np.clip(x - action, 0, 1)
        residual_prob = failure_probability(residual_x)
        selected = action > 0
        action_precision = float((selected * causes).sum() / max(selected.sum(), 1))
        rows.append(
            {
                "seed": seed,
                "method": method,
                "action_budget": budget,
                "action_strength": strength,
                "expected_failure_before": float(base_prob.mean()),
                "expected_failure_after": float(residual_prob.mean()),
                "absolute_failure_reduction": float(np.mean(base_prob - residual_prob)),
                "relative_failure_reduction": float(np.mean(base_prob - residual_prob) / max(float(base_prob.mean()), 1e-9)),
                "mean_axes_acted": float(selected.sum(axis=1).mean()),
                "action_cause_precision": action_precision,
            }
        )
    return pd.DataFrame(rows)


def mean_std(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    out = frame.groupby(groups, as_index=False)[metrics].agg(["mean", "std"])
    out.columns = ["_".join([str(x) for x in col if str(x)]) for col in out.columns]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compound-cause, missing-evidence, and remediation experiments")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260831, 20260832, 20260833, 20260834, 20260835])
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-validation", type=int, default=25000)
    parser.add_argument("--n-test", type=int, default=50000)
    parser.add_argument("--mask-rates", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7])
    args = parser.parse_args()
    out = ensure_dir(args.out)
    risk_frames, cause_frames, missing_frames, informative_missing_frames, remediation_frames = [], [], [], [], []
    manifests = []
    for seed in args.seeds:
        print(f"[RUN] compound seed={seed}", flush=True)
        risk, cause, state = compound_experiment(seed, args.n_train, args.n_validation, args.n_test)
        missing = missing_evidence_experiment(seed, state, args.mask_rates)
        informative_missing = pd.concat(
            [
                missing_evidence_experiment(seed, state, args.mask_rates, mechanism)
                for mechanism in ["MAR", "MNAR", "axis_dependent", "adversarial"]
            ],
            ignore_index=True,
        )
        remediation = remediation_experiment(seed, state)
        risk_frames.append(risk)
        cause_frames.append(cause)
        missing_frames.append(missing)
        informative_missing_frames.append(informative_missing)
        remediation_frames.append(remediation)
        manifests.append({"seed": seed, "mara_weights": dict(zip(AXES, state["w_mara"].tolist())), "mara_intercept": state["b_mara"], "thresholds": state["thresholds"]})
        print(f"[DONE] compound seed={seed}", flush=True)
    risk = pd.concat(risk_frames, ignore_index=True)
    cause = pd.concat(cause_frames, ignore_index=True)
    missing = pd.concat(missing_frames, ignore_index=True)
    informative_missing = pd.concat(informative_missing_frames, ignore_index=True)
    remediation = pd.concat(remediation_frames, ignore_index=True)
    risk.to_csv(out / "compound_risk_long.csv", index=False)
    cause.to_csv(out / "compound_cause_recovery_long.csv", index=False)
    missing.to_csv(out / "missing_evidence_long.csv", index=False)
    informative_missing.to_csv(out / "informative_missing_evidence_long.csv", index=False)
    remediation.to_csv(out / "remediation_long.csv", index=False)
    risk_metrics_cols = ["failure_auroc", "failure_auprc", "risk_brier", "risk_nll", "risk_ece", "aurc"]
    cause_metrics_cols = ["macro_f1", "micro_f1", "jaccard", "exact_match", "precision_at_k", "recall_at_k"]
    missing_metrics_cols = risk_metrics_cols + ["attribution_macro_f1_all_causes", "attribution_micro_f1_all_causes", "attribution_macro_f1_observable_causes", "attribution_micro_f1_observable_causes"]
    remediation_metrics_cols = ["expected_failure_after", "absolute_failure_reduction", "relative_failure_reduction", "mean_axes_acted", "action_cause_precision"]
    summaries = {
        "compound_risk_mean_std": mean_std(risk, ["method"], risk_metrics_cols),
        "compound_cause_recovery_mean_std": mean_std(cause[cause["combo"] == "ALL"], ["method"], cause_metrics_cols),
        "compound_by_combo_mean_std": mean_std(cause[cause["combo"] != "ALL"], ["method", "combo"], cause_metrics_cols),
        "missing_evidence_mean_std": mean_std(missing, ["mask_rate", "strategy"], missing_metrics_cols),
        "informative_missing_evidence_mean_std": mean_std(
            informative_missing, ["missingness_mechanism", "mask_rate", "strategy"], missing_metrics_cols
        ),
        "remediation_mean_std": mean_std(remediation, ["method"], remediation_metrics_cols),
    }
    for name, frame in summaries.items():
        frame.to_csv(out / f"{name}.csv", index=False)
        try:
            frame.to_markdown(out / f"table_{name}.md", index=False)
        except Exception:
            pass
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "axes": AXES,
                "compound_interventions": [[AXES[i] for i in combo] for combo in COMBOS],
                "seeds": args.seeds,
                "n_train": args.n_train,
                "n_validation": args.n_validation,
                "n_test": args.n_test,
                "mask_rates": args.mask_rates,
                "missingness_mechanisms": ["MCAR", "MAR", "MNAR", "axis_dependent", "adversarial"],
                "attribution_anchor_alpha": 0.90,
                "remediation_semantics": "Controlled axis score reduction with equal action budget; expected failure is recomputed from the known data-generating mechanism.",
                "models": manifests,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote TKDE compound suite to {out}")


if __name__ == "__main__":
    main()
