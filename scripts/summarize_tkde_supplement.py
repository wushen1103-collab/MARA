#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MARA_METHODS = {"mara_nonnegative", "mara_rank_fusion"}
PRIMARY_BASELINES = [
    "max_softmax_risk",
    "predictive_entropy",
    "ensemble_disagreement",
    "uncertainty_composite",
    "representation_support",
    "distribution_frontier",
    "scalar_axis_mean",
    "validation_knn_error",
    "rf_error_predictor",
]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_crossdomain(specs: list[str]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        condition, path = spec.split("=", 1)
        frame = pd.read_csv(Path(path) / "metrics_long.csv")
        frame.insert(2, "condition", condition)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_protocol_attribution(specs: list[str]) -> pd.DataFrame:
    expected = {
        "state_shift": ["A2_source_domain"],
        "temporal_shift": ["A4_frontier"],
        "state_time_compound": ["A2_source_domain", "A4_frontier"],
        "official_temporal": ["A4_frontier"],
        "structural_group_ood": ["A2_source_domain"],
    }
    rows = []
    for spec in specs:
        condition, path = spec.split("=", 1)
        for prediction_path in Path(path).glob("*/*/seed*/test_predictions.csv.gz"):
            seed_dir = prediction_path.parent
            protocol = seed_dir.parent.name
            dataset = seed_dir.parent.parent.name
            true_axes = expected.get(protocol)
            if not true_axes:
                continue
            frame = pd.read_csv(prediction_path)
            contrib_cols = [column for column in frame.columns if column.startswith("contrib_A")]
            mean_contrib = frame[contrib_cols].mean()
            axis_score = {column.replace("contrib_", ""): float(value) for column, value in mean_contrib.items()}
            ranked = sorted(axis_score, key=axis_score.get, reverse=True)
            predicted = set(ranked[: len(true_axes)])
            truth = set(true_axes)
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "protocol": protocol,
                    "seed": int(seed_dir.name.replace("seed", "")),
                    "protocol_label_type": "known dataset-level shift construction; not prediction-level causal ground truth",
                    "true_axes": "+".join(true_axes),
                    "predicted_topk_axes": "+".join(ranked[: len(true_axes)]),
                    "precision_at_k": len(predicted & truth) / len(predicted),
                    "recall_at_k": len(predicted & truth) / len(truth),
                    "exact_match": float(predicted == truth),
                    "true_axis_contribution_mass": float(sum(axis_score[axis] for axis in true_axes)),
                    "top1_axis": ranked[0],
                    **{f"mean_contrib_{axis}": axis_score.get(axis, 0.0) for axis in sorted(axis_score)},
                }
            )
    return pd.DataFrame(rows)


def mean_std_text(values: pd.Series, digits: int = 4) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if not len(values):
        return "NA"
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.{digits}f} &plusmn; {std:.{digits}f}"


def mean_std_values(values: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if not len(values):
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)


def crossdomain_summary(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = ["base_accuracy", "failure_auroc", "failure_auprc", "risk_brier", "risk_nll", "risk_ece", "aurc"]
    rows = []
    for keys, block in long.groupby(["dataset", "condition", "protocol", "method"], sort=True):
        dataset, condition, protocol, method = keys
        row = {"dataset": dataset, "condition": condition, "protocol": protocol, "method": method, "seeds": int(block["seed"].nunique())}
        for metric in metric_cols:
            mean, std = mean_std_values(block[metric])
            row[metric] = mean_std_text(block[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    summary = pd.DataFrame(rows)

    comparison_rows = []
    for keys, block in long.groupby(["dataset", "condition", "protocol"], sort=True):
        dataset, condition, protocol = keys
        means = block.groupby("method")["failure_auroc"].mean()
        baseline_candidates = means[means.index.isin(PRIMARY_BASELINES)]
        # The diagnostic head is fixed before test evaluation. The external
        # method is a descriptive upper envelope, so no post-selection
        # inferential p-value is reported for this comparison.
        mara = "mara_nonnegative"
        baseline = str(baseline_candidates.idxmax())
        pivot = block[block["method"].isin([mara, baseline])].pivot_table(index="seed", columns="method", values="failure_auroc", aggfunc="mean").dropna()
        delta = pivot[mara] - pivot[baseline]
        tolerance = 0.002
        mean_delta = float(delta.mean())
        verdict = "lead" if mean_delta > tolerance else "lag" if mean_delta < -tolerance else "tie"
        comparison_rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "protocol": protocol,
                "best_mara": mara,
                "best_external_baseline": baseline,
                "mara_selection": "fixed nonnegative diagnostic head",
                "external_selection": "descriptive test-set upper envelope across nine rerun baselines",
                "paired_seed_n": len(delta),
                "auroc_mara": mean_std_text(pivot[mara]),
                "auroc_external": mean_std_text(pivot[baseline]),
                "paired_delta_mean": mean_delta,
                "wilcoxon_p": float("nan"),
                "inference_status": "not computed because the external comparator is selected by test-set mean",
                "verdict_at_0.002": verdict,
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    ranks = long.groupby(["dataset", "condition", "protocol", "seed"], as_index=False).apply(
        lambda block: pd.Series(
            {
                "best_method": block.loc[block["failure_auroc"].idxmax(), "method"],
                "best_auroc": block["failure_auroc"].max(),
                "best_mara_auroc": block.loc[block["method"].isin(MARA_METHODS), "failure_auroc"].max(),
                "best_external_auroc": block.loc[block["method"].isin(PRIMARY_BASELINES), "failure_auroc"].max(),
            }
        ),
        include_groups=False,
    ).reset_index(drop=True)
    ranks["mara_wins_or_ties"] = ranks["best_mara_auroc"] >= ranks["best_external_auroc"] - 1e-6
    return summary, comparison, ranks


def selected_crossdomain_table(summary: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "max_softmax_risk",
        "predictive_entropy",
        "ensemble_disagreement",
        "representation_support",
        "distribution_frontier",
        "validation_knn_error",
        "rf_error_predictor",
        "mara_nonnegative",
        "mara_rank_fusion",
    ]
    return summary[summary["method"].isin(selected)].copy()


def add_compound_tables(compound_root: Path, out: Path) -> dict[str, pd.DataFrame]:
    names = [
        "compound_risk_mean_std",
        "compound_cause_recovery_mean_std",
        "compound_by_combo_mean_std",
        "missing_evidence_mean_std",
        "informative_missing_evidence_mean_std",
        "remediation_mean_std",
    ]
    tables = {}
    for name in names:
        tables[name] = pd.read_csv(compound_root / f"{name}.csv")
        tables[name].to_csv(out / f"{name}.csv", index=False)
    return tables


def add_scalability_table(scale_root: Path, out: Path) -> pd.DataFrame:
    frame = pd.read_csv(scale_root / "scalability_mean_std.csv")
    long = pd.read_csv(scale_root / "scalability_long.csv")
    long["support_auroc_abs_difference"] = long["support_auroc_loss_exact_minus_hnsw"].abs()
    absolute = (
        long.groupby("n")["support_auroc_abs_difference"]
        .agg(
            repeats="count",
            support_auroc_abs_difference_mean="mean",
            support_auroc_abs_difference_std="std",
            support_auroc_abs_difference_max="max",
        )
        .reset_index()
    )
    cols = [
        "n",
        "axis_construction_seconds_mean",
        "axis_construction_seconds_std",
        "mara_fit_seconds_mean",
        "mara_fit_seconds_std",
        "mara_inference_microseconds_per_sample_mean",
        "peak_rss_mb_mean",
        "hnsw_recall_at_k_mean",
        "support_auroc_loss_exact_minus_hnsw_mean",
    ]
    selected = frame[cols].copy()
    selected = selected.merge(absolute, on="n", how="left")
    selected.to_csv(out / "scalability_selected.csv", index=False)
    return selected


def write_markdown(path: Path, tables: dict[str, pd.DataFrame], audit: list[str]) -> None:
    chunks = [
        "# MARA TKDE Supplementary Experiment Tables",
        "",
        "All numerical results marked as rerun were produced with the public scripts in this repository. Cross-domain, compound-shift, missing-evidence, and remediation entries use five seeds. Scalability entries use three repeats at every workload size, including 1M.",
        "",
    ]
    for title, frame in tables.items():
        chunks.extend([f"## {title}", "", frame.to_markdown(index=False), ""])
    chunks.extend(["## Claim audit", "", *[f"- {item}" for item in audit], ""])
    path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross", nargs="+", required=True, help="condition=/path/to/result")
    parser.add_argument("--compound", required=True)
    parser.add_argument("--scalability", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = ensure_dir(args.out)
    cross = load_crossdomain(args.cross)
    protocol_attribution = load_protocol_attribution(args.cross)
    cross.to_csv(out / "crossdomain_metrics_long.csv", index=False)
    summary, comparison, ranks = crossdomain_summary(cross)
    summary.to_csv(out / "crossdomain_mean_std.csv", index=False)
    comparison.to_csv(out / "crossdomain_paired_comparison.csv", index=False)
    ranks.to_csv(out / "crossdomain_seed_winners.csv", index=False)
    protocol_attribution.to_csv(out / "protocol_shift_attribution_long.csv", index=False)
    protocol_attribution_summary = (
        protocol_attribution.groupby(["dataset", "condition", "protocol", "true_axes"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            precision_at_k_mean=("precision_at_k", "mean"),
            recall_at_k_mean=("recall_at_k", "mean"),
            exact_match_mean=("exact_match", "mean"),
            true_axis_contribution_mass_mean=("true_axis_contribution_mass", "mean"),
            true_axis_contribution_mass_std=("true_axis_contribution_mass", "std"),
        )
    )
    protocol_attribution_summary.to_csv(out / "protocol_shift_attribution_mean_std.csv", index=False)
    selected = selected_crossdomain_table(summary)
    selected.to_csv(out / "crossdomain_selected.csv", index=False)
    compound = add_compound_tables(Path(args.compound), out)
    scalability = add_scalability_table(Path(args.scalability), out)
    verdict_counts = comparison["verdict_at_0.002"].value_counts().to_dict()
    audit = [
        f"Cross-domain protocols: {cross[['dataset', 'condition', 'protocol']].drop_duplicates().shape[0]} across {cross['dataset'].nunique()} non-molecular datasets.",
        f"At an AUROC tolerance of 0.002, the fixed MARA nonnegative head leads the descriptive external envelope in {verdict_counts.get('lead', 0)}/10 protocols, ties in {verdict_counts.get('tie', 0)}/10, and lags in {verdict_counts.get('lag', 0)}/10.",
        "No p-value is attached to the external-envelope comparison because the displayed external method is selected by test-set mean within each protocol.",
        f"Cross-domain seed-level cases where the best MARA variant wins or ties the best external baseline: {int(ranks['mara_wins_or_ties'].sum())}/{len(ranks)}.",
        "ACSIncome is a Folktables task distributed through a documented Hugging Face mirror; OGBN-Arxiv uses the official OGB time split and a train-only structural-cluster OOD split.",
        "Real-domain cause labels are operational stress proxies, not causal ground truth; causal recovery claims are restricted to the controlled compound-intervention suite.",
        "The scalability workload is an empirical bootstrap of ACSIncome representations; exact-neighbor quality is measured on a fixed 500-query subset, while HNSW queries every record.",
    ]
    tables = {
        "Cross-domain paired comparison": comparison,
        "Cross-domain selected methods": selected,
        "Protocol-level shift attribution": protocol_attribution_summary,
        "Compound-shift failure risk": compound["compound_risk_mean_std"],
        "Compound cause recovery": compound["compound_cause_recovery_mean_std"],
        "Cause recovery by intervention combination": compound["compound_by_combo_mean_std"],
        "Missing evidence": compound["missing_evidence_mean_std"],
        "Informative missing evidence": compound["informative_missing_evidence_mean_std"],
        "Axis-guided remediation": compound["remediation_mean_std"],
        "Scalability": scalability,
    }
    write_markdown(out / "TKDE_supplement_tables.md", tables, audit)
    (out / "summary_manifest.json").write_text(
        json.dumps({"cross_specs": args.cross, "compound": args.compound, "scalability": args.scalability, "audit": audit}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote TKDE summary tables to {out}")


if __name__ == "__main__":
    main()
