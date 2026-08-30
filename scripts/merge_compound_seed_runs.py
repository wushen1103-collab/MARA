#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_tkde_compound_suite import AXES, COMBOS, mean_std


LONG_FILES = [
    "compound_risk_long.csv",
    "compound_cause_recovery_long.csv",
    "missing_evidence_long.csv",
    "informative_missing_evidence_long.csv",
    "remediation_long.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently run compound-suite seeds")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    roots = [Path(path) for path in args.inputs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    merged: dict[str, pd.DataFrame] = {}
    for name in LONG_FILES:
        frame = pd.concat([pd.read_csv(root / name) for root in roots], ignore_index=True)
        if frame.duplicated().any():
            raise ValueError(f"Duplicate rows detected while merging {name}")
        frame.to_csv(out / name, index=False)
        merged[name] = frame

    risk = merged["compound_risk_long.csv"]
    cause = merged["compound_cause_recovery_long.csv"]
    missing = merged["missing_evidence_long.csv"]
    informative = merged["informative_missing_evidence_long.csv"]
    remediation = merged["remediation_long.csv"]
    risk_metrics = ["failure_auroc", "failure_auprc", "risk_brier", "risk_nll", "risk_ece", "aurc"]
    cause_metrics = ["macro_f1", "micro_f1", "jaccard", "exact_match", "precision_at_k", "recall_at_k"]
    missing_metrics = risk_metrics + [
        "attribution_macro_f1_all_causes",
        "attribution_micro_f1_all_causes",
        "attribution_macro_f1_observable_causes",
        "attribution_micro_f1_observable_causes",
    ]
    remediation_metrics = [
        "expected_failure_after",
        "absolute_failure_reduction",
        "relative_failure_reduction",
        "mean_axes_acted",
        "action_cause_precision",
    ]
    summaries = {
        "compound_risk_mean_std": mean_std(risk, ["method"], risk_metrics),
        "compound_cause_recovery_mean_std": mean_std(cause[cause["combo"] == "ALL"], ["method"], cause_metrics),
        "compound_by_combo_mean_std": mean_std(cause[cause["combo"] != "ALL"], ["method", "combo"], cause_metrics),
        "missing_evidence_mean_std": mean_std(missing, ["mask_rate", "strategy"], missing_metrics),
        "informative_missing_evidence_mean_std": mean_std(
            informative, ["missingness_mechanism", "mask_rate", "strategy"], missing_metrics
        ),
        "remediation_mean_std": mean_std(remediation, ["method"], remediation_metrics),
    }
    for name, frame in summaries.items():
        frame.to_csv(out / f"{name}.csv", index=False)
        frame.to_markdown(out / f"table_{name}.md", index=False)

    manifests = [json.loads((root / "run_manifest.json").read_text(encoding="utf-8")) for root in roots]
    seeds = sorted({int(seed) for manifest in manifests for seed in manifest["seeds"]})
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "axes": AXES,
                "compound_interventions": [[AXES[i] for i in combo] for combo in COMBOS],
                "seeds": seeds,
                "n_train": manifests[0]["n_train"],
                "n_validation": manifests[0]["n_validation"],
                "n_test": manifests[0]["n_test"],
                "mask_rates": manifests[0]["mask_rates"],
                "missingness_mechanisms": manifests[0]["missingness_mechanisms"],
                "attribution_anchor_alpha": manifests[0]["attribution_anchor_alpha"],
                "remediation_semantics": manifests[0]["remediation_semantics"],
                "models": [model for manifest in manifests for model in manifest["models"]],
                "source_roots": [str(root) for root in roots],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Merged {len(seeds)} seeds into {out}")


if __name__ == "__main__":
    main()
