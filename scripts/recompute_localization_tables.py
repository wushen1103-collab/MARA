#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_mara_public_suite import stress_localization_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    updated = 0
    for run_dir in sorted(root.glob("*")):
        axis_path = run_dir / "axis_manifest.csv"
        pred_path = run_dir / "predictions.csv"
        if not axis_path.exists() or not pred_path.exists():
            continue
        axis = pd.read_csv(axis_path)
        pred = pd.read_csv(pred_path)
        keep = ["row_id", *[c for c in pred.columns if c.startswith("contrib_")]]
        pred_keep = pred[[c for c in keep if c in pred.columns]]
        frame = axis.merge(pred_keep, on="row_id", how="left")
        test_idx = frame.index[frame["split_role"] == "test"].to_numpy()
        loc = stress_localization_table(frame, test_idx)
        if not loc.empty:
            loc.insert(0, "split", run_dir.name.rsplit("_", 2)[-2] if "_" in run_dir.name else "")
            loc.insert(0, "dataset_id", run_dir.name.rsplit("_", 2)[0] if "_" in run_dir.name else run_dir.name)
            loc.to_csv(run_dir / "stress_localization.csv", index=False)
            updated += 1
    print(f"updated {updated} localization tables under {root}")


if __name__ == "__main__":
    main()
