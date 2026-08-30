# Reference Results

`reference/` contains compact aggregate outputs copied from the frozen paper
experiments. They support numerical regression checks without distributing raw
datasets, model caches, per-instance predictions, logs, or machine manifests.

- `molecular/`: matched baselines, aggregate metrics, paired statistics,
  predictor transfer, ablation, and selective prediction;
- `external/`: BOOM and modern graph-backbone summaries;
- `action/`: strict four-way support-acquisition summaries;
- `tkde/`: cross-domain, controlled-cause, missingness, remediation, and
  scalability summaries.

The files are evidence snapshots, not runtime inputs. Regenerate them from
`scripts/` and compare against `reference/SHA256SUMS` when validating an exact
checkout.
