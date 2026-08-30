#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path


def query_gpus() -> list[dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_gpu_percent": int(parts[4]),
            }
        )
    return rows


def load_average_1m() -> float:
    try:
        return float(os.getloadavg()[0])
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-requested", type=int, default=96)
    parser.add_argument("--reserve-cpu", type=int, default=30)
    parser.add_argument("--cpu-fraction", type=float, default=0.5)
    parser.add_argument("--max-gpus", type=int, default=3)
    parser.add_argument("--gpu-free-mib", type=int, default=2500)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cpu_total = os.cpu_count() or 1
    cpu_cap_fraction = max(1, int(cpu_total * args.cpu_fraction))
    cpu_cap_reserve = max(1, cpu_total - args.reserve_cpu)
    cpu_project_cap = min(cpu_cap_fraction, cpu_cap_reserve)
    if args.workers_requested > cpu_cap_reserve:
        raise SystemExit(f"Requested {args.workers_requested} exceeds CPU cap after reserve {cpu_cap_reserve}")
    load_aware_cap = max(1, cpu_project_cap - int(math.ceil(load_average_1m())))
    workers_selected = min(args.workers_requested, load_aware_cap)

    gpus = query_gpus()
    free_gpus = [
        g
        for g in gpus
        if g["memory_used_mib"] <= args.gpu_free_mib and g["utilization_gpu_percent"] <= 10
    ]
    selected = free_gpus[: args.max_gpus]
    state = {
        "cpu_total": cpu_total,
        "cpu_reserve": args.reserve_cpu,
        "cpu_cap_after_reserve": cpu_cap_reserve,
        "cpu_project_cap": cpu_project_cap,
        "cpu_load_1m": load_average_1m(),
        "workers_requested": args.workers_requested,
        "workers_selected": workers_selected,
        "gpu_free_threshold_mib": args.gpu_free_mib,
        "selected_gpu_ids": [g["index"] for g in selected],
        "free_gpu_ids": [g["index"] for g in free_gpus],
        "all_gpus": gpus,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"export MARA_WORKERS={workers_selected}")
    print(f"export CUDA_VISIBLE_DEVICES={','.join(str(g['index']) for g in selected)}")


if __name__ == "__main__":
    main()
