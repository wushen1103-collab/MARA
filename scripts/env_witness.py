#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys


def version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return f"missing: {exc}"
    return str(getattr(mod, "__version__", "unknown"))


def main() -> None:
    state = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
            "sklearn": version("sklearn"),
            "xgboost": version("xgboost"),
            "rdkit": version("rdkit"),
            "tdc": version("tdc"),
        },
    }
    try:
        state["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        state["nvidia_smi"] = f"unavailable: {exc}"
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
