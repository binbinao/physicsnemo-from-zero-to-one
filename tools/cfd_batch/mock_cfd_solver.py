#!/usr/bin/env python3
"""
Mock high-fidelity CFD solver for closed-loop demo (no Fluent/Icepak license).

Reads design_params.json in cwd, writes result_cd.txt using the same
ground-truth physics as ch07 generate_toy_cars + small mesh/solver noise.

Usage (from case work_dir):
    python /path/to/tools/cfd_batch/mock_cfd_solver.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Repo root for import
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ch07_drivaernet_optim" / "data"))
from generate_toy_cars import PARAM_NAMES, compute_cd  # noqa: E402


def main() -> None:
    work = Path.cwd()
    params_path = work / "design_params.json"
    if not params_path.exists():
        print(f"Missing {params_path}", file=sys.stderr)
        sys.exit(1)

    params = json.loads(params_path.read_text(encoding="utf-8"))
    vec = np.array([[params[n] for n in PARAM_NAMES]], dtype=np.float32)
    cd_true = float(compute_cd(vec)[0])
    # Simulate mesh/solver discrepancy vs surrogate training noise
    rng = np.random.RandomState(hash(work.name) % (2**32))
    cd_cfd = cd_true + rng.normal(0, 0.008)
    cd_cfd = float(np.clip(cd_cfd, 0.15, 0.60))

    (work / "result_cd.txt").write_text(f"cd={cd_cfd:.6f}\n", encoding="utf-8")
    meta = {
        "solver": "mock_cfd",
        "cd_ground_truth_formula": cd_true,
        "cd_reported": cd_cfd,
        "note": "Replace with Fluent/OpenFOAM for production",
    }
    (work / "cfd_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"mock CFD: cd={cd_cfd:.4f} (formula={cd_true:.4f}) -> {work / 'result_cd.txt'}")


if __name__ == "__main__":
    main()
