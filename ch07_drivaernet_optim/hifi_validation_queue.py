#!/usr/bin/env python3
"""
High-fidelity CFD re-validation queue (stub for CAE workflow).
=============================================================
After surrogate optimization, export Top-K designs for OpenFOAM/Fluent/Icepak rerun.

  python optimize.py --checkpoint outputs/best.pt --n_trials 200
  python hifi_validation_queue.py --checkpoint outputs/best.pt --top_k 5

Writes outputs/hifi_queue.csv — import into your CFD batch system.
Does NOT run CFD in-repo (no mesh/solver licenses).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from optimize import load_model, optuna_optimize, random_optimize

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def main():
    parser = argparse.ArgumentParser(description="Export Top-K designs for hi-fi CFD")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out", default="outputs/hifi_queue.csv")
    args = parser.parse_args()

    model, ckpt = load_model(args.checkpoint)
    if HAS_OPTUNA:
        study = optuna_optimize(model, ckpt, args.n_trials)
        trials = sorted(study.trials, key=lambda t: t.value)[: args.top_k]
        rows = []
        for i, t in enumerate(trials):
            row = {"rank": i + 1, "predicted_cd": t.value, **t.params}
            rows.append(row)
    else:
        best_cd, best_params = random_optimize(model, ckpt, args.n_trials)
        rows = [{"rank": 1, "predicted_cd": best_cd, **best_params}]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    meta = {
        "checkpoint": args.checkpoint,
        "data_tier": ckpt.get("meta", {}).get("data_tier", "unknown"),
        "note": "Run CFD on these rows; compare Cd to surrogate. See docs/CAE_DATA_GENERATION_SOP.md",
    }
    meta_path = os.path.join(os.path.dirname(args.out) or ".", "hifi_queue_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {args.out} ({len(rows)} designs)")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
