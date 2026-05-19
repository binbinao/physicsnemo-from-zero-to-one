#!/usr/bin/env python3
"""
Multi-objective design optimization (C38): minimize Cd and frontal-area proxy.

  python optimize_multi.py --checkpoint outputs/best.pt --n_trials 150

Objectives:
  1. Minimize predicted Cd
  2. Minimize width * height (frontal area proxy)

Requires Optuna >= 3.0 for multi-objective.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from optimize import load_model, check_design_space, predict_cd

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def main():
    p = argparse.ArgumentParser(description="Multi-objective surrogate optimization")
    p.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "outputs", "best.pt"))
    p.add_argument("--n_trials", type=int, default=150)
    args = p.parse_args()

    if not HAS_OPTUNA:
        print("Optuna required: pip install optuna")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"Missing checkpoint: {args.checkpoint}")
        sys.exit(1)

    model, ckpt = load_model(args.checkpoint)
    param_names = ckpt["param_names"]
    param_ranges = ckpt["param_ranges"]

    def objective(trial):
        params = {}
        for name in param_names:
            lo, hi = param_ranges[name]
            params[name] = trial.suggest_float(name, lo, hi)
        ok, warns = check_design_space(params, param_ranges)
        if not ok:
            trial.set_user_attr("ood_warnings", warns)
        cd = predict_cd(model, params, param_names, param_ranges)
        frontal = params["body_width"] * params["body_height"]
        return cd, frontal

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name="cd_frontal_pareto",
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    print(f"\nPareto front: {len(study.best_trials)} non-dominated trials (approx)")
    print("Top 5 by Cd:")
    sorted_trials = sorted(study.trials, key=lambda t: t.values[0] if t.values else 1e9)
    for t in sorted_trials[:5]:
        if t.values is None:
            continue
        print(f"  Cd={t.values[0]:.4f}  frontal={t.values[1]:.3f}  params={t.params}")

    out = os.path.join(os.path.dirname(args.checkpoint) or "outputs", "pareto_trials.csv")
    import csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        fields = ["trial", "cd", "frontal_proxy"] + param_names
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in study.trials:
            if t.values is None:
                continue
            row = {"trial": t.number, "cd": t.values[0], "frontal_proxy": t.values[1], **t.params}
            w.writerow(row)
    print(f"Wrote {out}")
    print("\nNext: python hifi_validation_queue.py --checkpoint", args.checkpoint)


if __name__ == "__main__":
    main()
