#!/usr/bin/env python3
"""
Multi-objective design optimization: minimize Cd and frontal-area proxy.

  python optimize_multi.py --checkpoint outputs/best.pt --n_trials 150
  python optimize_multi.py --export-hifi --top-k 5   # feasible Pareto → hifi_queue.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from design_constraints import check_engineering_constraints, penalty_score
from optimize import load_model, predict_cd

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class _TrialLike:
    def __init__(self, number: int, params: dict, values: tuple[float, float]):
        self.number = number
        self.params = params
        self.values = values


def _pareto_front(trials: list[_TrialLike]) -> list[_TrialLike]:
    front: list[_TrialLike] = []
    for t in trials:
        if t.values is None:
            continue
        dominated = False
        for u in trials:
            if u is t or u.values is None:
                continue
            if u.values[0] <= t.values[0] and u.values[1] <= t.values[1]:
                if u.values[0] < t.values[0] or u.values[1] < t.values[1]:
                    dominated = True
                    break
        if not dominated:
            front.append(t)
    return front


def random_multi_optimize(model, ckpt, n_trials: int) -> tuple[list[_TrialLike], list[_TrialLike]]:
    param_names = ckpt["param_names"]
    param_ranges = ckpt["param_ranges"]
    rng = np.random.RandomState(42)
    trials: list[_TrialLike] = []
    for i in range(n_trials):
        params = {n: rng.uniform(*param_ranges[n]) for n in param_names}
        cd = predict_cd(model, params, param_names, param_ranges)
        frontal = params["body_width"] * params["body_height"]
        pen = penalty_score(params)
        trials.append(_TrialLike(i, params, (cd + pen, frontal)))
    return trials, _pareto_front(trials)


def export_hifi_csv(trials, param_names, out_path: str, top_k: int) -> int:
    """Export feasible non-dominated trials sorted by Cd."""
    feas = []
    for t in trials:
        if t.values is None or not t.params:
            continue
        ok, _ = check_engineering_constraints(t.params)
        if ok:
            feas.append(t)
    feas = sorted(feas, key=lambda t: t.values[0])[:top_k]
    if not feas:
        return 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = ["rank", "predicted_cd"] + param_names
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t in enumerate(feas):
            row = {"rank": i + 1, "predicted_cd": t.values[0], **t.params}
            w.writerow(row)
    return len(feas)


def main():
    p = argparse.ArgumentParser(description="Multi-objective surrogate optimization")
    p.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "outputs", "best.pt"))
    p.add_argument("--n_trials", type=int, default=150)
    p.add_argument("--export-hifi", action="store_true", help="Write outputs/hifi_queue.csv from feasible Pareto")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--hifi-out", default="outputs/hifi_queue.csv")
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Missing checkpoint: {args.checkpoint}")
        sys.exit(1)

    model, ckpt = load_model(args.checkpoint)
    param_names = ckpt["param_names"]
    param_ranges = ckpt["param_ranges"]

    if HAS_OPTUNA:
        def objective(trial):
            params = {}
            for name in param_names:
                lo, hi = param_ranges[name]
                params[name] = trial.suggest_float(name, lo, hi)
            cd = predict_cd(model, params, param_names, param_ranges)
            frontal = params["body_width"] * params["body_height"]
            pen = penalty_score(params)
            return cd + pen, frontal

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            study_name="cd_frontal_pareto",
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
        all_trials = study.trials
        pareto_trials = study.best_trials
    else:
        print(f"Optuna not installed; random multi-objective search ({args.n_trials} trials)...")
        all_trials, pareto_trials = random_multi_optimize(model, ckpt, args.n_trials)

    feas_trials = [
        t for t in all_trials
        if t.values and check_engineering_constraints(t.params)[0]
    ]
    print(f"\nTrials: {len(all_trials)}  feasible: {len(feas_trials)}  Pareto~: {len(pareto_trials)}")

    out_dir = os.path.dirname(args.checkpoint) or "outputs"
    pareto_path = os.path.join(out_dir, "pareto_trials.csv")
    with open(pareto_path, "w", newline="", encoding="utf-8") as f:
        fields = ["trial", "cd", "frontal_proxy", "feasible"] + param_names
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in all_trials:
            if t.values is None:
                continue
            ok, _ = check_engineering_constraints(t.params)
            w.writerow({
                "trial": t.number,
                "cd": t.values[0],
                "frontal_proxy": t.values[1],
                "feasible": ok,
                **t.params,
            })
    print(f"Wrote {pareto_path}")

    if args.export_hifi:
        n = export_hifi_csv(pareto_trials or feas_trials, param_names, args.hifi_out, args.top_k)
        print(f"Exported {n} designs to {args.hifi_out} (engineering-feasible)")
        print("Next: python tools/cfd_batch/import_hifi_queue.py --csv", args.hifi_out, "--mode mock")


if __name__ == "__main__":
    main()
