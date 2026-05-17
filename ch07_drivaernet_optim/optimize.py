#!/usr/bin/env python3
"""
Chapter 7: Design Optimization with Trained Surrogate
=======================================================
  python optimize.py --checkpoint outputs/best.pt --n_trials 500

Uses Optuna if available, otherwise falls back to random search.
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from models.cd_mlp import CdMLP

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def load_model(ckpt_path):
    """Load trained model from checkpoint."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    train_args = ckpt.get("args", {})
    model = CdMLP(
        in_features=train_args.get("in_features", 7),
        hidden_dim=train_args.get("hidden", 128),
        n_layers=train_args.get("layers", 4),
        dropout=train_args.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict_cd(model, params_dict, param_names, param_ranges):
    """Predict Cd from a parameter dictionary."""
    x = torch.tensor([[params_dict[n] for n in param_names]], dtype=torch.float32)
    with torch.no_grad():
        cd = model(x).item()
    return cd


def optuna_optimize(model, ckpt, n_trials):
    """Optimize using Optuna."""
    param_names = ckpt["param_names"]
    param_ranges = ckpt["param_ranges"]

    def objective(trial):
        params = {}
        for name in param_names:
            lo, hi = param_ranges[name]
            params[name] = trial.suggest_float(name, lo, hi)

        cd = predict_cd(model, params, param_names, param_ranges)
        # Penalize extreme designs (regularization)
        penalty = 0.0
        # Prefer longer cars (more aerodynamic)
        penalty += 0.001 * max(0, 4.0 - params["body_length"])
        # Penalize very low ground clearance (impractical)
        penalty += 0.01 * max(0, 0.12 - params["ground_clearance"])
        return cd + penalty

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def random_optimize(model, ckpt, n_trials):
    """Fallback: random search if Optuna not installed."""
    param_names = ckpt["param_names"]
    param_ranges = ckpt["param_ranges"]

    best_cd = float("inf")
    best_params = None
    rng = np.random.RandomState(42)

    for _ in range(n_trials):
        params = {}
        for name in param_names:
            lo, hi = param_ranges[name]
            params[name] = rng.uniform(lo, hi)

        cd = predict_cd(model, params, param_names, param_ranges)
        if cd < best_cd:
            best_cd = cd
            best_params = params.copy()

    return best_cd, best_params


def main():
    parser = argparse.ArgumentParser(description="Design optimization via surrogate")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "best.pt"))
    parser.add_argument("--n_trials", type=int, default=500)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run train.py first.")
        sys.exit(1)

    model, ckpt = load_model(args.checkpoint)
    param_names = ckpt["param_names"]
    param_ranges = ckpt["param_ranges"]

    # Baseline: center of parameter space
    baseline_params = {n: (lo + hi) / 2 for n, (lo, hi) in param_ranges.items()}
    baseline_cd = predict_cd(model, baseline_params, param_names, param_ranges)
    print(f"Baseline design (center of param space): Cd = {baseline_cd:.4f}")

    # Optimize
    if HAS_OPTUNA:
        print(f"\nOptimizing with Optuna ({args.n_trials} trials)...")
        study = optuna_optimize(model, ckpt, args.n_trials)
        best_params = study.best_params
        best_cd = predict_cd(model, best_params, param_names, param_ranges)
    else:
        print(f"\nOptuna not installed, using random search ({args.n_trials} trials)...")
        best_cd, best_params = random_optimize(model, ckpt, args.n_trials)

    # Report
    improvement = (baseline_cd - best_cd) / baseline_cd * 100
    print(f"\n{'='*55}")
    print(f"  Optimization Results")
    print(f"{'='*55}")
    print(f"  Baseline Cd:  {baseline_cd:.4f}")
    print(f"  Optimized Cd: {best_cd:.4f}")
    print(f"  Improvement:  {improvement:.1f}%")
    print(f"\n  Best design parameters:")
    for name in param_names:
        lo, hi = param_ranges[name]
        val = best_params[name]
        pct = (val - lo) / (hi - lo) * 100
        print(f"    {name:20s} = {val:.3f}  ({pct:.0f}% of range)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
