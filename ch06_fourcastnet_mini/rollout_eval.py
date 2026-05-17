"""
Chapter 6: Autoregressive Rollout Evaluation
==============================================
Evaluate the AFNO model by rolling it forward in time autoregressively.
Compare single-step vs multi-step accuracy to see error accumulation.

Usage:
    python rollout_eval.py
    python rollout_eval.py --ckpt outputs/afno_weather.pt --rollout_steps 20
"""

import os
import sys
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from afno_model import AFNOMini


def rollout(model, initial_state: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Autoregressive rollout: predict steps into the future.

    Args:
        model: trained AFNO model
        initial_state: (1, C, H, W) tensor
        steps: number of steps to roll forward

    Returns:
        predictions: (steps+1, C, H, W) tensor (includes initial state)
    """
    model.eval()
    device = next(model.parameters()).device
    predictions = [initial_state.squeeze(0).cpu()]

    current = initial_state.to(device)
    with torch.no_grad():
        for _ in range(steps):
            current = model(current)
            predictions.append(current.squeeze(0).cpu())

    return torch.stack(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/afno_weather.pt")
    parser.add_argument("--data", default="data/toy_weather.pt")
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}. Run train_afno_mini.py first.")
        sys.exit(1)

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data to determine n_vars
    if os.path.exists(args.data):
        weather = torch.load(args.data, weights_only=False)
        data = weather['data']  # (T, C, H, W)
    else:
        print(f"Data not found: {args.data}")
        sys.exit(1)

    n_vars = data.shape[1]
    model = AFNOMini(
        in_channels=n_vars, out_channels=n_vars,
        embed_dim=cfg.get("embed_dim", 64),
        depth=cfg.get("depth", 4),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model, rolling out {args.rollout_steps} steps...")

    # Use last portion of data as test
    n_train = int(0.8 * len(data))
    test_data = data[n_train:]
    start_idx = 0
    initial = test_data[start_idx:start_idx + 1]

    # Rollout
    steps = min(args.rollout_steps, len(test_data) - start_idx - 1)
    preds = rollout(model, initial, steps)  # (steps+1, C, H, W)
    truth = test_data[start_idx:start_idx + steps + 1]

    # Compute per-step RMSE
    var_names = ["T", "Z", "U", "V"][:n_vars]
    rmse_per_step = []
    for t in range(steps + 1):
        rmse = torch.sqrt(((preds[t] - truth[t]) ** 2).mean()).item()
        rmse_per_step.append(rmse)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- RMSE vs rollout step ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(steps + 1), rmse_per_step, "o-", linewidth=2)
    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("RMSE")
    ax.set_title("Autoregressive Rollout: Error Accumulation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "rollout_rmse.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    # --- Per-variable RMSE ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in range(n_vars):
        var_rmse = []
        for t in range(steps + 1):
            r = torch.sqrt(((preds[t, v] - truth[t, v]) ** 2).mean()).item()
            var_rmse.append(r)
        ax.plot(range(steps + 1), var_rmse, "o-", label=var_names[v], markersize=4)
    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("RMSE")
    ax.set_title("Per-Variable Rollout Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "rollout_per_var.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    # --- Snapshots: truth vs prediction at select steps ---
    show_steps = [0, steps // 4, steps // 2, steps]
    show_steps = [s for s in show_steps if s <= steps]
    n_show = len(show_steps)
    var_idx = 0  # Temperature

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 6))
    for j, t in enumerate(show_steps):
        im0 = axes[0, j].imshow(truth[t, var_idx].numpy(), cmap="RdBu_r")
        axes[0, j].set_title(f"Truth t+{t}")
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])

        im1 = axes[1, j].imshow(preds[t, var_idx].numpy(), cmap="RdBu_r")
        axes[1, j].set_title(f"Pred t+{t}")
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])

    axes[0, 0].set_ylabel("Ground Truth")
    axes[1, 0].set_ylabel("AFNO Prediction")
    fig.suptitle(f"Temperature Field Rollout ({var_names[var_idx]})", fontsize=14)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "rollout_snapshots.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    print(f"\nRollout complete. Final RMSE at step {steps}: {rmse_per_step[-1]:.4f}")


if __name__ == "__main__":
    main()
