"""
Chapter 4: FNO Prediction Visualization
==========================================
Load FNO checkpoint and visualize predictions vs ground truth.

Usage:
    python visualize_airfoil.py
    python visualize_airfoil.py --ckpt outputs/fno_darcy.pt --n_show 4
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
from fno_model import FNO2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/fno_darcy.pt")
    parser.add_argument("--data", default="data/darcy_data.pt")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--n_show", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}. Run train_fno_mini.py first.")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"Data not found: {args.data}. Run dataset.py or train_fno_mini.py first.")
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = FNO2D(
        in_channels=1, out_channels=1,
        width=cfg.get("width", 32),
        modes_x=cfg.get("modes_x", 12),
        modes_y=cfg.get("modes_y", 12),
        n_layers=cfg.get("n_layers", 4),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = torch.load(args.data, weights_only=False)
    a_all, u_all = data['a'], data['u']
    # Use test set (last 20%)
    n_train = int(0.8 * len(a_all))
    a_test = a_all[n_train:]
    u_test = u_all[n_train:]

    os.makedirs(args.out_dir, exist_ok=True)
    n_show = min(args.n_show, len(a_test))

    with torch.no_grad():
        u_pred = model(a_test[:n_show])

    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        # Input permeability
        im0 = axes[i, 0].imshow(a_test[i, 0].numpy(), cmap="viridis")
        axes[i, 0].set_title(f"Input a(x) #{i}")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

        # Ground truth
        im1 = axes[i, 1].imshow(u_test[i, 0].numpy(), cmap="RdBu_r")
        axes[i, 1].set_title(f"True u(x) #{i}")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        # Prediction
        im2 = axes[i, 2].imshow(u_pred[i, 0].numpy(), cmap="RdBu_r")
        axes[i, 2].set_title(f"FNO Prediction #{i}")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("FNO2D: Darcy Flow Predictions", fontsize=14)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "fno_predictions.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    # Error statistics
    with torch.no_grad():
        u_pred_all = model(a_test)
    rel_error = torch.norm(u_pred_all - u_test) / torch.norm(u_test)
    print(f"Relative L2 error on test set: {rel_error.item():.4f}")

    # Per-sample errors
    fig, ax = plt.subplots(figsize=(8, 4))
    errors = []
    for i in range(len(a_test)):
        e = torch.norm(u_pred_all[i] - u_test[i]) / (torch.norm(u_test[i]) + 1e-8)
        errors.append(e.item())
    ax.bar(range(len(errors)), errors, alpha=0.7)
    ax.set_xlabel("Test Sample")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Per-Sample Prediction Error")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(args.out_dir, "error_distribution.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
