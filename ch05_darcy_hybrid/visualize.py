"""
Chapter 5: Comparison Visualization
======================================
Compare data-only FNO vs physics-informed FNO.

Usage:
    python visualize.py
"""

import os
import sys
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch04_fno_airfoil"))
from fno_model import FNO2D


def load_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        return None, None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = FNO2D(in_channels=1, out_channels=1,
                  width=cfg.get("width", 32),
                  modes_x=cfg.get("modes", 12),
                  modes_y=cfg.get("modes", 12),
                  n_layers=cfg.get("n_layers", 4))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("history", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_ckpt", default="outputs/fno_data_only.pt")
    parser.add_argument("--phys_ckpt", default="outputs/fno_physics.pt")
    parser.add_argument("--data", default="data/darcy_data.pt")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load models
    model_data, hist_data = load_model(args.data_ckpt)
    model_phys, hist_phys = load_model(args.phys_ckpt)

    if model_data is None and model_phys is None:
        print("No checkpoints found. Run train_data_fno.py and train_physics_fno.py first.")
        sys.exit(1)

    # --- Loss comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    if hist_data and "test" in hist_data:
        ax.semilogy(hist_data["test"], label="Data-only (test)", linestyle="--")
    if hist_phys and "test" in hist_phys:
        ax.semilogy(hist_phys["test"], label="Physics-informed (test)", linestyle="-")
    if hist_data and "train" in hist_data:
        ax.semilogy(hist_data["train"], label="Data-only (train)", linestyle="--", alpha=0.5)
    if hist_phys and "train" in hist_phys:
        ax.semilogy(hist_phys["train"], label="Physics-informed (train)", linestyle="-", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Data-only vs Physics-informed FNO")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "comparison_loss.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    # --- Prediction comparison ---
    if not os.path.exists(args.data):
        print(f"Data not found: {args.data}")
        return

    data = torch.load(args.data, weights_only=False)
    a_all, u_all = data['a'], data['u']
    n_train = 100
    a_test = a_all[n_train:n_train + 4]
    u_test = u_all[n_train:n_train + 4]

    models = {}
    if model_data is not None:
        models["Data-only"] = model_data
    if model_phys is not None:
        models["Physics-informed"] = model_phys

    n_show = min(4, len(a_test))
    n_cols = 2 + len(models)  # input + truth + predictions

    fig, axes = plt.subplots(n_show, n_cols, figsize=(4 * n_cols, 3 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        # Input
        axes[i, 0].imshow(a_test[i, 0].numpy(), cmap="viridis")
        axes[i, 0].set_title("Input a(x)" if i == 0 else "")

        # Truth
        axes[i, 1].imshow(u_test[i, 0].numpy(), cmap="RdBu_r")
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")

        for j, (name, model) in enumerate(models.items()):
            with torch.no_grad():
                pred = model(a_test[i:i+1])[0, 0].numpy()
            axes[i, 2 + j].imshow(pred, cmap="RdBu_r")
            axes[i, 2 + j].set_title(name if i == 0 else "")

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Darcy Flow: Data-only vs Physics-informed FNO", fontsize=14)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "comparison_predictions.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    # --- Error stats ---
    for name, model in models.items():
        with torch.no_grad():
            pred = model(a_all[n_train:n_train + 50])
            truth = u_all[n_train:n_train + 50]
            rel_err = torch.norm(pred - truth) / torch.norm(truth)
        print(f"{name:25s} | Relative L2 error: {rel_err.item():.4f}")


if __name__ == "__main__":
    main()
