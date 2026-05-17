"""
Chapter 2: Visualize from Checkpoint
======================================
Load a saved checkpoint and produce temperature plots.

Usage:
    python heat1d_visualize.py
    python heat1d_visualize.py --ckpt outputs/heat1d_hydra.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class HeatPINN(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/heat1d_raw.pt",
                        help="Path to checkpoint")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        print("Run heat1d_pinn_raw.py or heat1d_train.py first.")
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("args", ckpt.get("config", {}))
    hidden = cfg.get("hidden", 64)
    depth = cfg.get("depth", 4)

    model = HeatPINN(hidden=hidden, depth=depth)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}  (hidden={hidden}, depth={depth})")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Loss history ---
    if "history" in ckpt:
        h = ckpt["history"]
        fig, ax = plt.subplots(figsize=(8, 4))
        for key in ["loss", "pde", "ic", "bc"]:
            if key in h:
                ax.semilogy(h[key], label=key, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss History")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(args.out_dir, "vis_loss.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
        plt.close(fig)

    # --- Temperature snapshots ---
    nx = 300
    x_np = np.linspace(0, 1, nx)
    x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
    t_vals = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    fig, ax = plt.subplots(figsize=(8, 5))
    with torch.no_grad():
        for ti in t_vals:
            t_t = torch.full_like(x_t, ti)
            u = model(x_t, t_t).numpy().flatten()
            ax.plot(x_np, u, label=f"t={ti:.2f}", linewidth=1.5)

    # Analytical IC for reference
    u0 = np.exp(-50 * (x_np - 0.5) ** 2)
    ax.plot(x_np, u0, "k--", alpha=0.4, label="IC (exact)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("PINN Temperature Prediction")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "vis_temperature.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    # --- Space-time heatmap ---
    nx2, nt2 = 200, 200
    xg = np.linspace(0, 1, nx2)
    tg = np.linspace(0, 0.5, nt2)
    X, T = np.meshgrid(xg, tg)
    xt = torch.tensor(np.stack([X.ravel(), T.ravel()], axis=1),
                      dtype=torch.float32)
    with torch.no_grad():
        U = model.net(xt).numpy().reshape(nt2, nx2)

    fig, ax = plt.subplots(figsize=(8, 4))
    c = ax.pcolormesh(X, T, U, shading="auto", cmap="hot")
    fig.colorbar(c, ax=ax, label="u(x,t)")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Temperature Field (x-t plane)")
    fig.tight_layout()
    path = os.path.join(args.out_dir, "vis_heatmap.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    print("Visualization complete.")


if __name__ == "__main__":
    main()
