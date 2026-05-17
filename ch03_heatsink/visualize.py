"""
Chapter 3: Temperature Field Visualization
============================================
Load checkpoint and produce contour plots.

Usage:
    python visualize.py
    python visualize.py --ckpt outputs/heat_sink.pt
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from heat_sink_geometry import HeatSinkGeometry


class HeatSinkNet(nn.Module):
    def __init__(self, hidden=128, depth=5):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/heat_sink.pt")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        print("Run heat_sink_train.py first.")
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    scale = ckpt.get("scale", 30.0)

    model = HeatSinkNet(hidden=cfg.get("hidden", 128), depth=cfg.get("depth", 5))
    model.load_state_dict(ckpt["model"])
    model.eval()

    geo = HeatSinkGeometry(fin_height=cfg.get("fin_height", 20.0),
                           n_fins=cfg.get("n_fins", 3))

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Loss curves ---
    if "history" in ckpt:
        h = ckpt["history"]
        fig, ax = plt.subplots(figsize=(8, 4))
        for key in h:
            ax.semilogy(h[key], label=key, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Heat Sink Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(args.out_dir, "loss_curves.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
        plt.close(fig)

    # --- Temperature contour ---
    nx, ny = 400, 250
    xg = np.linspace(-35, 35, nx)
    yg = np.linspace(-2, 35, ny)
    X, Y = np.meshgrid(xg, yg)

    xf = torch.tensor(X.ravel() / scale, dtype=torch.float32).unsqueeze(1)
    yf = torch.tensor(Y.ravel() / scale, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        T = model(xf, yf).numpy().reshape(ny, nx)

    mask = geo.contains(X.ravel(), Y.ravel()).reshape(ny, nx)
    T_masked = np.where(mask, T, np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    c = ax.contourf(X, Y, T_masked, levels=30, cmap="hot")
    fig.colorbar(c, ax=ax, label="Temperature")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Heat Sink Temperature Distribution")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(args.out_dir, "temperature_contour.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    # --- Vertical profile through center fin ---
    x_center = 0.0
    y_profile = np.linspace(0, 30, 200)
    xp = torch.full((200, 1), x_center / scale)
    yp = torch.tensor(y_profile / scale, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        T_profile = model(xp, yp).numpy().flatten()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(T_profile, y_profile, "r-", linewidth=2)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("y (mm)")
    ax.set_title("Temperature Profile (x = 0, center fin)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "center_profile.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    print("Visualization complete.")


if __name__ == "__main__":
    main()
