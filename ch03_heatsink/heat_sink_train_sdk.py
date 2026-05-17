#!/usr/bin/env python3
"""
Chapter 3 (SDK): Heat Sink 2D PINN with PhysicsNeMo FullyConnected
=====================================================================
Uses PhysicsNeMo's FullyConnected model with skip connections.
Solves steady-state heat conduction: k∇²T + Q = 0
on a 2D heat sink cross-section with mixed BCs.

Usage:
    python heat_sink_train_sdk.py --steps 500
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.models.mlp import FullyConnected


# ── Geometry: 2D heat sink cross-section ──────────────────
# Domain: [0, Lx] x [0, Ly] with fins on top
LX = 0.1   # 10 cm base width
LY_BASE = 0.01  # 1 cm base height
LY_FIN = 0.05   # 5 cm fin height
N_FINS = 5
FIN_WIDTH = 0.005  # 5 mm per fin
FIN_GAP = (LX - N_FINS * FIN_WIDTH) / (N_FINS + 1)


def is_inside_domain(x, y):
    """Check if point is in the heat sink geometry."""
    # Base region: always inside
    in_base = (0 <= x <= LX) and (0 <= y <= LY_BASE)
    if in_base:
        return True
    # Fin region
    if y <= LY_BASE + LY_FIN:
        for i in range(N_FINS):
            fin_left = FIN_GAP + i * (FIN_WIDTH + FIN_GAP)
            fin_right = fin_left + FIN_WIDTH
            if fin_left <= x <= fin_right:
                return True
    return False


def sample_interior(n, device="cpu"):
    """Sample random interior points."""
    pts = []
    while len(pts) < n:
        x = np.random.uniform(0, LX, n * 3)
        y = np.random.uniform(0, LY_BASE + LY_FIN, n * 3)
        for xi, yi in zip(x, y):
            if is_inside_domain(xi, yi):
                pts.append([xi, yi])
                if len(pts) >= n:
                    break
    return torch.tensor(pts, dtype=torch.float32, device=device)


def sample_bottom(n, device="cpu"):
    """Bottom BC: T = T_source (Dirichlet)."""
    x = torch.rand(n, device=device) * LX
    y = torch.zeros(n, device=device)
    return torch.stack([x, y], dim=1)


def sample_top_robin(n, device="cpu"):
    """Top + sides of fins: Robin BC (convection)."""
    pts = []
    for _ in range(n * 5):
        # Randomly pick fin top or fin sides
        i = np.random.randint(N_FINS)
        fin_left = FIN_GAP + i * (FIN_WIDTH + FIN_GAP)
        choice = np.random.choice(["top", "left", "right"])
        if choice == "top":
            x = np.random.uniform(fin_left, fin_left + FIN_WIDTH)
            y = LY_BASE + LY_FIN
            nx, ny = 0.0, 1.0
        elif choice == "left":
            x = fin_left
            y = np.random.uniform(LY_BASE, LY_BASE + LY_FIN)
            nx, ny = -1.0, 0.0
        else:
            x = fin_left + FIN_WIDTH
            y = np.random.uniform(LY_BASE, LY_BASE + LY_FIN)
            nx, ny = 1.0, 0.0
        pts.append([x, y, nx, ny])
        if len(pts) >= n:
            break
    pts = torch.tensor(pts[:n], dtype=torch.float32, device=device)
    return pts[:, :2], pts[:, 2:]  # coords, normals


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PhysicsNeMo SDK] Device: {device}")

    # ── PhysicsNeMo model: 2D input (x,y) → 1D output (T) ──
    model = FullyConnected(
        in_features=2,
        out_features=1,
        layer_size=args.hidden,
        num_layers=args.depth,
        activation_fn="tanh",
        skip_connections=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PhysicsNeMo FullyConnected: {n_params:,} params (skip_connections=True)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Physical params
    k = args.k           # thermal conductivity (W/m·K)
    Q = args.Q           # heat source (W/m³)
    h_conv = args.h_conv # convection coefficient
    T_inf = args.T_inf   # ambient temperature
    T_src = args.T_source

    history = {"total": [], "pde": [], "bot": [], "robin": []}

    for step in range(1, args.steps + 1):
        model.train()

        # Interior PDE: k∇²T + Q = 0
        pts_int = sample_interior(args.n_interior, device)
        pts_int.requires_grad_(True)
        T_int = model(pts_int)
        dT = torch.autograd.grad(T_int, pts_int, torch.ones_like(T_int),
                                  create_graph=True)[0]
        d2T_dx2 = torch.autograd.grad(dT[:, 0:1], pts_int, torch.ones_like(T_int),
                                       create_graph=True)[0][:, 0:1]
        d2T_dy2 = torch.autograd.grad(dT[:, 1:2], pts_int, torch.ones_like(T_int),
                                       create_graph=True)[0][:, 1:2]
        laplacian = d2T_dx2 + d2T_dy2
        pde_res = k * laplacian + Q
        l_pde = (pde_res ** 2).mean()

        # Bottom Dirichlet: T = T_source
        pts_bot = sample_bottom(args.n_bottom, device)
        T_bot = model(pts_bot)
        l_bot = ((T_bot - T_src) ** 2).mean()

        # Robin BC on fin surfaces: -k dT/dn = h(T - T_inf)
        pts_rob, normals = sample_top_robin(args.n_robin, device)
        pts_rob.requires_grad_(True)
        T_rob = model(pts_rob)
        dT_rob = torch.autograd.grad(T_rob, pts_rob, torch.ones_like(T_rob),
                                      create_graph=True)[0]
        dT_dn = (dT_rob * normals).sum(dim=1, keepdim=True)
        robin_res = -k * dT_dn - h_conv * (T_rob - T_inf)
        l_robin = (robin_res ** 2).mean()

        loss = args.w_pde * l_pde + args.w_bottom * l_bot + args.w_robin * l_robin

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        for key, val in [("total", loss), ("pde", l_pde),
                          ("bot", l_bot), ("robin", l_robin)]:
            history[key].append(val.item())

        if step % max(1, args.steps // 20) == 0 or step == 1:
            print(f"[{step:5d}/{args.steps}] loss={loss.item():.4e} "
                  f"pde={l_pde.item():.4e} bot={l_bot.item():.4e} robin={l_robin.item():.4e}")

    # ── Save ──────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "heat_sink_sdk.pt")
    torch.save({"model_state": model.state_dict(), "history": history}, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # ── Visualize ─────────────────────────────────────────
    model.eval()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax = axes[0]
    for key, color in [("pde", "blue"), ("bot", "orange"), ("robin", "red")]:
        ax.semilogy(history[key], alpha=0.7, color=color, label=key)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Heat Sink PINN (SDK)")
    ax.legend()

    # Temperature
    ax = axes[1]
    n_vis = 3000
    pts_vis = sample_interior(n_vis, device)
    with torch.no_grad():
        T_vis = model(pts_vis).cpu().numpy()
    pts_np = pts_vis.cpu().numpy()
    sc = ax.scatter(pts_np[:, 0] * 1000, pts_np[:, 1] * 1000,
                     c=T_vis.flatten(), cmap="hot", s=3)
    plt.colorbar(sc, ax=ax, label="T (K)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Temperature field")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "heat_sink_sdk.png")
    plt.savefig(fig_path, dpi=120)
    print(f"Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=float, default=200.0)
    parser.add_argument("--Q", type=float, default=1e5)
    parser.add_argument("--T_source", type=float, default=350.0)
    parser.add_argument("--h_conv", type=float, default=50.0)
    parser.add_argument("--T_inf", type=float, default=300.0)
    parser.add_argument("--n_interior", type=int, default=1000)
    parser.add_argument("--n_bottom", type=int, default=200)
    parser.add_argument("--n_robin", type=int, default=300)
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bottom", type=float, default=10.0)
    parser.add_argument("--w_robin", type=float, default=1.0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
