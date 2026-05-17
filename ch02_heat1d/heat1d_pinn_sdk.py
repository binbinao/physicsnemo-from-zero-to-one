#!/usr/bin/env python3
"""
Chapter 2 (SDK): 1D Heat Conduction PINN using PhysicsNeMo FullyConnected
===========================================================================
Uses physicsnemo.models.mlp.FullyConnected as the backbone.
Same physics, same loss — just swapping the network for PhysicsNeMo's.

Usage:
    python heat1d_pinn_sdk.py --steps 1000
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PhysicsNeMo model
from physicsnemo.models.mlp import FullyConnected


# ── Physics parameters ─────────────────────────────────────
ALPHA = 0.01    # thermal diffusivity
T_MAX = 1.0     # time domain [0, T_MAX]
L = 1.0         # spatial domain [0, L]


def sample_pde_points(n, device="cpu"):
    """Random interior points for PDE residual."""
    t = torch.rand(n, 1, device=device) * T_MAX
    x = torch.rand(n, 1, device=device) * L
    return t, x


def sample_ic_points(n, device="cpu"):
    """Initial condition: T(x, 0) = sin(pi * x)"""
    x = torch.rand(n, 1, device=device) * L
    t = torch.zeros(n, 1, device=device)
    T_exact = torch.sin(np.pi * x)
    return t, x, T_exact


def sample_bc_points(n, device="cpu"):
    """Boundary conditions: T(0,t) = T(1,t) = 0"""
    t = torch.rand(n, 1, device=device) * T_MAX
    # Left boundary
    x_left = torch.zeros(n // 2, 1, device=device)
    t_left = t[:n // 2]
    # Right boundary
    x_right = torch.ones(n - n // 2, 1, device=device) * L
    t_right = t[n // 2:]
    x_bc = torch.cat([x_left, x_right], dim=0)
    t_bc = torch.cat([t_left, t_right], dim=0)
    T_bc = torch.zeros(n, 1, device=device)
    return t_bc, x_bc, T_bc


def pde_residual(model, t, x):
    """Compute PDE residual: dT/dt - alpha * d²T/dx²"""
    t.requires_grad_(True)
    x.requires_grad_(True)
    inp = torch.cat([t, x], dim=1)
    T = model(inp)

    dT_dt = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True)[0]
    dT_dx = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x, torch.ones_like(dT_dx), create_graph=True)[0]

    return dT_dt - ALPHA * d2T_dx2


def analytical_solution(t, x):
    """Exact: T(x,t) = sin(pi*x) * exp(-alpha * pi^2 * t)"""
    return np.sin(np.pi * x) * np.exp(-ALPHA * np.pi**2 * t)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PhysicsNeMo SDK] Device: {device}")

    # ── PhysicsNeMo FullyConnected model ──────────────────
    # in_features=2 (t, x), out_features=1 (T)
    model = FullyConnected(
        in_features=2,
        out_features=1,
        layer_size=args.hidden,
        num_layers=args.depth,
        activation_fn="tanh",
        skip_connections=True,  # PhysicsNeMo feature: skip connections
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PhysicsNeMo FullyConnected: {n_params:,} parameters")
    print(f"  hidden={args.hidden}, depth={args.depth}, skip_connections=True")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steps // 3, gamma=0.5)

    history = {"pde": [], "ic": [], "bc": [], "total": []}

    for step in range(1, args.steps + 1):
        model.train()

        # Sample points
        t_pde, x_pde = sample_pde_points(args.n_pde, device)
        t_ic, x_ic, T_ic = sample_ic_points(args.n_ic, device)
        t_bc, x_bc, T_bc = sample_bc_points(args.n_bc, device)

        # PDE loss
        res = pde_residual(model, t_pde, x_pde)
        l_pde = (res ** 2).mean()

        # IC loss
        T_pred_ic = model(torch.cat([t_ic, x_ic], dim=1))
        l_ic = nn.functional.mse_loss(T_pred_ic, T_ic)

        # BC loss
        T_pred_bc = model(torch.cat([t_bc, x_bc], dim=1))
        l_bc = nn.functional.mse_loss(T_pred_bc, T_bc)

        # Weighted total
        loss = 10 * l_pde + 100 * l_ic + 100 * l_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        history["pde"].append(l_pde.item())
        history["ic"].append(l_ic.item())
        history["bc"].append(l_bc.item())
        history["total"].append(loss.item())

        if step % max(1, args.steps // 20) == 0 or step == 1:
            print(f"[Step {step:5d}/{args.steps}]  loss={loss.item():.4e}  "
                  f"pde={l_pde.item():.4e}  ic={l_ic.item():.4e}  bc={l_bc.item():.4e}")

    # ── Checkpoint using PhysicsNeMo utils ────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "heat1d_sdk.pt")
    torch.save({
        "model_state": model.state_dict(),
        "history": history,
        "args": vars(args),
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

    # ── Visualize ─────────────────────────────────────────
    model.eval()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    ax = axes[0]
    for key, color in [("pde", "blue"), ("ic", "orange"), ("bc", "green")]:
        ax.semilogy(history[key], alpha=0.7, color=color, label=key)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training losses (SDK)")
    ax.legend()

    # Temperature at t=0.5
    ax = axes[1]
    x_test = np.linspace(0, L, 100)
    t_test = 0.5
    with torch.no_grad():
        inp = torch.tensor(
            np.column_stack([np.full_like(x_test, t_test), x_test]),
            dtype=torch.float32, device=device)
        T_pred = model(inp).cpu().numpy().flatten()
    T_exact = analytical_solution(t_test, x_test)
    ax.plot(x_test, T_exact, "k--", label="Analytical")
    ax.plot(x_test, T_pred, "r-", label="PINN (SDK)")
    ax.set_xlabel("x")
    ax.set_ylabel("T")
    ax.set_title(f"Temperature at t={t_test}")
    ax.legend()

    # Error heatmap
    ax = axes[2]
    t_grid = np.linspace(0, T_MAX, 50)
    x_grid = np.linspace(0, L, 50)
    TT, XX = np.meshgrid(t_grid, x_grid)
    with torch.no_grad():
        inp = torch.tensor(
            np.column_stack([TT.ravel(), XX.ravel()]),
            dtype=torch.float32, device=device)
        T_pred_grid = model(inp).cpu().numpy().reshape(TT.shape)
    T_exact_grid = analytical_solution(TT, XX)
    error = np.abs(T_pred_grid - T_exact_grid)
    c = ax.pcolormesh(t_grid, x_grid, error, cmap="hot")
    plt.colorbar(c, ax=ax)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Absolute error")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "heat1d_sdk_results.png")
    plt.savefig(fig_path, dpi=120)
    print(f"Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_pde", type=int, default=2000)
    parser.add_argument("--n_ic", type=int, default=200)
    parser.add_argument("--n_bc", type=int, default=200)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
