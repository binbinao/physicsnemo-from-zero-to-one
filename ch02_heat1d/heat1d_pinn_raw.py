"""
Chapter 2: 1D Heat Equation PINN (Raw PyTorch)
================================================
PDE:  ∂u/∂t = α ∂²u/∂x²
      α = 0.1, x ∈ [0, 1], t ∈ [0, 0.5]

IC:   u(x, 0) = exp(-50 (x - 0.5)²)   (Gaussian peak)
BC:   u(0, t) = u(1, t) = 0            (Dirichlet)

This script trains a PINN from scratch using only PyTorch.
No PhysicsNeMo or other framework dependencies.

Usage:
    python heat1d_pinn_raw.py
    python heat1d_pinn_raw.py --hidden 64 --depth 4 --steps 5000
"""

import argparse
import os
import math

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class PINN(nn.Module):
    """Fully-connected network: (x, t) -> u(x, t)."""

    def __init__(self, hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def sample_interior(n: int, device: str) -> tuple:
    """Random collocation points inside the domain."""
    x = torch.rand(n, 1, device=device, requires_grad=True)
    t = torch.rand(n, 1, device=device, requires_grad=True) * 0.5
    return x, t


def sample_ic(n: int, device: str) -> tuple:
    """Initial condition points at t = 0."""
    x = torch.rand(n, 1, device=device, requires_grad=True)
    t = torch.zeros(n, 1, device=device, requires_grad=True)
    u_exact = torch.exp(-50.0 * (x - 0.5) ** 2)
    return x, t, u_exact


def sample_bc(n: int, device: str) -> tuple:
    """Boundary condition points at x = 0 and x = 1."""
    t = torch.rand(n, 1, device=device, requires_grad=True) * 0.5
    # x = 0
    x_left = torch.zeros(n, 1, device=device, requires_grad=True)
    # x = 1
    x_right = torch.ones(n, 1, device=device, requires_grad=True)
    return x_left, x_right, t


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------
def pde_residual(model: PINN, x: torch.Tensor, t: torch.Tensor,
                 alpha: float = 0.1) -> torch.Tensor:
    """Compute ∂u/∂t - α ∂²u/∂x²."""
    u = model(x, t)
    # First derivatives
    du = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u),
                             create_graph=True)
    du_dx, du_dt = du[0], du[1]
    # Second derivative w.r.t. x
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx),
                                   create_graph=True)[0]
    return du_dt - alpha * d2u_dx2


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = PINN(hidden=args.hidden, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steps // 3,
                                                 gamma=0.5)

    # Loss weights
    w_pde, w_ic, w_bc = 1.0, 10.0, 10.0

    history = {"loss": [], "pde": [], "ic": [], "bc": []}

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()

        # --- PDE loss ---
        x_int, t_int = sample_interior(args.n_pde, device)
        res = pde_residual(model, x_int, t_int, alpha=0.1)
        loss_pde = (res ** 2).mean()

        # --- IC loss ---
        x_ic, t_ic, u_ic = sample_ic(args.n_ic, device)
        u_pred_ic = model(x_ic, t_ic)
        loss_ic = ((u_pred_ic - u_ic) ** 2).mean()

        # --- BC loss ---
        x_left, x_right, t_bc = sample_bc(args.n_bc, device)
        loss_bc = (model(x_left, t_bc) ** 2).mean() + \
                  (model(x_right, t_bc) ** 2).mean()

        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()

        history["loss"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["bc"].append(loss_bc.item())

        if step % max(1, args.steps // 20) == 0 or step == 1:
            print(f"[Step {step:5d}/{args.steps}]  loss={loss.item():.4e}  "
                  f"pde={loss_pde.item():.4e}  ic={loss_ic.item():.4e}  "
                  f"bc={loss_bc.item():.4e}")

    # Save checkpoint
    os.makedirs("outputs", exist_ok=True)
    ckpt_path = "outputs/heat1d_raw.pt"
    torch.save({"model": model.state_dict(), "args": vars(args),
                "history": history}, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

    return model, history


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(model, history, device="cpu"):
    os.makedirs("outputs", exist_ok=True)
    model.eval()

    # --- Loss curves ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for key in ["loss", "pde", "ic", "bc"]:
        ax.semilogy(history[key], label=key, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/loss_curves.png", dpi=150)
    print("Saved outputs/loss_curves.png")
    plt.close(fig)

    # --- Temperature evolution ---
    nx, nt = 200, 6
    x_np = np.linspace(0, 1, nx)
    t_vals = np.linspace(0, 0.5, nt)
    x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1).to(device)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    with torch.no_grad():
        for ti in t_vals:
            t_t = torch.full_like(x_t, ti)
            u = model(x_t, t_t).cpu().numpy().flatten()
            ax.plot(x_np, u, label=f"t={ti:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("Temperature Evolution (PINN Prediction)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/temperature_evolution.png", dpi=150)
    print("Saved outputs/temperature_evolution.png")
    plt.close(fig)

    # --- 2D heatmap ---
    nx2, nt2 = 200, 200
    x_grid = np.linspace(0, 1, nx2)
    t_grid = np.linspace(0, 0.5, nt2)
    X, T = np.meshgrid(x_grid, t_grid)
    xt = torch.tensor(np.stack([X.ravel(), T.ravel()], axis=1),
                      dtype=torch.float32).to(device)
    with torch.no_grad():
        U = model.net(xt).cpu().numpy().reshape(nt2, nx2)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    c = ax.pcolormesh(X, T, U, shading="auto", cmap="hot")
    fig.colorbar(c, ax=ax, label="u(x,t)")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Temperature Field")
    fig.tight_layout()
    fig.savefig("outputs/temperature_field.png", dpi=150)
    print("Saved outputs/temperature_field.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D Heat Equation PINN")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer width")
    parser.add_argument("--depth", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_pde", type=int, default=2000, help="PDE collocation points")
    parser.add_argument("--n_ic", type=int, default=200, help="IC sample points")
    parser.add_argument("--n_bc", type=int, default=200, help="BC sample points per side")
    args = parser.parse_args()

    model, history = train(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    visualize(model, history, device)
    print("\nDone! Check the outputs/ directory for plots.")
