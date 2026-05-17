#!/usr/bin/env python3
"""
Chapter 2 (GPU): 1D Heat Conduction PINN — Multi-GPU with PhysicsNeMo
========================================================================
Full-scale training with:
  - Mixed precision (AMP)
  - Multi-GPU via DistributedDataParallel (DDP)
  - PhysicsNeMo checkpoint utilities
  - Larger network + more collocation points

Usage:
    # Single GPU
    python heat1d_pinn_gpu.py --steps 5000

    # Multi-GPU (e.g., 4 GPUs)
    torchrun --nproc_per_node=4 heat1d_pinn_gpu.py --steps 5000 --ddp
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.models.mlp import FullyConnected
from physicsnemo.utils import load_checkpoint, save_checkpoint

# ── Physics ───────────────────────────────────────────────
ALPHA = 0.01
T_MAX = 1.0
L = 1.0


def setup_ddp():
    """Initialize DDP if available."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size, True
    return 0, 1, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def sample_pde_points(n, device):
    t = torch.rand(n, 1, device=device) * T_MAX
    x = torch.rand(n, 1, device=device) * L
    return t, x


def sample_ic_points(n, device):
    x = torch.rand(n, 1, device=device) * L
    t = torch.zeros(n, 1, device=device)
    T_exact = torch.sin(np.pi * x)
    return t, x, T_exact


def sample_bc_points(n, device):
    t = torch.rand(n, 1, device=device) * T_MAX
    x_left = torch.zeros(n // 2, 1, device=device)
    t_left = t[:n // 2]
    x_right = torch.ones(n - n // 2, 1, device=device) * L
    t_right = t[n // 2:]
    x_bc = torch.cat([x_left, x_right], dim=0)
    t_bc = torch.cat([t_left, t_right], dim=0)
    T_bc = torch.zeros(n, 1, device=device)
    return t_bc, x_bc, T_bc


def pde_residual(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    inp = torch.cat([t, x], dim=1)
    T = model(inp)
    dT_dt = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True)[0]
    dT_dx = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x, torch.ones_like(dT_dx), create_graph=True)[0]
    return dT_dt - ALPHA * d2T_dx2


def analytical_solution(t, x):
    return np.sin(np.pi * x) * np.exp(-ALPHA * np.pi**2 * t)


def train(args):
    rank, world_size, use_ddp = setup_ddp() if args.ddp else (0, 1, False)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[Heat1D GPU] Device: {device}, World size: {world_size}")

    # ── Model ─────────────────────────────────────────────
    model = FullyConnected(
        in_features=2,
        out_features=1,
        layer_size=args.hidden,
        num_layers=args.depth,
        activation_fn="tanh",
        skip_connections=True,
        weight_norm=True,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank])

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params:,} parameters | hidden={args.hidden}, depth={args.depth}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.steps // 4, T_mult=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Training ──────────────────────────────────────────
    history = {"pde": [], "ic": [], "bc": [], "total": []}
    best_loss = float("inf")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_gpu")
    os.makedirs(out_dir, exist_ok=True)

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            # PDE
            t_pde, x_pde = sample_pde_points(args.n_pde, device)
            res = pde_residual(model, t_pde, x_pde)
            l_pde = (res ** 2).mean()

            # IC
            t_ic, x_ic, T_ic = sample_ic_points(args.n_ic, device)
            T_pred_ic = model(torch.cat([t_ic, x_ic], dim=1))
            l_ic = nn.functional.mse_loss(T_pred_ic, T_ic)

            # BC
            t_bc, x_bc, T_bc = sample_bc_points(args.n_bc, device)
            T_pred_bc = model(torch.cat([t_bc, x_bc], dim=1))
            l_bc = nn.functional.mse_loss(T_pred_bc, T_bc)

            loss = 10 * l_pde + 100 * l_ic + 100 * l_bc

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if rank == 0:
            history["pde"].append(l_pde.item())
            history["ic"].append(l_ic.item())
            history["bc"].append(l_bc.item())
            history["total"].append(loss.item())

            if step % max(1, args.steps // 20) == 0 or step == 1:
                print(f"[Step {step:5d}/{args.steps}]  loss={loss.item():.4e}  "
                      f"pde={l_pde.item():.4e}  ic={l_ic.item():.4e}  bc={l_bc.item():.4e}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(
                    out_dir,
                    models=model.module if use_ddp else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=step,
                )

    # ── Visualization (rank 0 only) ──────────────────────
    if rank == 0:
        eval_model = model.module if use_ddp else model
        eval_model.eval()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss curves
        ax = axes[0]
        for key, color in [("pde", "blue"), ("ic", "orange"), ("bc", "green")]:
            ax.semilogy(history[key], alpha=0.7, color=color, label=key)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training losses (GPU)")
        ax.legend()

        # Temperature comparison
        ax = axes[1]
        x_test = np.linspace(0, L, 100)
        for t_val, color in [(0.1, "blue"), (0.5, "green"), (0.9, "red")]:
            with torch.no_grad():
                inp = torch.tensor(
                    np.column_stack([np.full_like(x_test, t_val), x_test]),
                    dtype=torch.float32, device=device)
                T_pred = eval_model(inp).cpu().numpy().flatten()
            T_exact = analytical_solution(t_val, x_test)
            ax.plot(x_test, T_exact, "--", color=color, alpha=0.5)
            ax.plot(x_test, T_pred, "-", color=color, label=f"t={t_val}")
        ax.set_xlabel("x")
        ax.set_ylabel("T")
        ax.set_title("PINN vs Analytical")
        ax.legend()

        # L2 error over time
        ax = axes[2]
        t_vals = np.linspace(0.01, T_MAX, 50)
        errors = []
        for t_val in t_vals:
            with torch.no_grad():
                inp = torch.tensor(
                    np.column_stack([np.full(100, t_val), np.linspace(0, L, 100)]),
                    dtype=torch.float32, device=device)
                pred = eval_model(inp).cpu().numpy().flatten()
            exact = analytical_solution(t_val, np.linspace(0, L, 100))
            errors.append(np.sqrt(np.mean((pred - exact)**2)))
        ax.plot(t_vals, errors)
        ax.set_xlabel("t")
        ax.set_ylabel("L2 error")
        ax.set_title("Error vs time")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "heat1d_gpu_results.png"), dpi=150)
        print(f"\nBest loss: {best_loss:.4e}")
        print(f"Results saved to {out_dir}/")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_pde", type=int, default=4096)
    parser.add_argument("--n_ic", type=int, default=512)
    parser.add_argument("--n_bc", type=int, default=512)
    parser.add_argument("--ddp", action="store_true", help="Enable DDP multi-GPU")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
