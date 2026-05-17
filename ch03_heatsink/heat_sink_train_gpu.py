#!/usr/bin/env python3
"""
Chapter 3 (GPU): Heat Sink PINN — Multi-GPU with AMP
=======================================================
Full-scale 2D heat sink with larger network and more points.

Usage:
    python heat_sink_train_gpu.py --steps 5000
    torchrun --nproc_per_node=4 heat_sink_train_gpu.py --steps 5000 --ddp
"""

import os
import sys
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
from physicsnemo.utils import save_checkpoint

sys.path.insert(0, os.path.dirname(__file__))
from heat_sink_geometry import HeatSinkGeometry


def setup_ddp():
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


def laplacian_2d(T, pts):
    """Compute 2D Laplacian via autograd."""
    dT = torch.autograd.grad(T, pts, torch.ones_like(T), create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT[:, 0:1], pts, torch.ones_like(T),
                                   create_graph=True)[0][:, 0:1]
    d2T_dy2 = torch.autograd.grad(dT[:, 1:2], pts, torch.ones_like(T),
                                   create_graph=True)[0][:, 1:2]
    return d2T_dx2 + d2T_dy2


# ── Geometry constants ────────────────────────────────────
LX = 0.1
LY_BASE = 0.01
LY_FIN = 0.05
N_FINS = 5
FIN_WIDTH = 0.005
FIN_GAP = (LX - N_FINS * FIN_WIDTH) / (N_FINS + 1)


def sample_interior(n, device):
    """Sample interior points of heat sink."""
    pts = []
    while len(pts) < n:
        x = np.random.uniform(0, LX, n * 3)
        y = np.random.uniform(0, LY_BASE + LY_FIN, n * 3)
        for xi, yi in zip(x, y):
            in_base = (0 <= xi <= LX) and (0 <= yi <= LY_BASE)
            in_fin = False
            if yi <= LY_BASE + LY_FIN:
                for i in range(N_FINS):
                    fl = FIN_GAP + i * (FIN_WIDTH + FIN_GAP)
                    if fl <= xi <= fl + FIN_WIDTH:
                        in_fin = True
                        break
            if in_base or in_fin:
                pts.append([xi, yi])
                if len(pts) >= n:
                    break
    return torch.tensor(pts, dtype=torch.float32, device=device)


def sample_bottom(n, device):
    x = torch.rand(n, device=device) * LX
    y = torch.zeros(n, device=device)
    return torch.stack([x, y], dim=1)


def sample_robin(n, device):
    """Sample fin surfaces with normals."""
    pts = []
    for _ in range(n * 5):
        i = np.random.randint(N_FINS)
        fl = FIN_GAP + i * (FIN_WIDTH + FIN_GAP)
        choice = np.random.choice(["top", "left", "right"])
        if choice == "top":
            x, y = np.random.uniform(fl, fl + FIN_WIDTH), LY_BASE + LY_FIN
            nx, ny = 0.0, 1.0
        elif choice == "left":
            x, y = fl, np.random.uniform(LY_BASE, LY_BASE + LY_FIN)
            nx, ny = -1.0, 0.0
        else:
            x, y = fl + FIN_WIDTH, np.random.uniform(LY_BASE, LY_BASE + LY_FIN)
            nx, ny = 1.0, 0.0
        pts.append([x, y, nx, ny])
        if len(pts) >= n:
            break
    t = torch.tensor(pts[:n], dtype=torch.float32, device=device)
    return t[:, :2], t[:, 2:]


def train(args):
    rank, world_size, use_ddp = setup_ddp() if args.ddp else (0, 1, False)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[Heat Sink GPU] Device: {device}, World: {world_size}")

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
        print(f"Model: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    k = args.k
    Q = args.Q
    h_conv = args.h_conv
    T_inf = args.T_inf
    T_src = args.T_source

    history = {"total": [], "pde": [], "bot": [], "robin": []}
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_gpu")
    os.makedirs(out_dir, exist_ok=True)

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            # PDE
            pts_int = sample_interior(args.n_interior, device)
            pts_int.requires_grad_(True)
            T_int = model(pts_int)
            lap = laplacian_2d(T_int, pts_int)
            l_pde = ((k * lap + Q) ** 2).mean()

            # Bottom Dirichlet
            pts_bot = sample_bottom(args.n_bottom, device)
            T_bot = model(pts_bot)
            l_bot = ((T_bot - T_src) ** 2).mean()

            # Robin
            pts_rob, normals = sample_robin(args.n_robin, device)
            pts_rob.requires_grad_(True)
            T_rob = model(pts_rob)
            dT_rob = torch.autograd.grad(T_rob, pts_rob, torch.ones_like(T_rob),
                                          create_graph=True)[0]
            dT_dn = (dT_rob * normals).sum(dim=1, keepdim=True)
            l_robin = ((-k * dT_dn - h_conv * (T_rob - T_inf)) ** 2).mean()

            loss = args.w_pde * l_pde + args.w_bottom * l_bot + args.w_robin * l_robin

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if rank == 0:
            for key, val in [("total", loss), ("pde", l_pde),
                              ("bot", l_bot), ("robin", l_robin)]:
                history[key].append(val.item())

            if step % max(1, args.steps // 20) == 0 or step == 1:
                print(f"[{step:5d}/{args.steps}] loss={loss.item():.4e} "
                      f"pde={l_pde.item():.4e} bot={l_bot.item():.4e} "
                      f"robin={l_robin.item():.4e}")

    if rank == 0:
        save_checkpoint(out_dir, models=model.module if use_ddp else model,
                        optimizer=optimizer, epoch=args.steps)
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        for key, color in [("pde", "blue"), ("bot", "orange"), ("robin", "red")]:
            ax.semilogy(history[key], alpha=0.7, color=color, label=key)
        ax.legend()
        ax.set_title("Loss curves (GPU)")

        ax = axes[1]
        eval_model = model.module if use_ddp else model
        eval_model.eval()
        pts_vis = sample_interior(5000, device)
        with torch.no_grad():
            T_vis = eval_model(pts_vis).cpu().numpy()
        pts_np = pts_vis.cpu().numpy()
        sc = ax.scatter(pts_np[:, 0]*1000, pts_np[:, 1]*1000,
                         c=T_vis.flatten(), cmap="hot", s=2)
        plt.colorbar(sc, ax=ax, label="T (K)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        ax.set_title("Temperature field")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "heat_sink_gpu.png"), dpi=150)
        print(f"\nSaved to {out_dir}/")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=float, default=200.0)
    parser.add_argument("--Q", type=float, default=1e5)
    parser.add_argument("--T_source", type=float, default=350.0)
    parser.add_argument("--h_conv", type=float, default=50.0)
    parser.add_argument("--T_inf", type=float, default=300.0)
    parser.add_argument("--n_interior", type=int, default=4096)
    parser.add_argument("--n_bottom", type=int, default=512)
    parser.add_argument("--n_robin", type=int, default=1024)
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bottom", type=float, default=10.0)
    parser.add_argument("--w_robin", type=float, default=1.0)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
