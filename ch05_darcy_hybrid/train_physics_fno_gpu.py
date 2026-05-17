#!/usr/bin/env python3
"""
Chapter 5 (GPU): Physics-Informed FNO — Full GPU Training
============================================================
PhysicsNeMo FNO with hybrid data+physics loss on GPU.
Demonstrates the key insight: physics loss enables generalization
from small data (50-100 samples vs 5000).

Usage:
    python train_physics_fno_gpu.py --epochs 100 --n_train 100
    torchrun --nproc_per_node=4 train_physics_fno_gpu.py --ddp
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.models.fno import FNO
from physicsnemo.utils import save_checkpoint

sys.path.insert(0, os.path.dirname(__file__))
from darcy_residual import darcy_residual_simple


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        return rank, dist.get_world_size(), True
    return 0, 1, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    rank, world_size, use_ddp = setup_ddp() if args.ddp else (0, 1, False)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[PI-FNO GPU] Device: {device}, World: {world_size}")

    # ── Data ──────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "darcy_data.pt")
    if not os.path.exists(data_path):
        alt = os.path.join(os.path.dirname(script_dir), "ch04_fno_airfoil", "data", "darcy_data.pt")
        if os.path.exists(alt):
            data_path = alt
        else:
            sys.path.insert(0, os.path.join(os.path.dirname(script_dir), "ch04_fno_airfoil"))
            from dataset import generate_darcy_data
            data = generate_darcy_data(n_samples=5000, resolution=64)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            torch.save(data, data_path)

    data = torch.load(data_path, weights_only=False)
    K = data.get("K", data.get("a")).to(device)
    U = data.get("U", data.get("u")).to(device)
    dx = 1.0 / K.shape[-1]

    n_train = min(args.n_train, len(K) - 100)
    n_test = min(200, len(K) - n_train)
    x_train, y_train = K[:n_train], U[:n_train]
    x_test, y_test = K[n_train:n_train+n_test], U[n_train:n_train+n_test]

    train_ds = TensorDataset(x_train, y_train)
    sampler = DistributedSampler(train_ds) if use_ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(sampler is None), sampler=sampler, drop_last=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size)

    if rank == 0:
        print(f"Data: train={n_train} (small!), test={n_test}")
        print(f"lambda_data={args.lambda_data}, lambda_physics={args.lambda_physics}")

    # ── Model ─────────────────────────────────────────────
    model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layers=2,
        decoder_layer_size=128,
        dimension=2,
        latent_channels=args.latent_channels,
        num_fno_layers=args.n_layers,
        num_fno_modes=args.n_modes,
        padding=8,
        coord_features=True,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank])

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"PhysicsNeMo FNO: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"total": [], "data": [], "physics": [], "test": []}
    out_dir = os.path.join(script_dir, "outputs_gpu")
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if use_ddp:
            sampler.set_epoch(epoch)

        model.train()
        ep_total, ep_data, ep_phys, cnt = 0., 0., 0., 0
        for kb, ub in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                u_pred = model(kb)
                l_data = nn.functional.mse_loss(u_pred, ub)
                # Physics: Darcy residual
                l_phys = darcy_residual_simple(u_pred, f=1.0, dx=dx).mean()
                loss = args.lambda_data * l_data + args.lambda_physics * l_phys

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = len(kb)
            ep_total += loss.item() * bs
            ep_data += l_data.item() * bs
            ep_phys += l_phys.item() * bs
            cnt += bs

        ep_total /= cnt
        ep_data /= cnt
        ep_phys /= cnt
        scheduler.step()

        model.eval()
        v_loss, v_cnt = 0., 0
        with torch.no_grad():
            for kb, ub in test_loader:
                v_loss += nn.functional.mse_loss(model(kb), ub).item() * len(kb)
                v_cnt += len(kb)
        v_loss /= max(v_cnt, 1)

        if rank == 0:
            for k, v in [("total", ep_total), ("data", ep_data),
                          ("physics", ep_phys), ("test", v_loss)]:
                history[k].append(v)

            if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
                print(f"[Epoch {epoch:3d}/{args.epochs}] "
                      f"total={ep_total:.6f} data={ep_data:.6f} "
                      f"phys={ep_phys:.4f} test={v_loss:.6f}")

    if rank == 0:
        save_checkpoint(out_dir, models=model.module if use_ddp else model,
                        optimizer=optimizer, epoch=args.epochs)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.semilogy(history["data"], label="Data loss")
        ax.semilogy([abs(x) for x in history["physics"]], label="|Physics loss|")
        ax.semilogy(history["test"], label="Test loss", linestyle="--")
        ax.legend()
        ax.set_title(f"PI-FNO (n_train={n_train})")

        ax = axes[1]
        eval_model = model.module if use_ddp else model
        with torch.no_grad():
            pred = eval_model(x_test[:1]).cpu()
        err = (pred[0, 0] - y_test[0, 0].cpu()).abs()
        im = ax.imshow(err.numpy(), cmap="hot")
        plt.colorbar(im, ax=ax)
        ax.set_title("Absolute error")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "physics_fno_gpu.png"), dpi=150)
        print(f"\nSaved to {out_dir}/")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--lambda_data", type=float, default=1.0)
    parser.add_argument("--lambda_physics", type=float, default=0.1)
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_modes", type=int, default=16)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
