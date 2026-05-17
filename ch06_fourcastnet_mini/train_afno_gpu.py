#!/usr/bin/env python3
"""
Chapter 6 (GPU): FourCastNet AFNO — Full-Scale GPU Training with DDP
=======================================================================
PhysicsNeMo AFNO with:
  - Multi-GPU DDP
  - AMP mixed precision
  - Autoregressive rollout evaluation
  - PhysicsNeMo checkpoint utilities
  - Configurable for ERA5 or toy data

Usage:
    # Single GPU
    python train_afno_gpu.py --epochs 50 --embed_dim 256 --depth 8

    # Multi-GPU
    torchrun --nproc_per_node=4 train_afno_gpu.py --epochs 50 --ddp

    # Full ERA5 scale (if data available)
    torchrun --nproc_per_node=8 train_afno_gpu.py --epochs 100 --embed_dim 768 --depth 12 --patch_size 4
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.models.afno import AFNO
from physicsnemo.utils import save_checkpoint

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_weather_dataset


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


def autoregressive_rollout(model, x0, steps, device):
    """Roll out model autoregressively for `steps` steps."""
    preds = [x0]
    x = x0
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            x = model(x)
            preds.append(x)
    return torch.stack(preds, dim=1)  # (B, steps+1, C, H, W)


def train(args):
    rank, world_size, use_ddp = setup_ddp() if args.ddp else (0, 1, False)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[AFNO GPU] Device: {device}, World: {world_size}")

    # ── Data ──────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    train_ds, test_ds = load_weather_dataset(data_dir, lead_time=args.lead_time)

    sampler = DistributedSampler(train_ds) if use_ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler, drop_last=True,
        num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        num_workers=2, pin_memory=True)

    n_vars = train_ds[0][0].shape[0]
    H, W = train_ds[0][0].shape[1], train_ds[0][0].shape[2]

    if rank == 0:
        print(f"Data: train={len(train_ds)}, test={len(test_ds)}, "
              f"vars={n_vars}, grid=({H},{W})")

    # ── Model ─────────────────────────────────────────────
    model = AFNO(
        inp_shape=[H, W],
        in_channels=n_vars,
        out_channels=n_vars,
        patch_size=[args.patch_size, args.patch_size],
        embed_dim=args.embed_dim,
        depth=args.depth,
        mlp_ratio=4.0,
        drop_rate=args.drop_rate,
        num_blocks=args.num_blocks,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"PhysicsNeMo AFNO: {n_params:,} params")
        print(f"  embed_dim={args.embed_dim}, depth={args.depth}, "
              f"patch_size={args.patch_size}, num_blocks={args.num_blocks}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion = nn.MSELoss()

    history = {"train": [], "test": []}
    out_dir = os.path.join(script_dir, "outputs_gpu")
    os.makedirs(out_dir, exist_ok=True)
    best_test = float("inf")

    for epoch in range(1, args.epochs + 1):
        if use_ddp:
            sampler.set_epoch(epoch)

        model.train()
        t_loss, cnt = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item() * len(x_batch)
            cnt += len(x_batch)
        t_loss /= max(cnt, 1)
        scheduler.step()

        model.eval()
        v_loss, v_cnt = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    v_loss += criterion(model(x_batch), y_batch).item() * len(x_batch)
                v_cnt += len(x_batch)
        v_loss /= max(v_cnt, 1)

        if rank == 0:
            history["train"].append(t_loss)
            history["test"].append(v_loss)

            if v_loss < best_test:
                best_test = v_loss
                save_checkpoint(out_dir, models=model.module if use_ddp else model,
                                optimizer=optimizer, scheduler=scheduler, epoch=epoch)

            if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
                print(f"[Epoch {epoch:3d}/{args.epochs}] "
                      f"train={t_loss:.6f} test={v_loss:.6f} "
                      f"best={best_test:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")

    # ── Autoregressive rollout evaluation ─────────────────
    if rank == 0:
        eval_model = model.module if use_ddp else model
        eval_model.eval()

        # Take first test sample and rollout
        x0 = test_ds[0][0].unsqueeze(0).to(device)
        rollout_steps = min(args.rollout_steps, len(test_ds) - 1)
        rollout = autoregressive_rollout(eval_model, x0, rollout_steps, device)

        # Compute per-step RMSE
        rmses = []
        for step in range(1, rollout_steps + 1):
            target = test_ds[step][0].unsqueeze(0).to(device)
            pred = rollout[:, step]
            rmse = torch.sqrt(nn.functional.mse_loss(pred, target)).item()
            rmses.append(rmse)

        # ── Plots ────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        ax.semilogy(history["train"], label="Train")
        ax.semilogy(history["test"], label="Test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("AFNO Training (GPU)")
        ax.legend()

        ax = axes[1]
        ax.plot(range(1, rollout_steps + 1), rmses, "o-")
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("RMSE")
        ax.set_title("Autoregressive Error Growth")

        ax = axes[2]
        # Show first variable at final rollout step
        pred_final = rollout[0, -1, 0].cpu().numpy()
        ax.imshow(pred_final, cmap="RdBu_r")
        ax.set_title(f"Prediction at step {rollout_steps}")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "afno_gpu_results.png"), dpi=150)
        print(f"\nBest test MSE: {best_test:.6f}")
        print(f"Rollout RMSE at step {rollout_steps}: {rmses[-1]:.4f}")
        print(f"Saved to {out_dir}/")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lead_time", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
