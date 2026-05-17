#!/usr/bin/env python3
"""
Chapter 4 (GPU): FNO for Darcy Flow — Full-Scale GPU Training
================================================================
PhysicsNeMo FNO with:
  - AMP (mixed precision)
  - Multi-GPU DDP
  - coord_features enabled (GPU has memory)
  - Larger model + full dataset
  - PhysicsNeMo checkpoint/logging utilities

Usage:
    python train_fno_gpu.py --epochs 100
    torchrun --nproc_per_node=4 train_fno_gpu.py --epochs 100 --ddp
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
from physicsnemo.utils import save_checkpoint, load_checkpoint

sys.path.insert(0, os.path.dirname(__file__))
from dataset import generate_darcy_data


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


def load_or_generate_data(data_dir, n_samples=5000, resolution=64):
    """Load or generate Darcy data at full scale."""
    data_path = os.path.join(data_dir, "darcy_data.pt")
    if os.path.exists(data_path):
        return torch.load(data_path, weights_only=False)
    print(f"Generating {n_samples} Darcy samples at {resolution}x{resolution}...")
    data = generate_darcy_data(n_samples=n_samples, resolution=resolution, seed=42)
    os.makedirs(data_dir, exist_ok=True)
    torch.save(data, data_path)
    return data


def train(args):
    rank, world_size, use_ddp = setup_ddp() if args.ddp else (0, 1, False)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[FNO GPU] Device: {device}, World: {world_size}")

    # ── Data ──────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = load_or_generate_data(
        os.path.join(script_dir, "data"),
        n_samples=args.n_samples,
        resolution=args.resolution)

    K = data.get("K", data.get("a"))
    U = data.get("U", data.get("u"))
    n_total = len(K)
    n_train = int(n_total * 0.8)

    x_train, y_train = K[:n_train].to(device), U[:n_train].to(device)
    x_test, y_test = K[n_train:].to(device), U[n_train:].to(device)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    sampler = DistributedSampler(train_ds) if use_ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler, drop_last=True,
        pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    if rank == 0:
        print(f"Data: train={n_train}, test={n_total - n_train}, "
              f"resolution={args.resolution}")

    # ── Model ─────────────────────────────────────────────
    model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layers=2,
        decoder_layer_size=128,
        decoder_activation_fn="silu",
        dimension=2,
        latent_channels=args.latent_channels,
        num_fno_layers=args.n_layers,
        num_fno_modes=args.n_modes,
        padding=8,
        padding_type="constant",
        activation_fn="gelu",
        coord_features=True,  # GPU has enough memory
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank])

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"PhysicsNeMo FNO: {n_params:,} params")
        print(f"  latent={args.latent_channels}, layers={args.n_layers}, "
              f"modes={args.n_modes}, coord_features=True")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
        for xb, yb in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item() * len(xb)
            cnt += len(xb)
        t_loss /= max(cnt, 1)
        scheduler.step()

        # Validation
        model.eval()
        v_loss, v_cnt = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    v_loss += criterion(model(xb), yb).item() * len(xb)
                v_cnt += len(xb)
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
                      f"train={t_loss:.6f}  test={v_loss:.6f}  "
                      f"best={best_test:.6f}")

    # ── Results ───────────────────────────────────────────
    if rank == 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        ax.semilogy(history["train"], label="Train")
        ax.semilogy(history["test"], label="Test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("FNO Training (GPU)")
        ax.legend()

        eval_model = model.module if use_ddp else model
        eval_model.eval()
        idx = 0
        with torch.no_grad():
            pred = eval_model(x_test[idx:idx+1]).cpu()

        ax = axes[1]
        im = ax.imshow(y_test[idx, 0].cpu().numpy(), cmap="viridis")
        plt.colorbar(im, ax=ax)
        ax.set_title("Ground Truth")

        ax = axes[2]
        im = ax.imshow(pred[0, 0].numpy(), cmap="viridis")
        plt.colorbar(im, ax=ax)
        ax.set_title("FNO Prediction")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fno_gpu_results.png"), dpi=150)
        print(f"\nBest test MSE: {best_test:.6f}")
        print(f"Saved to {out_dir}/")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_modes", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
