#!/usr/bin/env python3
"""
Chapter 6 (SDK): Mini FourCastNet using PhysicsNeMo AFNO
==========================================================
Uses physicsnemo.models.afno.AFNO as backbone for weather prediction.

The PhysicsNeMo AFNO expects:
  - inp_shape: [H, W] spatial dimensions
  - in_channels / out_channels
  - patch_size, embed_dim, depth, etc.

Usage:
    python train_afno_sdk.py --epochs 30
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.models.afno import AFNO

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_weather_dataset


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PhysicsNeMo AFNO SDK] Device: {device}")

    # ── Data ──────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    train_ds, test_ds = load_weather_dataset(
        data_dir=data_dir, lead_time=args.lead_time)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    n_vars = train_ds[0][0].shape[0]
    H, W = train_ds[0][0].shape[1], train_ds[0][0].shape[2]

    # ── PhysicsNeMo AFNO ─────────────────────────────────
    # patch_size=1 for pixel-level (our data is small 32x32)
    model = AFNO(
        inp_shape=[H, W],
        in_channels=n_vars,
        out_channels=n_vars,
        patch_size=[args.patch_size, args.patch_size],
        embed_dim=args.embed_dim,
        depth=args.depth,
        mlp_ratio=4.0,
        drop_rate=0.0,
        num_blocks=args.num_blocks,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PhysicsNeMo AFNO: {n_params:,} parameters")
    print(f"  inp_shape=[{H},{W}], embed_dim={args.embed_dim}, "
          f"depth={args.depth}, patch_size={args.patch_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    history = {"train": [], "test": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, cnt = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(x_batch)
            cnt += len(x_batch)
        t_loss /= max(cnt, 1)
        scheduler.step()

        model.eval()
        v_loss, v_cnt = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                v_loss += criterion(model(x_batch), y_batch).item() * len(x_batch)
                v_cnt += len(x_batch)
        v_loss /= max(v_cnt, 1)

        history["train"].append(t_loss)
        history["test"].append(v_loss)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{args.epochs}] "
                  f"train={t_loss:.6f} test={v_loss:.6f}")

    # ── Save ──────────────────────────────────────────────
    out_dir = os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "afno_weather_sdk.pt")
    torch.save({
        "model": model.state_dict(),
        "config": vars(args),
        "history": history,
    }, ckpt_path)
    print(f"\nCheckpoint: {ckpt_path}")

    # ── Plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history["train"], label="Train")
    ax.semilogy(history["test"], label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("AFNO Weather (PhysicsNeMo SDK)")
    ax.legend()
    fig_path = os.path.join(out_dir, "afno_sdk_loss.png")
    plt.savefig(fig_path, dpi=120)
    print(f"Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lead_time", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
