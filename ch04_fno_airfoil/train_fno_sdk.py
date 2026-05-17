#!/usr/bin/env python3
"""
Chapter 4 (SDK): FNO for Darcy Flow using PhysicsNeMo FNO
===========================================================
Swaps our hand-built FNO for physicsnemo.models.fno.FNO.
Same data pipeline, same training logic — new backbone.

Usage:
    python train_fno_sdk.py --epochs 30
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.models.fno import FNO

sys.path.insert(0, os.path.dirname(__file__))
from dataset import generate_darcy_data


def load_or_generate_data(data_dir):
    """Load or generate Darcy flow data."""
    data_path = os.path.join(data_dir, "darcy_data.pt")
    if os.path.exists(data_path):
        return torch.load(data_path, weights_only=False)
    print("Generating Darcy flow data...")
    data = generate_darcy_data(n_samples=1200, resolution=64, seed=42)
    os.makedirs(data_dir, exist_ok=True)
    torch.save(data, data_path)
    return data


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PhysicsNeMo FNO SDK] Device: {device}")

    # ── Data ──────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    data = load_or_generate_data(data_dir)

    # Support both key naming conventions
    K = data.get("K", data.get("a"))   # (N, 1, H, W) permeability
    U = data.get("U", data.get("u"))   # (N, 1, H, W) pressure

    # For CPU environments with limited memory, reduce dataset
    max_train = min(150, len(K) - 50) if device == "cpu" else min(1000, len(K) - 200)
    n_train = max_train
    n_test = min(50, len(K) - n_train)
    x_train, y_train = K[:n_train], U[:n_train]
    x_test, y_test = K[n_train:n_train+n_test], U[n_train:n_train+n_test]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size)

    print(f"Data: train={len(x_train)}, test={len(x_test)}, "
          f"shape={list(x_train.shape)}")

    # ── PhysicsNeMo FNO ──────────────────────────────────
    # The PhysicsNeMo FNO expects input shape (B, in_channels, H, W)
    # On CPU with limited memory, reduce model size
    use_coords = torch.cuda.is_available()  # coord_features doubles memory
    model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layers=2,
        decoder_layer_size=64,
        decoder_activation_fn="silu",
        dimension=2,
        latent_channels=args.latent_channels,
        num_fno_layers=args.n_layers,
        num_fno_modes=args.n_modes,
        padding=8,
        padding_type="constant",
        activation_fn="gelu",
        coord_features=use_coords,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PhysicsNeMo FNO: {n_params:,} parameters")
    print(f"  latent_channels={args.latent_channels}, layers={args.n_layers}, "
          f"modes={args.n_modes}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    history = {"train": [], "test": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, cnt = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(xb)
            cnt += len(xb)
        t_loss /= cnt
        scheduler.step()

        model.eval()
        v_loss, v_cnt = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += criterion(model(xb), yb).item() * len(xb)
                v_cnt += len(xb)
        v_loss /= v_cnt

        history["train"].append(t_loss)
        history["test"].append(v_loss)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{args.epochs}] "
                  f"train={t_loss:.6f}  test={v_loss:.6f}")

    # ── Save ──────────────────────────────────────────────
    out_dir = os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "fno_darcy_sdk.pt")
    torch.save({
        "model_state": model.state_dict(),
        "history": history,
        "args": vars(args),
    }, ckpt_path)
    print(f"\nCheckpoint: {ckpt_path}")

    # ── Plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.semilogy(history["train"], label="train")
    ax.semilogy(history["test"], label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("FNO Training (SDK)")
    ax.legend()

    model.eval()
    idx = 0
    with torch.no_grad():
        sample_pred = model(x_test[idx:idx+1].to(device)).cpu()

    ax = axes[1]
    im = ax.imshow(y_test[idx, 0].numpy(), cmap="viridis")
    ax.set_title("Ground truth")
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(sample_pred[0, 0].numpy(), cmap="viridis")
    ax.set_title("FNO prediction (SDK)")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fno_sdk_results.png")
    plt.savefig(fig_path, dpi=120)
    print(f"Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_modes", type=int, default=16)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
