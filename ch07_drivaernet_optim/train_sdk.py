#!/usr/bin/env python3
"""
Chapter 7 (SDK): Train Cd Surrogate with PhysicsNeMo FullyConnected
=====================================================================
Uses physicsnemo.models.mlp.FullyConnected for the car aero surrogate.

Usage:
    python train_sdk.py --epochs 200
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.models.mlp import FullyConnected

sys.path.insert(0, os.path.dirname(__file__))
from data.generate_toy_cars import generate_car_data


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_or_generate_data(n_samples=2000):
    data_path = os.path.join(_SCRIPT_DIR, "data", "car_aero_data.pt")
    if os.path.exists(data_path):
        return torch.load(data_path, weights_only=False)
    print(f"Generating {n_samples} car samples...")
    data = generate_car_data(n_samples)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    torch.save(data, data_path)
    return data


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PhysicsNeMo SDK] Device: {device}")

    data = load_or_generate_data(args.n_samples)
    params = data["params"]  # (N, 7)
    cd = data["cd"]          # (N,)

    # Normalize inputs
    x_mean, x_std = params.mean(0), params.std(0)
    params_norm = (params - x_mean) / (x_std + 1e-8)

    n = len(cd)
    n_train = int(n * 0.8)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    x_train, y_train = params_norm[train_idx], cd[train_idx]
    x_val, y_val = params_norm[val_idx], cd[val_idx]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True)

    # ── PhysicsNeMo FullyConnected ────────────────────────
    model = FullyConnected(
        in_features=7,
        out_features=1,
        layer_size=args.hidden,
        num_layers=args.layers,
        activation_fn="silu",
        skip_connections=True,
        weight_norm=True,  # PhysicsNeMo feature
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PhysicsNeMo FullyConnected: {n_params:,} parameters")
    print(f"  hidden={args.hidden}, layers={args.layers}, skip_connections=True, weight_norm=True")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    out_dir = os.path.join(_SCRIPT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    history = {"train": [], "val": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= n_train
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val.to(device)).squeeze(-1)
            val_mse = nn.functional.mse_loss(val_pred, y_val.to(device)).item()
            val_mae = (val_pred - y_val.to(device)).abs().mean().item()
            val_rel = val_mae / y_val.abs().mean().item()

        history["train"].append(epoch_loss)
        history["val"].append(val_mse)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  train_mse {epoch_loss:.6f}  "
                  f"val_mse {val_mse:.6f}  val_rel {val_rel:.2%}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
                "val_mse": val_mse,
                "param_names": data["param_names"],
                "param_ranges": data["param_ranges"],
                "args": vars(args),
            }, os.path.join(out_dir, "best_sdk.pt"))

    print(f"\nBest val MSE: {best_val:.6f}")
    print(f"Checkpoint: {out_dir}/best_sdk.pt")

    # ── Plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history["train"], label="Train")
    ax.semilogy(history["val"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Cd Surrogate (PhysicsNeMo SDK)")
    ax.legend()
    fig_path = os.path.join(out_dir, "train_sdk_loss.png")
    plt.savefig(fig_path, dpi=120)
    print(f"Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=2000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
