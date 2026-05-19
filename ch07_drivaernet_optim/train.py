#!/usr/bin/env python3
"""
Chapter 7: Train Cd surrogate model
=====================================
  python train.py                         # uses defaults
  python train.py --epochs 200 --lr 1e-3  # override

Reads data from data/car_aero_data.pt (run data/generate_toy_cars.py first).
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Import model & data generator ─────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from models.cd_mlp import CdMLP
from data.generate_toy_cars import generate_car_data


# Resolve paths relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_or_generate_data(data_path=None, n_samples=2000):
    if data_path is None:
        data_path = os.path.join(_SCRIPT_DIR, "data", "car_aero_data.pt")
    """Load pre-generated data or generate on the fly."""
    if os.path.exists(data_path):
        return torch.load(data_path, weights_only=False)
    print(f"Data not found at {data_path}, generating {n_samples} samples...")
    data = generate_car_data(n_samples)
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    torch.save(data, data_path)
    return data


def train(args):
    # ── Data ──────────────────────────────────────────────
    data = load_or_generate_data(n_samples=args.n_samples)
    params = data["params"]  # (N, 7)
    cd = data["cd"]          # (N,)

    n = len(cd)
    n_train = int(n * 0.8)
    n_val = n - n_train
    perm = torch.randperm(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    x_train, y_train = params[train_idx], cd[train_idx]
    x_val, y_val = params[val_idx], cd[val_idx]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True
    )

    # ── Model ─────────────────────────────────────────────
    model = CdMLP(
        in_features=7,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        dropout=args.dropout,
    )
    # Set normalization stats
    model.set_normalization(x_train.mean(0), x_train.std(0))
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────
    best_val = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb).squeeze(-1)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= n_train
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val).squeeze(-1)
            val_mse = nn.functional.mse_loss(val_pred, y_val).item()
            val_mae = (val_pred - y_val).abs().mean().item()
            val_rel = val_mae / y_val.abs().mean().item()

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  train_mse {epoch_loss:.6f}  "
                  f"val_mse {val_mse:.6f}  val_mae {val_mae:.4f}  val_rel {val_rel:.2%}")

        if val_mse < best_val:
            best_val = val_mse
            try:
                git_commit = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                git_commit = "unknown"
            data_path = os.path.join(_SCRIPT_DIR, "data", "car_aero_data.pt")
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "input_mean": model.input_mean,
                "input_std": model.input_std,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "param_names": data["param_names"],
                "param_ranges": data["param_ranges"],
                "args": vars(args),
                "meta": {
                    "data_tier": "toy_synthetic",
                    "data_path": data_path,
                    "git_commit": git_commit,
                    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                    "n_samples": int(n),
                    "note": "Not DrivAerNet production data unless you replace car_aero_data.pt",
                },
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

    print(f"\nBest val MSE: {best_val:.6f}")
    print(f"Checkpoint saved: {args.out_dir}/best.pt")

    # ── Quick evaluation ──────────────────────────────────
    model.eval()
    with torch.no_grad():
        all_pred = model(params).squeeze(-1)
        all_mae = (all_pred - cd).abs().mean().item()
    print(f"Overall MAE: {all_mae:.4f}  (Cd range: {cd.min():.3f}-{cd.max():.3f})")


def main():
    parser = argparse.ArgumentParser(description="Train Cd surrogate model")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default=os.path.join(_SCRIPT_DIR, "outputs"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
