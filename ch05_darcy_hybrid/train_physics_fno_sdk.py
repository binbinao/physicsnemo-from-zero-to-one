#!/usr/bin/env python3
"""
Chapter 5 (SDK): Physics-Informed FNO using PhysicsNeMo FNO
==============================================================
Uses PhysicsNeMo FNO as backbone + Darcy PDE residual loss.
Same hybrid data+physics approach, same data — PhysicsNeMo model.

Usage:
    python train_physics_fno_sdk.py --epochs 30 --n_train 100
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
from darcy_residual import darcy_residual_simple as darcy_pde_residual


def load_data(data_dir):
    """Load Darcy data (generate if missing)."""
    data_path = os.path.join(data_dir, "darcy_data.pt")
    if os.path.exists(data_path):
        return torch.load(data_path, weights_only=False)
    # Fallback: try ch04's data directory
    alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "ch04_fno_airfoil", "data", "darcy_data.pt")
    if os.path.exists(alt_path):
        data = torch.load(alt_path, weights_only=False)
        os.makedirs(data_dir, exist_ok=True)
        torch.save(data, data_path)
        return data
    # Generate
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "ch04_fno_airfoil"))
    from dataset import generate_darcy_data
    data = generate_darcy_data(n_samples=1200, resolution=64, seed=42)
    os.makedirs(data_dir, exist_ok=True)
    torch.save(data, data_path)
    return data


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Physics-Informed FNO SDK] Device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = load_data(os.path.join(script_dir, "data"))

    # Support both key naming conventions
    K = data.get("K", data.get("a"))  # (N, 1, H, W)
    U = data.get("U", data.get("u"))  # (N, 1, H, W)
    dx = data.get("dx", 1.0 / K.shape[-1])

    n_total = len(K)
    n_train = min(args.n_train, n_total - 20)  # keep at least 20 for test
    x_train, y_train = K[:n_train], U[:n_train]
    x_test, y_test = K[n_train:n_train+20], U[n_train:n_train+20]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size)

    lambda_d = args.lambda_data
    lambda_p = args.lambda_physics
    print(f"lambda_data={lambda_d}, lambda_physics={lambda_p}, n_train={n_train}")

    # ── PhysicsNeMo FNO ──────────────────────────────────
    use_coords = torch.cuda.is_available()
    model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layers=2,
        decoder_layer_size=64,
        dimension=2,
        latent_channels=args.latent_channels,
        num_fno_layers=args.n_layers,
        num_fno_modes=args.n_modes,
        padding=8,
        coord_features=use_coords,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PhysicsNeMo FNO: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"total": [], "data": [], "physics": [], "test": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_total, ep_data, ep_phys, cnt = 0., 0., 0., 0
        for kb, ub in train_loader:
            kb, ub = kb.to(device), ub.to(device)
            kb.requires_grad_(True)

            u_pred = model(kb)

            # Data loss
            l_data = nn.functional.mse_loss(u_pred, ub)

            # Physics loss: Darcy PDE residual (simplified: -Δu = f)
            l_phys = darcy_pde_residual(u_pred, f=1.0, dx=dx).mean()

            loss = lambda_d * l_data + lambda_p * l_phys

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

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
                kb, ub = kb.to(device), ub.to(device)
                v_loss += nn.functional.mse_loss(model(kb), ub).item() * len(kb)
                v_cnt += len(kb)
        v_loss /= v_cnt

        for k, v in [("total", ep_total), ("data", ep_data),
                      ("physics", ep_phys), ("test", v_loss)]:
            history[k].append(v)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{args.epochs}] "
                  f"total={ep_total:.6f} data={ep_data:.6f} "
                  f"phys={ep_phys:.6f} test={v_loss:.6f}")

    # ── Save ──────────────────────────────────────────────
    out_dir = os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "fno_physics_sdk.pt")
    torch.save({
        "model_state": model.state_dict(),
        "history": history,
        "args": vars(args),
    }, ckpt_path)
    print(f"Saved {ckpt_path}")

    # ── Compare ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.semilogy(history["data"], label="Data loss")
    ax.semilogy(history["physics"], label="Physics loss")
    ax.semilogy(history["test"], label="Test loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Physics-Informed FNO (SDK)")
    ax.legend()

    ax = axes[1]
    model.eval()
    with torch.no_grad():
        pred = model(x_test[:1].to(device)).cpu()
    err = (pred[0, 0] - y_test[0, 0]).abs()
    im = ax.imshow(err.numpy(), cmap="hot")
    plt.colorbar(im, ax=ax)
    ax.set_title("Absolute error (sample 0)")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "physics_fno_sdk.png")
    plt.savefig(fig_path, dpi=120)
    print(f"Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--lambda_data", type=float, default=1.0)
    parser.add_argument("--lambda_physics", type=float, default=0.1)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_modes", type=int, default=16)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
