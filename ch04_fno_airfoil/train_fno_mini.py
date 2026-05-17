"""
Chapter 4: FNO Training Script
================================
Train FNO2D on synthetic Darcy flow data.
Generates data on-the-fly if not found.

Usage:
    python train_fno_mini.py
    python train_fno_mini.py --epochs 50 --width 64
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try Hydra
try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

sys.path.insert(0, os.path.dirname(__file__))
from fno_model import FNO2D
from dataset import generate_darcy_data_fast


def get_data(cfg: dict):
    """Load or generate Darcy data."""
    data_path = os.path.join(cfg.get("data_dir", "data"), "darcy_data.pt")
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = torch.load(data_path, weights_only=False)
    else:
        print("Generating Darcy data...")
        n = cfg.get("n_samples", 200)
        res = cfg.get("resolution", 64)
        data = generate_darcy_data_fast(n, res)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save(data, data_path)
        print(f"Saved to {data_path}")

    a, u = data['a'], data['u']
    # Train/test split (80/20)
    n_train = int(0.8 * len(a))
    train_ds = TensorDataset(a[:n_train], u[:n_train])
    test_ds = TensorDataset(a[n_train:], u[n_train:])
    return train_ds, test_ds


def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds, test_ds = get_data(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 16),
                              shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 16))

    model = FNO2D(
        in_channels=1, out_channels=1,
        width=cfg.get("width", 32),
        modes_x=cfg.get("modes_x", 12),
        modes_y=cfg.get("modes_y", 12),
        n_layers=cfg.get("n_layers", 4),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.get("epochs", 30) // 3, gamma=0.5)
    criterion = nn.MSELoss()

    epochs = cfg.get("epochs", 30)
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for a_batch, u_batch in train_loader:
            a_batch, u_batch = a_batch.to(device), u_batch.to(device)
            optimizer.zero_grad()
            u_pred = model(a_batch)
            loss = criterion(u_pred, u_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(a_batch)
        train_loss /= len(train_ds)
        scheduler.step()

        # Test
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for a_batch, u_batch in test_loader:
                a_batch, u_batch = a_batch.to(device), u_batch.to(device)
                u_pred = model(a_batch)
                test_loss += criterion(u_pred, u_batch).item() * len(a_batch)
        test_loss /= len(test_ds)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{epochs}] "
                  f"train_loss={train_loss:.6f}  test_loss={test_loss:.6f}")

    # Save
    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "fno_darcy.pt")
    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "history": history,
    }, ckpt_path)
    print(f"\nCheckpoint: {ckpt_path}")

    # Quick loss plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history["train_loss"], label="Train")
    ax.semilogy(history["test_loss"], label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("FNO Training on Darcy Flow")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fno_loss.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    return model, history


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def get_defaults():
    return {
        "width": 32, "modes_x": 12, "modes_y": 12, "n_layers": 4,
        "epochs": 30, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4,
        "n_samples": 200, "resolution": 64,
        "data_dir": "data", "output_dir": "outputs",
    }

if HAS_HYDRA:
    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main_hydra(cfg: DictConfig):
        flat = get_defaults()
        flat.update(OmegaConf.to_container(cfg, resolve=True))
        train(flat)

def main_argparse():
    parser = argparse.ArgumentParser()
    for k, v in get_defaults().items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    train(vars(args))

if __name__ == "__main__":
    if HAS_HYDRA:
        main_hydra()
    else:
        print("[INFO] hydra-core not found, using argparse.")
        main_argparse()
