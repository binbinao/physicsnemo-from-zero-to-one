"""
Chapter 5: Physics-Informed FNO (Hybrid)
==========================================
Train FNO with BOTH data loss and Darcy PDE residual loss.
Demonstrates how physics regularization improves generalization,
especially with limited training data.

Loss = λ_data * MSE(u_pred, u_true) + λ_physics * MSE(PDE_residual, 0)

Usage:
    python train_physics_fno.py
    python train_physics_fno.py --lambda_physics 0.1 --n_train 50
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

sys.path.insert(0, os.path.dirname(__file__))
from require_ch04 import require_ch04

require_ch04()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch04_fno_airfoil"))
from fno_model import FNO2D
from darcy_residual import darcy_residual

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False


def generate_data(n_samples=200, resolution=64):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch04_fno_airfoil"))
    from dataset import generate_darcy_data_fast
    data_path = os.path.join("data", "darcy_data.pt")
    if os.path.exists(data_path):
        return torch.load(data_path, weights_only=False)
    os.makedirs("data", exist_ok=True)
    data = generate_darcy_data_fast(n_samples, resolution)
    torch.save(data, data_path)
    return data


def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Physics-informed FNO] Device: {device}")

    data = generate_data(cfg.get("n_samples", 200), cfg.get("resolution", 64))
    a, u = data['a'], data['u']

    n_train = cfg.get("n_train", 100)
    n_test = min(50, len(a) - n_train)

    train_ds = TensorDataset(a[:n_train], u[:n_train])
    test_ds = TensorDataset(a[n_train:n_train + n_test], u[n_train:n_train + n_test])
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 16), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 16))

    model = FNO2D(in_channels=1, out_channels=1,
                  width=cfg.get("width", 32),
                  modes_x=cfg.get("modes", 12),
                  modes_y=cfg.get("modes", 12),
                  n_layers=cfg.get("n_layers", 4)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    epochs = cfg.get("epochs", 50)
    lambda_data = cfg.get("lambda_data", 1.0)
    lambda_physics = cfg.get("lambda_physics", 0.1)

    print(f"lambda_data={lambda_data}, lambda_physics={lambda_physics}, "
          f"n_train={n_train}")

    history = {"train": [], "test": [], "data_loss": [], "physics_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_data, epoch_phys = 0.0, 0.0, 0.0
        count = 0

        for ab, ub in train_loader:
            ab, ub = ab.to(device), ub.to(device)
            optimizer.zero_grad()

            u_pred = model(ab)

            # Data loss
            data_loss = nn.functional.mse_loss(u_pred, ub)

            # Physics loss: Darcy residual should be zero
            res = darcy_residual(ab, u_pred, f=1.0)
            physics_loss = (res ** 2).mean()

            loss = lambda_data * data_loss + lambda_physics * physics_loss
            loss.backward()
            optimizer.step()

            bs = len(ab)
            epoch_loss += loss.item() * bs
            epoch_data += data_loss.item() * bs
            epoch_phys += physics_loss.item() * bs
            count += bs

        epoch_loss /= count
        epoch_data /= count
        epoch_phys /= count

        # Test
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for ab, ub in test_loader:
                ab, ub = ab.to(device), ub.to(device)
                v_loss += nn.functional.mse_loss(model(ab), ub).item() * len(ab)
        v_loss /= n_test

        history["train"].append(epoch_loss)
        history["test"].append(v_loss)
        history["data_loss"].append(epoch_data)
        history["physics_loss"].append(epoch_phys)

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{epochs}] total={epoch_loss:.6f} "
                  f"data={epoch_data:.6f} phys={epoch_phys:.6f} "
                  f"test={v_loss:.6f}")

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "history": history},
               os.path.join(out_dir, "fno_physics.pt"))
    print(f"Saved {out_dir}/fno_physics.pt")
    return history


def get_defaults():
    return {
        "width": 32, "modes": 12, "n_layers": 4,
        "epochs": 50, "batch_size": 16, "lr": 1e-3,
        "n_samples": 200, "resolution": 64, "n_train": 100,
        "lambda_data": 1.0, "lambda_physics": 0.1,
        "output_dir": "outputs",
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
    train(vars(parser.parse_args()))

if __name__ == "__main__":
    if HAS_HYDRA:
        main_hydra()
    else:
        main_argparse()
