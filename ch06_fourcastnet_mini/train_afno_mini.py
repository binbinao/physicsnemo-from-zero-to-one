"""
Chapter 6: Train Mini FourCastNet (AFNO)
==========================================
Train the AFNO model on synthetic weather data for next-step prediction.

Usage:
    python train_afno_mini.py
    python train_afno_mini.py --epochs 30 --embed_dim 128
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

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

sys.path.insert(0, os.path.dirname(__file__))
from afno_model import AFNOMini
from dataset import load_weather_dataset


def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AFNO Mini] Device: {device}")

    train_ds, test_ds = load_weather_dataset(
        data_dir=cfg.get("data_dir", "data"),
        lead_time=cfg.get("lead_time", 1))

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8),
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 8))

    n_vars = train_ds[0][0].shape[0]  # auto-detect from data

    model = AFNOMini(
        in_channels=n_vars,
        out_channels=n_vars,
        embed_dim=cfg.get("embed_dim", 64),
        depth=cfg.get("depth", 4),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 5e-4),
                                   weight_decay=cfg.get("weight_decay", 1e-5))
    epochs = cfg.get("epochs", 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train": [], "test": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        count = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(x_batch)
            count += len(x_batch)
        t_loss /= max(count, 1)
        scheduler.step()

        model.eval()
        v_loss = 0.0
        v_count = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                v_loss += criterion(model(x_batch), y_batch).item() * len(x_batch)
                v_count += len(x_batch)
        v_loss /= max(v_count, 1)

        history["train"].append(t_loss)
        history["test"].append(v_loss)

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{epochs}] train={t_loss:.6f} test={v_loss:.6f}")

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "afno_weather.pt")
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "history": history,
        "config": cfg,
    }
    torch.save(ckpt, ckpt_path)
    print(f"\nCheckpoint: {ckpt_path}")

    # Loss plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history["train"], label="Train")
    ax.semilogy(history["test"], label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("AFNO Weather Prediction Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "afno_loss.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

    return model


def get_defaults():
    return {
        "embed_dim": 64, "depth": 4, "mlp_ratio": 4.0,
        "epochs": 30, "batch_size": 8, "lr": 5e-4, "weight_decay": 1e-5,
        "lead_time": 1, "data_dir": "data", "output_dir": "outputs",
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
