#!/usr/bin/env python3
"""
Chapter 7 (GPU): Full Aero Surrogate Pipeline — GPU Training + Optuna
========================================================================
Complete end-to-end pipeline:
  1. Train Cd surrogate with PhysicsNeMo FullyConnected (GPU + AMP)
  2. Hyperparameter search with Optuna
  3. Design optimization with best model
  4. Export to ONNX + TensorRT

Usage:
    # Train only
    python train_gpu.py --epochs 500

    # With Optuna HPO
    python train_gpu.py --epochs 500 --hpo --hpo_trials 50

    # Multi-GPU
    torchrun --nproc_per_node=4 train_gpu.py --epochs 500 --ddp
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.models.mlp import FullyConnected
from physicsnemo.utils import save_checkpoint

sys.path.insert(0, os.path.dirname(__file__))
from data.generate_toy_cars import generate_car_data

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def load_data(n_samples=5000):
    data_path = os.path.join(_SCRIPT_DIR, "data", "car_aero_data.pt")
    if os.path.exists(data_path):
        return torch.load(data_path, weights_only=False)
    data = generate_car_data(n_samples)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    torch.save(data, data_path)
    return data


def train_model(args, hidden=None, layers=None, lr=None, trial=None):
    """Train a single model. Returns val_mse."""
    rank, world_size, use_ddp = setup_ddp() if args.ddp else (0, 1, False)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    hidden = hidden or args.hidden
    layers = layers or args.layers
    lr = lr or args.lr

    data = load_data(args.n_samples)
    params = data["params"].to(device)
    cd = data["cd"].to(device)

    x_mean, x_std = params.mean(0), params.std(0)
    params_norm = (params - x_mean) / (x_std + 1e-8)

    n = len(cd)
    n_train = int(n * 0.8)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    x_train, y_train = params_norm[train_idx], cd[train_idx]
    x_val, y_val = params_norm[val_idx], cd[val_idx]

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                              batch_size=args.batch_size, shuffle=True)

    model = FullyConnected(
        in_features=7,
        out_features=1,
        layer_size=hidden,
        num_layers=layers,
        activation_fn="silu",
        skip_connections=True,
        weight_norm=True,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val = float("inf")
    out_dir = os.path.join(_SCRIPT_DIR, "outputs_gpu")
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(xb).squeeze(-1)
                loss = nn.functional.mse_loss(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val).squeeze(-1)
            val_mse = nn.functional.mse_loss(val_pred, y_val).item()

        if val_mse < best_val:
            best_val = val_mse
            if trial is None:  # Only save for final training
                ckpt = {
                    "model_state": (model.module if use_ddp else model).state_dict(),
                    "x_mean": x_mean.cpu(),
                    "x_std": x_std.cpu(),
                    "val_mse": val_mse,
                    "param_names": data["param_names"],
                    "param_ranges": data["param_ranges"],
                    "args": {"hidden": hidden, "layers": layers, "lr": lr},
                }
                torch.save(ckpt, os.path.join(out_dir, "best_gpu.pt"))

        if rank == 0 and trial is None:
            if epoch % 50 == 0 or epoch == args.epochs - 1:
                val_rel = (val_pred - y_val).abs().mean().item() / y_val.abs().mean().item()
                print(f"epoch {epoch:4d}  val_mse={val_mse:.6f}  val_rel={val_rel:.2%}")

        # Optuna pruning
        if trial is not None:
            try:
                import optuna
                trial.report(val_mse, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            except ImportError:
                pass

    cleanup_ddp()
    return best_val


def run_hpo(args):
    """Hyperparameter optimization with Optuna."""
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Run: pip install optuna")
        return

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", [64, 128, 256, 512])
        layers = trial.suggest_int("layers", 3, 8)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        return train_model(args, hidden=hidden, layers=layers, lr=lr, trial=trial)

    study = optuna.create_study(direction="minimize",
                                 pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))
    study.optimize(objective, n_trials=args.hpo_trials)

    print(f"\n{'='*55}")
    print(f"  HPO Results ({args.hpo_trials} trials)")
    print(f"{'='*55}")
    print(f"  Best val MSE: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")
    print(f"{'='*55}")

    # Retrain with best params
    print("\nRetraining with best hyperparameters...")
    train_model(args, **study.best_params)


def train(args):
    rank = 0
    if args.ddp and "RANK" in os.environ:
        rank = int(os.environ["RANK"])

    if args.hpo and rank == 0:
        run_hpo(args)
    else:
        if rank == 0:
            print(f"[Cd Surrogate GPU] Training with hidden={args.hidden}, "
                  f"layers={args.layers}")
        best = train_model(args)
        if rank == 0:
            print(f"\nBest val MSE: {best:.6f}")
            print(f"Checkpoint: {_SCRIPT_DIR}/outputs_gpu/best_gpu.pt")

            # Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"Best val MSE: {best:.6f}\n"
                    f"hidden={args.hidden}, layers={args.layers}",
                    ha="center", va="center", fontsize=14,
                    transform=ax.transAxes)
            ax.set_title("Cd Surrogate (GPU)")
            ax.axis("off")
            plt.savefig(os.path.join(_SCRIPT_DIR, "outputs_gpu", "train_gpu_result.png"), dpi=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO")
    parser.add_argument("--hpo_trials", type=int, default=50)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
