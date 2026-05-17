"""
Chapter 2: Hydra-based 1D Heat Equation Training
==================================================
Same PINN as heat1d_pinn_raw.py but with Hydra config management.
Falls back to argparse if hydra-core is not installed.

Usage (with Hydra):
    python heat1d_train.py
    python heat1d_train.py arch=large training=full
    python heat1d_train.py arch.hidden=96 training.steps=2000

Usage (without Hydra):
    python heat1d_train.py --hidden 64 --depth 4 --steps 3000
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Try Hydra import
try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Network (same as raw version)
# ---------------------------------------------------------------------------
class HeatPINN(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


# ---------------------------------------------------------------------------
# PDE / sampling (reused from raw version)
# ---------------------------------------------------------------------------
def sample_interior(n, device):
    x = torch.rand(n, 1, device=device, requires_grad=True)
    t = torch.rand(n, 1, device=device, requires_grad=True) * 0.5
    return x, t

def sample_ic(n, device):
    x = torch.rand(n, 1, device=device, requires_grad=True)
    t = torch.zeros(n, 1, device=device, requires_grad=True)
    u_exact = torch.exp(-50.0 * (x - 0.5) ** 2)
    return x, t, u_exact

def sample_bc(n, device):
    t = torch.rand(n, 1, device=device, requires_grad=True) * 0.5
    x_left = torch.zeros(n, 1, device=device, requires_grad=True)
    x_right = torch.ones(n, 1, device=device, requires_grad=True)
    return x_left, x_right, t

def pde_residual(model, x, t, alpha=0.1):
    u = model(x, t)
    du = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u),
                             create_graph=True)
    du_dx, du_dt = du[0], du[1]
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx),
                                   create_graph=True)[0]
    return du_dt - alpha * d2u_dx2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(cfg: dict):
    """Train with a flat config dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: hidden={cfg['hidden']}, depth={cfg['depth']}, "
          f"steps={cfg['steps']}, lr={cfg['lr']}")

    model = HeatPINN(hidden=cfg["hidden"], depth=cfg["depth"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["steps"])

    w_pde, w_ic, w_bc = cfg.get("w_pde", 1.0), cfg.get("w_ic", 10.0), cfg.get("w_bc", 10.0)
    n_pde = cfg.get("n_pde", 2000)
    n_ic = cfg.get("n_ic", 200)
    n_bc = cfg.get("n_bc", 200)
    alpha = cfg.get("alpha", 0.1)

    history = {"loss": [], "pde": [], "ic": [], "bc": []}
    log_interval = max(1, cfg["steps"] // 20)

    for step in range(1, cfg["steps"] + 1):
        optimizer.zero_grad()

        # PDE
        x_int, t_int = sample_interior(n_pde, device)
        res = pde_residual(model, x_int, t_int, alpha)
        loss_pde = (res ** 2).mean()

        # IC
        x_ic, t_ic, u_ic = sample_ic(n_ic, device)
        loss_ic = ((model(x_ic, t_ic) - u_ic) ** 2).mean()

        # BC
        x_left, x_right, t_bc = sample_bc(n_bc, device)
        loss_bc = (model(x_left, t_bc) ** 2).mean() + \
                  (model(x_right, t_bc) ** 2).mean()

        loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()

        history["loss"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["bc"].append(loss_bc.item())

        if step % log_interval == 0 or step == 1:
            print(f"[{step:5d}/{cfg['steps']}] loss={loss.item():.4e} "
                  f"pde={loss_pde.item():.4e} ic={loss_ic.item():.4e} "
                  f"bc={loss_bc.item():.4e}")

    # Save
    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "heat1d_hydra.pt")
    torch.save({"model": model.state_dict(), "config": cfg,
                "history": history}, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")
    return model, history


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
if HAS_HYDRA:
    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main_hydra(cfg: DictConfig):
        print(OmegaConf.to_yaml(cfg))
        flat = OmegaConf.to_container(cfg, resolve=True)
        # Merge arch + training into flat dict
        merged = {}
        merged.update(flat.get("arch", {}))
        merged.update(flat.get("training", {}))
        merged.update({k: v for k, v in flat.items()
                       if k not in ("arch", "training")})
        run_training(merged)


def main_argparse():
    """Fallback when Hydra is not installed."""
    import argparse
    parser = argparse.ArgumentParser(description="Heat1D PINN (argparse fallback)")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_pde", type=int, default=2000)
    parser.add_argument("--n_ic", type=int, default=200)
    parser.add_argument("--n_bc", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    run_training(vars(args))


if __name__ == "__main__":
    if HAS_HYDRA:
        main_hydra()
    else:
        print("[INFO] hydra-core not found, using argparse fallback.")
        main_argparse()
