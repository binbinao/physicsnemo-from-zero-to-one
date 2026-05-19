"""
Chapter 3: Heat Sink Training Script
======================================
Train a PINN for steady-state 2D heat conduction in a finned heat sink.

Constraints:
  1. Interior: k(∂²T/∂x² + ∂²T/∂y²) + Q = 0
  2. Bottom BC: T = T_source (Dirichlet, heat source)
  3. Side BC: ∂T/∂n = 0 (insulated / Neumann)
  4. Robin BC: -k ∂T/∂n = h(T - T_inf) (convective cooling on fins)

Usage:
    python heat_sink_train.py
    python heat_sink_train.py --steps 5000 --hidden 128
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from heat_sink_geometry import HeatSinkGeometry
from equations import heat_conduction_2d, robin_boundary, neumann_boundary

# Try Hydra
try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class HeatSinkNet(nn.Module):
    """(x, y) -> T(x, y)"""

    def __init__(self, hidden: int = 128, depth: int = 5):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Geometry
    geo = HeatSinkGeometry(
        fin_height=cfg.get("fin_height", 20.0),
        n_fins=cfg.get("n_fins", 3)
    )

    # Physics params
    k = cfg.get("k", 1.0)           # thermal conductivity
    Q = cfg.get("Q", 0.0)           # volumetric heat source
    T_source = cfg.get("T_source", 100.0)  # bottom temperature
    h_conv = cfg.get("h_conv", 10.0)       # convection coefficient
    T_inf = cfg.get("T_inf", 0.0)          # ambient temperature

    # Network
    model = HeatSinkNet(hidden=cfg.get("hidden", 128),
                        depth=cfg.get("depth", 5)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    steps = cfg.get("steps", 3000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # Sampling sizes
    n_int = cfg.get("n_interior", 3000)
    n_bot = cfg.get("n_bottom", 300)
    n_side = cfg.get("n_side", 300)
    n_robin = cfg.get("n_robin", 500)

    # Loss weights
    w_pde = cfg.get("w_pde", 1.0)
    w_bot = cfg.get("w_bottom", 10.0)
    w_side = cfg.get("w_side", 1.0)
    w_robin = cfg.get("w_robin", 5.0)

    history = {"loss": [], "pde": [], "bottom": [], "side": [], "robin": []}
    log_interval = max(1, steps // 20)

    # Normalization scale (geometry is in mm, normalize to ~unit)
    scale = 30.0

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        # --- Interior PDE ---
        xi, yi = geo.sample_interior(n_int)
        x = torch.tensor(xi / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        y = torch.tensor(yi / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        T_pred = model(x, y)
        res_pde = heat_conduction_2d(T_pred, x, y, k=k, Q=Q)
        loss_pde = (res_pde ** 2).mean()

        # --- Bottom BC (Dirichlet) ---
        xb, yb = geo.sample_bottom_bc(n_bot)
        xb_t = torch.tensor(xb / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        yb_t = torch.tensor(yb / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        T_bot = model(xb_t, yb_t)
        loss_bot = ((T_bot - T_source) ** 2).mean()

        # --- Side BC (Neumann, insulated) ---
        xs, ys = geo.sample_side_bc(n_side)
        # Normal vectors: left wall nx=-1, right wall nx=+1
        nx_side = np.where(xs < 0, -1.0, 1.0)
        ny_side = np.zeros_like(xs)
        xs_t = torch.tensor(xs / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        ys_t = torch.tensor(ys / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        nx_t = torch.tensor(nx_side, dtype=torch.float32, device=device).unsqueeze(1)
        ny_t = torch.tensor(ny_side, dtype=torch.float32, device=device).unsqueeze(1)
        T_side = model(xs_t, ys_t)
        res_side = neumann_boundary(T_side, xs_t, ys_t, nx_t, ny_t)
        loss_side = (res_side ** 2).mean()

        # --- Robin BC (convective) ---
        xr, yr, nxr, nyr = geo.sample_robin_bc(n_robin)
        xr_t = torch.tensor(xr / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        yr_t = torch.tensor(yr / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        nxr_t = torch.tensor(nxr, dtype=torch.float32, device=device).unsqueeze(1)
        nyr_t = torch.tensor(nyr, dtype=torch.float32, device=device).unsqueeze(1)
        T_robin = model(xr_t, yr_t)
        res_robin = robin_boundary(T_robin, xr_t, yr_t, nxr_t, nyr_t,
                                   k=k, h_conv=h_conv, T_inf=T_inf)
        loss_robin = (res_robin ** 2).mean()

        # Total loss
        loss = (w_pde * loss_pde + w_bot * loss_bot +
                w_side * loss_side + w_robin * loss_robin)
        loss.backward()
        optimizer.step()
        scheduler.step()

        history["loss"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["bottom"].append(loss_bot.item())
        history["side"].append(loss_side.item())
        history["robin"].append(loss_robin.item())

        if step % log_interval == 0 or step == 1:
            print(f"[{step:5d}/{steps}] loss={loss.item():.4e} "
                  f"pde={loss_pde.item():.4e} bot={loss_bot.item():.4e} "
                  f"side={loss_side.item():.4e} robin={loss_robin.item():.4e}")

    # Save
    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "heat_sink.pt")
    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "history": history,
        "scale": scale,
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # CAE-style validation report (see validator.py)
    from validator import run_validation, save_report
    cfg_val = dict(cfg)
    cfg_val["scale"] = scale
    report = run_validation(model, geo, cfg_val, device=device)
    vpath = save_report(report, out_dir)
    print(f"Validation report: {vpath}  pass_all={report['pass_all']}")

    return model, history, geo, scale


# ---------------------------------------------------------------------------
# Quick visualization
# ---------------------------------------------------------------------------
def quick_viz(model, geo, scale, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # Grid
    nx, ny = 300, 200
    xg = np.linspace(-35, 35, nx)
    yg = np.linspace(-2, 35, ny)
    X, Y = np.meshgrid(xg, yg)

    # Predict
    xf = torch.tensor(X.ravel() / scale, dtype=torch.float32, device=device).unsqueeze(1)
    yf = torch.tensor(Y.ravel() / scale, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        T = model(xf, yf).cpu().numpy().reshape(ny, nx)

    # Mask outside geometry
    mask = geo.contains(X.ravel(), Y.ravel()).reshape(ny, nx)
    T_masked = np.where(mask, T, np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    c = ax.pcolormesh(X, Y, T_masked, shading="auto", cmap="hot")
    fig.colorbar(c, ax=ax, label="Temperature")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Heat Sink Temperature Field")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(out_dir, "temperature_field.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def get_default_cfg():
    return {
        "hidden": 128, "depth": 5, "steps": 3000, "lr": 1e-3,
        "fin_height": 20.0, "n_fins": 3,
        "k": 1.0, "Q": 0.0, "T_source": 100.0, "h_conv": 10.0, "T_inf": 0.0,
        "n_interior": 3000, "n_bottom": 300, "n_side": 300, "n_robin": 500,
        "w_pde": 1.0, "w_bottom": 10.0, "w_side": 1.0, "w_robin": 5.0,
        "output_dir": "outputs",
    }


if HAS_HYDRA:
    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main_hydra(cfg: DictConfig):
        flat = get_default_cfg()
        flat.update(OmegaConf.to_container(cfg, resolve=True))
        model, history, geo, scale = train(flat)
        quick_viz(model, geo, scale, flat["output_dir"])


def main_argparse():
    parser = argparse.ArgumentParser()
    defaults = get_default_cfg()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    cfg = vars(args)
    model, history, geo, scale = train(cfg)
    quick_viz(model, geo, scale, cfg["output_dir"])


if __name__ == "__main__":
    if HAS_HYDRA:
        main_hydra()
    else:
        print("[INFO] hydra-core not found, using argparse fallback.")
        main_argparse()
