"""
Chapter 3: Pointwise validation report (CAE-style sanity checks).
================================================================
Not a sign-off tool — compares PINN against:
  - PDE / BC residual RMS on dense samples
  - Order-of-magnitude 1D fin tip temperature bound

Usage:
    python validator.py --checkpoint outputs/model.pt   # if saved
    # Or called automatically at end of heat_sink_train.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

from heat_sink_geometry import HeatSinkGeometry
from equations import heat_conduction_2d, robin_boundary, neumann_boundary


def fin_tip_temperature_bound(T_source: float, T_inf: float, h: float, k: float,
                            fin_length: float) -> float:
    """1D fin with convection at tip: loose upper bound on |T_source - T_tip|."""
    m = np.sqrt(h / max(k * fin_length, 1e-6))
    eta = np.tanh(m * fin_length) / (m * fin_length + 1e-8)
    T_tip = T_inf + (T_source - T_inf) * (1.0 - min(eta, 1.0))
    return float(T_tip)


def run_validation(
    model: torch.nn.Module,
    geo: HeatSinkGeometry,
    cfg: dict,
    device: str = "cpu",
    n_int: int = 2000,
    n_bc: int = 400,
) -> dict[str, Any]:
    model.eval()
    k = cfg.get("k", 1.0)
    Q = cfg.get("Q", 0.0)
    T_source = cfg.get("T_source", 100.0)
    h_conv = cfg.get("h_conv", 10.0)
    T_inf = cfg.get("T_inf", 0.0)
    scale = cfg.get("scale", 30.0)

    # Interior PDE residual
    x, y = geo.sample_interior(n_int)
    x_t = torch.tensor(x / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    y_t = torch.tensor(y / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    T = model(x_t, y_t)
    res_pde = heat_conduction_2d(T, x_t, y_t, k=k, Q=Q)
    pde_rms = float(torch.sqrt((res_pde**2).mean()).detach().cpu())

    # Robin on fin tops (sample)
    xr, yr, nx, ny = geo.sample_robin_bc(n_bc)
    xr_t = torch.tensor(xr / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    yr_t = torch.tensor(yr / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    nx_t = torch.tensor(nx, dtype=torch.float32, device=device).unsqueeze(1)
    ny_t = torch.tensor(ny, dtype=torch.float32, device=device).unsqueeze(1)
    Tr = model(xr_t, yr_t)
    res_robin = robin_boundary(Tr, xr_t, yr_t, nx_t, ny_t, k=k, h_conv=h_conv, T_inf=T_inf)
    robin_rms = float(torch.sqrt((res_robin**2).mean()).detach().cpu())

    # Temperature range vs 1D bound
    with torch.no_grad():
        T_vals = T.detach().cpu().numpy()
    T_max = float(T_vals.max())
    T_min = float(T_vals.min())
    fin_h = cfg.get("fin_height", 20.0)
    T_tip_bound = fin_tip_temperature_bound(T_source, T_inf, h_conv, k, fin_h / scale)

    report = {
        "pde_residual_rms": pde_rms,
        "robin_residual_rms": robin_rms,
        "T_min": T_min,
        "T_max": T_max,
        "T_source_config": T_source,
        "fin_tip_1d_bound": T_tip_bound,
        "pass_pde": pde_rms < 1e-1,
        "pass_robin": robin_rms < 1e-1,
        "pass_temperature_range": T_min >= T_inf - 5.0 and T_max <= T_source + 50.0,
        "note": "2D PINN micro-model; not comparable to Icepak sign-off without imported CFD reference.",
    }
    report["pass_all"] = report["pass_pde"] and report["pass_robin"] and report["pass_temperature_range"]
    return report


def load_and_validate(ckpt_path: str, device: str = "cpu") -> dict:
    from heat_sink_train import HeatSinkNet
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["scale"] = ckpt.get("scale", 30.0)
    geo = HeatSinkGeometry(fin_height=cfg.get("fin_height", 20.0), n_fins=cfg.get("n_fins", 3))
    model = HeatSinkNet(hidden=cfg.get("hidden", 128), depth=cfg.get("depth", 5))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return run_validation(model, geo, cfg, device=device)


def save_report(report: dict, out_dir: str = "outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "validation_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run ch03 PINN validation report")
    p.add_argument("--checkpoint", default="outputs/heat_sink.pt")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()
    rep = load_and_validate(args.checkpoint)
    print(json.dumps(rep, indent=2))
    print("Saved:", save_report(rep, args.out_dir))
