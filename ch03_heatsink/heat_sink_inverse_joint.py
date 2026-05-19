#!/usr/bin/env python3
"""
Chapter 3: Joint inversion — learn fin_height in ONE PINN training.

Usage:
    python heat_sink_inverse_joint.py --target_temp 40
    python heat_sink_inverse_joint.py --fast          # 800 steps, CI/demo
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from heat_sink_geometry import HeatSinkGeometry
from equations import heat_conduction_2d, robin_boundary


class ParametricPINN(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))


def train_joint(
    target_temp: float,
    steps: int,
    h_init: float,
    h_min: float,
    h_max: float,
    device: str,
    log_every: int,
    tol_temp: float,
) -> tuple[float, float, int]:
    model = ParametricPINN().to(device)
    fin_height = nn.Parameter(torch.tensor([h_init], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "lr": 1e-3}, {"params": [fin_height], "lr": 2e-2}],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    scale = 30.0
    T_source, k, h_conv, T_inf = 100.0, 1.0, 10.0, 0.0
    w_target = 200.0
    last_step = steps

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        h = torch.clamp(fin_height, h_min, h_max)
        # Resample domain at current h (non-diff through CSG; tip loss drives h)
        geo = HeatSinkGeometry(fin_height=float(h.item()))

        xi, yi = geo.sample_interior(1500)
        x = torch.tensor(xi / scale, device=device, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(yi / scale, device=device, dtype=torch.float32).unsqueeze(1)
        x.requires_grad_(True)
        y.requires_grad_(True)
        T = model(x, y)
        loss_pde = (heat_conduction_2d(T, x, y, k=k) ** 2).mean()

        xb, yb = geo.sample_bottom_bc(200)
        xbt = torch.tensor(xb / scale, device=device, dtype=torch.float32).unsqueeze(1)
        ybt = torch.tensor(yb / scale, device=device, dtype=torch.float32).unsqueeze(1)
        loss_bot = ((model(xbt, ybt) - T_source) ** 2).mean()

        xr, yr, nxr, nyr = geo.sample_robin_bc(400)
        xrt = torch.tensor(xr / scale, device=device, dtype=torch.float32).unsqueeze(1)
        yrt = torch.tensor(yr / scale, device=device, dtype=torch.float32).unsqueeze(1)
        xrt.requires_grad_(True)
        yrt.requires_grad_(True)
        nxrt = torch.tensor(nxr, device=device, dtype=torch.float32).unsqueeze(1)
        nyrt = torch.tensor(nyr, device=device, dtype=torch.float32).unsqueeze(1)
        Tr = model(xrt, yrt)
        loss_robin = (
            robin_boundary(Tr, xrt, yrt, nxrt, nyrt, k=k, h_conv=h_conv, T_inf=T_inf) ** 2
        ).mean()

        # Differentiable tip line w.r.t. fin_height (do not detach h here)
        tip_y_norm = (10.0 + h.squeeze()) / scale
        xt = (
            torch.linspace(-18.0, 18.0, 80, device=device, dtype=torch.float32) / scale
        ).unsqueeze(1)
        yt = tip_y_norm.expand_as(xt)
        T_tip = model(xt, yt).mean()
        loss_obs = (T_tip - target_temp) ** 2

        loss = loss_pde + 10.0 * loss_bot + 5.0 * loss_robin + w_target * loss_obs
        loss.backward()
        optimizer.step()
        scheduler.step()

        err = float(abs(T_tip.item() - target_temp))
        if step % log_every == 0 or step == 1:
            print(
                f"step {step:5d}  h={float(h.item()):6.2f} mm  "
                f"T_tip={float(T_tip.item()):7.3f}  |err|={err:.3f}  loss={loss.item():.4e}"
            )
        if err < tol_temp:
            last_step = step
            print(f"Early stop at step {step}: |T_tip-target| < {tol_temp}")
            break
    else:
        last_step = steps

    h_final = float(torch.clamp(fin_height, h_min, h_max).item())
    with torch.no_grad():
        h_t = torch.tensor([h_final], device=device, dtype=torch.float32)
        tip_y_norm = (10.0 + h_t.squeeze()) / scale
        xt = (
            torch.linspace(-18.0, 18.0, 80, device=device, dtype=torch.float32) / scale
        ).unsqueeze(1)
        yt = tip_y_norm.expand_as(xt)
        T_tip_f = float(model(xt, yt).mean().item())
    return h_final, T_tip_f, last_step


def main():
    p = argparse.ArgumentParser(description="Joint inversion for fin height")
    p.add_argument("--target_temp", type=float, default=40.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--fast", action="store_true", help="800 steps, looser tol (demo/CI)")
    p.add_argument("--h_init", type=float, default=20.0)
    p.add_argument("--h_min", type=float, default=5.0)
    p.add_argument("--h_max", type=float, default=40.0)
    args = p.parse_args()
    steps = 800 if args.fast else args.steps
    tol = 2.5 if args.fast else 0.5
    log_every = 200 if args.fast else 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Joint inversion on {device} (steps={steps}, tol={tol})")
    h, T_tip, used = train_joint(
        args.target_temp, steps, args.h_init, args.h_min, args.h_max, device, log_every, tol
    )
    print(f"\nResult: fin_height={h:.2f} mm  T_tip={T_tip:.3f}  target={args.target_temp}  steps={used}")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/inverse_joint_result.txt", "w", encoding="utf-8") as f:
        f.write(
            f"fin_height_mm={h}\nT_tip={T_tip}\ntarget={args.target_temp}\nsteps={used}\n"
        )


if __name__ == "__main__":
    main()
