#!/usr/bin/env python3
"""
Chapter 3: Joint inversion — learn fin_height in ONE PINN training.

Contrast with heat_sink_inverse.py (brute-force retrain per height).

Usage:
    python heat_sink_inverse_joint.py --target_temp 40 --steps 3000
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
    """(x, y) -> T; geometry resampled each step from learnable fin_height."""

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
) -> tuple[float, float]:
    model = ParametricPINN().to(device)
    fin_height = nn.Parameter(torch.tensor([h_init], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": [fin_height], "lr": 5e-4}],
        lr=1e-3,
    )
    scale = 30.0
    T_source, k, h_conv, T_inf = 100.0, 1.0, 10.0, 0.0
    w_target = 50.0

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        h = torch.clamp(fin_height, h_min, h_max)
        geo = HeatSinkGeometry(fin_height=float(h.detach().item()))

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

        tip_y = 10.0 + float(h.item())
        x_tips = np.linspace(-18, 18, 80)
        y_tips = np.full(80, tip_y)
        xt = torch.tensor(x_tips / scale, device=device, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(y_tips / scale, device=device, dtype=torch.float32).unsqueeze(1)
        T_tip = model(xt, yt).mean()
        loss_obs = (T_tip - target_temp) ** 2

        loss = loss_pde + 10.0 * loss_bot + 5.0 * loss_robin + w_target * loss_obs
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == 1:
            print(
                f"step {step:5d}  h={float(h.item()):6.2f} mm  "
                f"T_tip={float(T_tip.item()):7.3f}  loss={loss.item():.4e}"
            )

    h_final = float(torch.clamp(fin_height, h_min, h_max).item())
    with torch.no_grad():
        tip_y = 10.0 + h_final
        xt = torch.tensor(
            np.linspace(-18, 18, 80) / scale, device=device, dtype=torch.float32
        ).unsqueeze(1)
        yt = torch.tensor(
            np.full(80, tip_y) / scale, device=device, dtype=torch.float32
        ).unsqueeze(1)
        T_tip_f = float(model(xt, yt).mean().item())
    return h_final, T_tip_f


def main():
    p = argparse.ArgumentParser(description="Joint inversion for fin height")
    p.add_argument("--target_temp", type=float, default=40.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--h_init", type=float, default=20.0)
    p.add_argument("--h_min", type=float, default=5.0)
    p.add_argument("--h_max", type=float, default=40.0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Joint inversion on {device} (single training, learnable fin_height)")
    h, T_tip = train_joint(
        args.target_temp, args.steps, args.h_init, args.h_min, args.h_max, device
    )
    print(f"\nResult: fin_height={h:.2f} mm  T_tip_avg={T_tip:.3f}  target={args.target_temp}")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/inverse_joint_result.txt", "w", encoding="utf-8") as f:
        f.write(f"fin_height_mm={h}\nT_tip={T_tip}\ntarget={args.target_temp}\n")


if __name__ == "__main__":
    main()
