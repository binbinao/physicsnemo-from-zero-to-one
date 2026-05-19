"""
Chapter 3: Inverse Problem — Optimize Fin Height
==================================================
Given a target average temperature at the fin tips, find the optimal
fin_height that achieves it using gradient-free optimization (scipy).

This demonstrates how PINNs enable inverse design:
  1. Train a parameterized PINN where fin_height is an input
  2. Use scipy.optimize to find the fin_height that minimizes
     |T_tip_avg - T_target|

For simplicity, we re-train a small PINN for each candidate fin_height
(brute-force approach suitable for demonstration — NOT joint inversion
in a single PINN training as described in some textbook PINN papers).

Usage:
    python heat_sink_inverse.py
    python heat_sink_inverse.py --target_temp 30.0 --n_evals 10
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

sys.path.insert(0, os.path.dirname(__file__))
from heat_sink_geometry import HeatSinkGeometry
from equations import heat_conduction_2d, robin_boundary


class QuickPINN(nn.Module):
    def __init__(self, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))


def quick_train(fin_height: float, steps: int = 500, device: str = "cpu"):
    """Train a small PINN for given fin_height and return avg tip temperature."""
    geo = HeatSinkGeometry(fin_height=fin_height)
    model = QuickPINN(hidden=64, depth=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scale = 30.0

    T_source = 100.0
    k, h_conv, T_inf = 1.0, 10.0, 0.0

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        # Interior
        xi, yi = geo.sample_interior(1000)
        x = torch.tensor(xi / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        y = torch.tensor(yi / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        T = model(x, y)
        res = heat_conduction_2d(T, x, y, k=k)
        loss_pde = (res ** 2).mean()

        # Bottom BC
        xb = np.random.uniform(-30, 30, 200)
        yb = np.zeros(200)
        xbt = torch.tensor(xb / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        ybt = torch.tensor(yb / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        loss_bot = ((model(xbt, ybt) - T_source) ** 2).mean()

        # Robin BC
        xr, yr, nxr, nyr = geo.sample_robin_bc(300)
        xrt = torch.tensor(xr / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        yrt = torch.tensor(yr / scale, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        nxrt = torch.tensor(nxr, dtype=torch.float32, device=device).unsqueeze(1)
        nyrt = torch.tensor(nyr, dtype=torch.float32, device=device).unsqueeze(1)
        Tr = model(xrt, yrt)
        res_r = robin_boundary(Tr, xrt, yrt, nxrt, nyrt, k=k, h_conv=h_conv, T_inf=T_inf)
        loss_robin = (res_r ** 2).mean()

        loss = loss_pde + 10.0 * loss_bot + 5.0 * loss_robin
        loss.backward()
        optimizer.step()

    # Evaluate average tip temperature
    model.eval()
    tip_y = 10.0 + fin_height  # top of fins
    x_tips = np.linspace(-20, 20, 100)
    y_tips = np.full(100, tip_y)
    xt = torch.tensor(x_tips / scale, dtype=torch.float32, device=device).unsqueeze(1)
    yt = torch.tensor(y_tips / scale, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        T_tip = model(xt, yt).cpu().numpy().mean()

    return float(T_tip)


def main():
    parser = argparse.ArgumentParser(description="Inverse: optimize fin height")
    parser.add_argument("--target_temp", type=float, default=40.0,
                        help="Target average temperature at fin tips")
    parser.add_argument("--n_evals", type=int, default=8,
                        help="Number of fin heights to evaluate")
    parser.add_argument("--steps", type=int, default=500,
                        help="Training steps per evaluation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target tip temperature: {args.target_temp}")
    print(f"Evaluating {args.n_evals} fin heights...\n")

    fin_heights = np.linspace(5.0, 40.0, args.n_evals)
    tip_temps = []

    for i, fh in enumerate(fin_heights):
        T_tip = quick_train(fh, steps=args.steps, device=device)
        tip_temps.append(T_tip)
        print(f"  fin_height={fh:5.1f} mm  ->  T_tip_avg={T_tip:.2f}")

    tip_temps = np.array(tip_temps)

    # Find best
    errors = np.abs(tip_temps - args.target_temp)
    best_idx = np.argmin(errors)
    print(f"\nBest fin_height = {fin_heights[best_idx]:.1f} mm "
          f"(T_tip = {tip_temps[best_idx]:.2f}, target = {args.target_temp})")

    # Plot
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fin_heights, tip_temps, "bo-", label="PINN prediction")
    ax.axhline(y=args.target_temp, color="r", linestyle="--",
               label=f"Target = {args.target_temp}")
    ax.axvline(x=fin_heights[best_idx], color="g", linestyle=":",
               alpha=0.7, label=f"Optimal h = {fin_heights[best_idx]:.1f} mm")
    ax.set_xlabel("Fin Height (mm)")
    ax.set_ylabel("Average Tip Temperature")
    ax.set_title("Inverse Problem: Fin Height Optimization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/inverse_optimization.png", dpi=150)
    print("Saved outputs/inverse_optimization.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
