"""
Collocation Point Density Convergence Study
=============================================
Trains the 1D heat PINN with varying collocation point counts and measures
L2 error against the Fourier series analytical solution.

This is the PINN equivalent of a mesh convergence / grid independence study
in FEM/FDM.

PDE:  u_t = α u_xx,  α = 0.01, x ∈ [0,1], t ∈ [0, 0.5]
IC:   u(x, 0) = exp(-50(x - 0.5)²)
BC:   u(0, t) = u(1, t) = 0

Usage:
    python convergence_study.py              # full study (~20 min on CPU)
    python convergence_study.py --steps 500  # quick sanity check for CI
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA = 0.01
T_EVAL = 0.5
N_FOURIER = 50
N_VALUES = [50, 100, 200, 500, 1000, 2000, 4000]

# ---------------------------------------------------------------------------
# PINN (same architecture as heat1d_pinn_raw.py)
# ---------------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


# ---------------------------------------------------------------------------
# Analytical solution via Fourier series
# ---------------------------------------------------------------------------
def analytical_solution(x: np.ndarray, t: float, n_terms: int = N_FOURIER) -> np.ndarray:
    """Fourier series solution for u_t = α u_xx with Gaussian IC and Dirichlet BC."""
    # Compute Fourier coefficients b_n = 2 * ∫₀¹ u₀(x) sin(nπx) dx
    x_quad = np.linspace(0, 1, 2000)
    u0_quad = np.exp(-50.0 * (x_quad - 0.5) ** 2)
    dx_quad = x_quad[1] - x_quad[0]

    u = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        sin_basis = np.sin(n * np.pi * x_quad)
        b_n = 2.0 * np.trapezoid(u0_quad * sin_basis, dx=dx_quad)
        decay = np.exp(-ALPHA * (n * np.pi) ** 2 * t)
        u += b_n * decay * np.sin(n * np.pi * x)
    return u


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------
def pde_residual(model: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    u = model(x, t)
    du = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u),
                             create_graph=True)
    du_dx, du_dt = du[0], du[1]
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx),
                                   create_graph=True)[0]
    return du_dt - ALPHA * d2u_dx2


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
def train_single(n_pde: int, steps: int, seed: int, device: str) -> float:
    """Train a PINN with n_pde collocation points and return L2 error."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PINN(hidden=64, depth=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, steps // 3),
                                                 gamma=0.5)

    for _ in range(steps):
        optimizer.zero_grad()

        # PDE loss
        x_int = torch.rand(n_pde, 1, device=device, requires_grad=True)
        t_int = torch.rand(n_pde, 1, device=device, requires_grad=True) * 0.5
        res = pde_residual(model, x_int, t_int)
        loss_pde = (res ** 2).mean()

        # IC loss
        n_ic = max(50, n_pde // 10)
        x_ic = torch.rand(n_ic, 1, device=device, requires_grad=True)
        t_ic = torch.zeros(n_ic, 1, device=device, requires_grad=True)
        u_ic_exact = torch.exp(-50.0 * (x_ic - 0.5) ** 2)
        loss_ic = ((model(x_ic, t_ic) - u_ic_exact) ** 2).mean()

        # BC loss
        n_bc = max(50, n_pde // 10)
        t_bc = torch.rand(n_bc, 1, device=device, requires_grad=True) * 0.5
        x_left = torch.zeros(n_bc, 1, device=device, requires_grad=True)
        x_right = torch.ones(n_bc, 1, device=device, requires_grad=True)
        loss_bc = (model(x_left, t_bc) ** 2).mean() + (model(x_right, t_bc) ** 2).mean()

        loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Compute L2 error at t = T_EVAL
    model.eval()
    x_test = np.linspace(0, 1, 500)
    u_exact = analytical_solution(x_test, T_EVAL)

    x_t = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device)
    t_t = torch.full_like(x_t, T_EVAL)
    with torch.no_grad():
        u_pred = model(x_t, t_t).cpu().numpy().flatten()

    l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2)) / (np.sqrt(np.mean(u_exact ** 2)) + 1e-12)
    return l2_error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PINN Collocation Convergence Study")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Steps per run: {args.steps}")
    print("=" * 60)

    errors = []
    print(f"\n{'N':>6s}  |  {'L2 Error':>12s}  |  {'Time (s)':>8s}")
    print("-" * 42)
    for n in N_VALUES:
        t0 = time.time()
        err = train_single(n, args.steps, args.seed, device)
        elapsed = time.time() - t0
        errors.append(err)
        print(f"{n:>6d}  |  {err:>12.6e}  |  {elapsed:>8.1f}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'N':>6s}  {'L2 Error':>12s}")
    print("-" * 22)
    for n, err in zip(N_VALUES, errors):
        print(f"{n:>6d}  {err:>12.6e}")

    # Estimate convergence rate via linear regression in log-log space
    log_n = np.log(np.array(N_VALUES, dtype=float))
    log_err = np.log(np.array(errors))
    slope, intercept = np.polyfit(log_n, log_err, 1)
    print(f"\nEstimated convergence rate: O(N^{{{slope:.2f}}})")

    # Plot
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N_VALUES, errors, "o-", linewidth=2, markersize=8, label="PINN L2 error")
    # Reference slope
    n_ref = np.array(N_VALUES, dtype=float)
    ref_line = errors[0] * (n_ref / n_ref[0]) ** (-0.5)
    ax.loglog(N_VALUES, ref_line, "--", color="gray", alpha=0.7, label=r"$O(N^{-1/2})$ reference")
    ax.set_xlabel("Number of Collocation Points (N)", fontsize=12)
    ax.set_ylabel("Relative L2 Error", fontsize=12)
    ax.set_title("PINN Convergence: Collocation Point Density", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/convergence_study.png", dpi=150)
    print("\nPlot saved to outputs/convergence_study.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
