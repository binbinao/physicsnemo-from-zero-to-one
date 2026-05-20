#!/usr/bin/env python3
"""
Chapter 1 - Solution B: Physics-Informed Neural Network (PINN) for Spring Oscillator

  python pinn_spring.py --epochs 5000

PDE: m*x'' + k*x = 0
Training: zero data, three-term loss = PDE residual + IC position + IC velocity
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Physical parameters ───────────────────────────────────
M = 1.0
K = 4.0  # ω = 2 rad/s
X0 = 1.0
V0 = 0.0
T_MAX = 10.0


def analytical_solution(t):
    omega = np.sqrt(K / M)
    return X0 * np.cos(omega * t) + (V0 / omega) * np.sin(omega * t)


# ── PINN model (same architecture as MLP; differs in loss) ─
class PINN(nn.Module):
    def __init__(self, hidden=32, depth=4):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


# ── PINN three-term loss ───────────────────────────────────
def pinn_loss(model, t_collocation, x0=X0, v0=V0, m=M, k=K):
    """
    Returns (pde_loss, ic_pos_loss, ic_vel_loss)
    """
    device = t_collocation.device

    # ---- 1) PDE residual loss: m*x'' + k*x = 0 ----
    t = t_collocation.requires_grad_(True)
    x = model(t)

    # First derivative dx/dt
    dx = torch.autograd.grad(
        outputs=x,
        inputs=t,
        grad_outputs=torch.ones_like(x),
        create_graph=True,  # Required: need to differentiate dx again
    )[0]

    # Second derivative d²x/dt²
    ddx = torch.autograd.grad(
        outputs=dx,
        inputs=t,
        grad_outputs=torch.ones_like(dx),
        create_graph=True,
    )[0]

    residual = m * ddx + k * x
    loss_pde = (residual**2).mean()

    # ---- 2) IC position loss: x(0) = x0 ----
    t0 = torch.zeros(1, 1, requires_grad=True, device=device)
    x_at_0 = model(t0)
    loss_ic_pos = ((x_at_0 - x0) ** 2).squeeze()

    # ---- 3) IC velocity loss: dx/dt|_{t=0} = v0 ----
    dx_at_0 = torch.autograd.grad(
        x_at_0, t0, torch.ones_like(x_at_0), create_graph=True
    )[0]
    loss_ic_vel = ((dx_at_0 - v0) ** 2).squeeze()

    return loss_pde, loss_ic_pos, loss_ic_vel


# ── Training ─────────────────────────────────────────────
def train(model, epochs=5000, n_collocation=256, lr=1e-3,
          w_pde=1.0, w_ic_pos=100.0, w_ic_vel=100.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"pde": [], "ic_pos": [], "ic_vel": [], "total": []}

    for epoch in range(epochs):
        # Resample collocation points each epoch (acts as data augmentation)
        t_col = torch.rand(n_collocation, 1) * T_MAX

        optimizer.zero_grad()
        l_pde, l_ic_p, l_ic_v = pinn_loss(model, t_col)
        loss = w_pde * l_pde + w_ic_pos * l_ic_p + w_ic_vel * l_ic_v
        loss.backward()
        optimizer.step()

        history["pde"].append(l_pde.item())
        history["ic_pos"].append(l_ic_p.item())
        history["ic_vel"].append(l_ic_v.item())
        history["total"].append(loss.item())

        if epoch % 500 == 0:
            print(
                f"epoch {epoch:5d}  total {loss.item():.4e}  "
                f"pde {l_pde.item():.4e}  ic_pos {l_ic_p.item():.4e}  "
                f"ic_vel {l_ic_v.item():.4e}"
            )

    return history


# ── Visualization ─────────────────────────────────────────
def visualize(t_test, pred, truth, history, save_path="pinn_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Three-term loss
    for key, color, label in [
        ("pde", "blue", "PDE residual"),
        ("ic_pos", "orange", "IC position"),
        ("ic_vel", "green", "IC velocity"),
    ]:
        axes[0].semilogy(history[key], color=color, alpha=0.7, label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("PINN Loss (3 terms)")
    axes[0].legend(fontsize=8)

    # training interval
    mask = t_test <= T_MAX
    axes[1].plot(t_test[mask], truth[mask], "k--", label="Analytical")
    axes[1].plot(t_test[mask], pred[mask], "r-", alpha=0.8, label="PINN")
    axes[1].set_title(f"Training interval [0, {T_MAX}]")
    axes[1].legend()

    # extrapolation
    axes[2].plot(t_test, truth, "k--", label="Analytical")
    axes[2].plot(t_test, pred, "r-", alpha=0.8, label="PINN")
    axes[2].axvline(T_MAX, color="gray", ls=":", label="Train boundary")
    axes[2].set_title(f"Extrapolation [0, {T_MAX * 3}]")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved to {save_path}")
    plt.close()


# ── main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PINN solver for spring oscillator")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-collocation", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    model = PINN(hidden=args.hidden, depth=args.depth)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training mode: zero data, physics-driven")

    history = train(model, epochs=args.epochs, n_collocation=args.n_collocation, lr=args.lr)

    # Inference (extrapolate to 3*T_MAX to show PINN's periodic learning)
    t_test = torch.linspace(0, T_MAX * 3, 600).reshape(-1, 1)
    with torch.no_grad():
        pred = model(t_test).numpy().flatten()
    truth = analytical_solution(t_test.numpy().flatten())

    visualize(t_test.numpy().flatten(), pred, truth, history)

    # Training interval & extrapolation error
    t_np = t_test.numpy().flatten()
    mask_train = t_np <= T_MAX
    mask_extrap = t_np > T_MAX
    print(f"Training interval MAE: {np.mean(np.abs(pred[mask_train] - truth[mask_train])):.6f}")
    print(f"Extrapolation MAE: {np.mean(np.abs(pred[mask_extrap] - truth[mask_extrap])):.6f}")


if __name__ == "__main__":
    main()
