#!/usr/bin/env python3
"""
第 1 章 · 解法一：用数据驱动的 MLP 拟合弹簧振子

  python mlp_spring.py --epochs 1000

物理方程：m·ẍ + k·x = 0  (简谐振动)
训练方式：用解析解生成 1000 个 (t, x) 数据点，MSE 拟合。
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")  # 无头环境兼容
import matplotlib.pyplot as plt

# ── 物理参数 ──────────────────────────────────────────────
M = 1.0  # 质量 kg
K = 4.0  # 劲度系数 N/m → ω = 2 rad/s
X0 = 1.0  # 初始位移
V0 = 0.0  # 初始速度
T_MAX = 10.0  # 模拟时间


# ── 解析解 ────────────────────────────────────────────────
def analytical_solution(t, m=M, k=K, x0=X0, v0=V0):
    """x(t) = x0·cos(ωt) + (v0/ω)·sin(ωt)"""
    omega = np.sqrt(k / m)
    return x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)


# ── 数据集 ────────────────────────────────────────────────
def make_dataset(n=1000, t_max=T_MAX):
    t = np.linspace(0, t_max, n).reshape(-1, 1).astype(np.float32)
    x = analytical_solution(t).astype(np.float32)
    return torch.tensor(t), torch.tensor(x)


# ── MLP 模型 ──────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden=32, depth=3):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


# ── 训练 ──────────────────────────────────────────────────
def train(model, t, x, epochs=1000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(t)
        loss = nn.functional.mse_loss(pred, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"epoch {epoch:5d}  loss {loss.item():.6f}")
    return losses


# ── 可视化 ─────────────────────────────────────────────────
def visualize(t_test, pred, truth, losses, save_path="mlp_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # loss curve
    axes[0].semilogy(losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss")

    # fit in training interval
    mask_train = t_test <= T_MAX
    axes[1].plot(t_test[mask_train], truth[mask_train], "k--", label="Analytical")
    axes[1].plot(t_test[mask_train], pred[mask_train], "r-", alpha=0.8, label="MLP")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x(t)")
    axes[1].set_title(f"Training interval [0, {T_MAX}]")
    axes[1].legend()

    # extrapolation
    axes[2].plot(t_test, truth, "k--", label="Analytical")
    axes[2].plot(t_test, pred, "r-", alpha=0.8, label="MLP")
    axes[2].axvline(T_MAX, color="gray", ls=":", label="Train boundary")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x(t)")
    axes[2].set_title(f"Extrapolation [0, {T_MAX * 2}]")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"图表已保存到 {save_path}")
    plt.close()


# ── main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MLP 拟合弹簧振子")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    t, x = make_dataset()
    model = MLP(hidden=args.hidden, depth=args.depth)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    losses = train(model, t, x, epochs=args.epochs, lr=args.lr)

    # 推理 & 可视化（含外推到 2×T_MAX）
    t_test = torch.linspace(0, T_MAX * 2, 500).reshape(-1, 1)
    with torch.no_grad():
        pred = model(t_test).numpy().flatten()
    truth = analytical_solution(t_test.numpy().flatten())
    visualize(t_test.numpy().flatten(), pred, truth, losses)

    # 外推误差
    mask_extrap = t_test.numpy().flatten() > T_MAX
    if mask_extrap.any():
        extrap_err = np.mean(np.abs(pred[mask_extrap] - truth[mask_extrap]))
        print(f"外推区间 [{T_MAX}, {T_MAX*2}] 平均绝对误差: {extrap_err:.4f}")


if __name__ == "__main__":
    main()
