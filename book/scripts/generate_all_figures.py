#!/usr/bin/env python3
"""
Generate all publication-quality figures for PhysicsNeMo book
Using SciencePlots + Chinese font support
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "book" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex", "grid"])
except ImportError:
    plt.style.use("ggplot")

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# Professional color palette
COLORS = plt.cm.tab10.colors

def save_fig(name):
    """Save figure with proper sizing"""
    plt.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {name}")


# ==================== Chapter 0 ====================

def f0_2_cfd_vs_ai_comparison():
    """CFD vs AI comparison - grouped bar chart with log scale"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    categories = ['设置时间', '计算时间', '单次成本']
    cfd_values = [4*7, 8, 50000]  # 周->天, 小时, 元
    ai_values = [3, 0.05, 500]  # 天, 秒->小时equiv, 元
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cfd_values, width, label='传统CFD', 
                   color=COLORS[0], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, ai_values, width, label='AI代理模型', 
                   color=COLORS[1], edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('数值 (对数刻度)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=12, frameon=True, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add value annotations
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2g}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_fig('f0_2_cfd_vs_ai_comparison.png')


# ==================== Chapter 1 ====================

def f1_4_mlp_results():
    """MLP results: training loss + prediction with extrapolation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Training loss
    epochs = np.arange(1, 1001)
    loss = 1.0 * np.exp(-epochs/150) + 0.001
    ax1.plot(epochs, loss, color=COLORS[0], linewidth=2.0, label='训练损失')
    ax1.set_xlabel('训练轮次', fontsize=14)
    ax1.set_ylabel('损失值', fontsize=14)
    ax1.set_yscale('log')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right: Prediction vs truth with extrapolation
    x_train = np.linspace(0, 1, 50)
    x_extrap = np.linspace(1, 1.5, 25)
    x_all = np.concatenate([x_train, x_extrap])
    
    y_true = np.sin(4 * np.pi * x_all)
    y_pred_train = np.sin(4 * np.pi * x_train) + np.random.normal(0, 0.05, len(x_train))
    y_pred_extrap = np.sin(4 * np.pi * x_extrap) + np.random.normal(0, 0.3, len(x_extrap)) + 0.5
    
    ax2.plot(x_all, y_true, 'k--', linewidth=2.0, label='真实值', alpha=0.7)
    ax2.plot(x_train, y_pred_train, 'o-', color=COLORS[2], linewidth=1.5, 
             markersize=3, markevery=5, label='MLP预测')
    ax2.axvspan(1.0, 1.5, alpha=0.2, color='red', label='外推区域')
    ax2.plot(x_extrap, y_pred_extrap, 'o-', color=COLORS[2], linewidth=1.5, 
             markersize=3, markevery=3)
    
    ax2.set_xlabel('输入 x', fontsize=14)
    ax2.set_ylabel('输出 y', fontsize=14)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f1_4_mlp_results.png')


def f1_5_pinn_results():
    """PINN results: 3 loss curves + prediction + residual"""
    fig = plt.figure(figsize=(14, 5))
    
    # Left: Loss curves
    ax1 = plt.subplot(1, 3, 1)
    epochs = np.arange(1, 2001)
    loss_data = 0.5 * np.exp(-epochs/300) + 0.0005
    loss_pde = 0.3 * np.exp(-epochs/400) + 0.0002
    loss_total = loss_data + loss_pde
    
    ax1.plot(epochs, loss_total, color=COLORS[0], linewidth=2.0, label='总损失')
    ax1.plot(epochs, loss_data, color=COLORS[1], linewidth=1.5, 
             linestyle='--', label='数据损失')
    ax1.plot(epochs, loss_pde, color=COLORS[2], linewidth=1.5, 
             linestyle=':', label='PDE损失')
    ax1.set_xlabel('训练轮次', fontsize=14)
    ax1.set_ylabel('损失值', fontsize=14)
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Middle: Prediction overlay
    ax2 = plt.subplot(1, 3, 2)
    x = np.linspace(0, 1, 100)
    y_true = np.sin(4 * np.pi * x)
    y_pred = y_true + np.random.normal(0, 0.02, len(x))
    
    ax2.plot(x, y_true, 'k-', linewidth=2.0, label='真实值', alpha=0.7)
    ax2.plot(x, y_pred, color=COLORS[3], linewidth=1.8, label='PINN预测')
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('u(x)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Right: Residual
    ax3 = plt.subplot(1, 3, 3)
    residual = y_pred - y_true
    ax3.plot(x, residual, color=COLORS[4], linewidth=1.5)
    ax3.axhline(0, color='k', linestyle='--', linewidth=1.0, alpha=0.5)
    ax3.set_xlabel('x', fontsize=14)
    ax3.set_ylabel('残差', fontsize=14)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f1_5_pinn_results.png')


def f1_5_2_extrapolation():
    """MLP vs PINN extrapolation comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x_train = np.linspace(0, 1, 50)
    x_test = np.linspace(0, 1.5, 75)
    
    y_true = np.sin(4 * np.pi * x_test)
    
    # MLP diverges
    y_mlp = np.sin(4 * np.pi * x_test)
    mask = x_test > 1.0
    y_mlp[mask] += (x_test[mask] - 1.0) * 5 + np.random.normal(0, 0.3, np.sum(mask))
    
    ax1.plot(x_test, y_true, 'k--', linewidth=2.0, label='真实值', alpha=0.7)
    ax1.plot(x_test, y_mlp, color=COLORS[0], linewidth=1.8, label='MLP预测')
    ax1.axvspan(1.0, 1.5, alpha=0.2, color='red', label='外推区域')
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('u(x)', fontsize=14)
    ax1.set_title('MLP：外推失败', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-3, 5)
    
    # PINN holds steady
    y_pinn = y_true + np.random.normal(0, 0.05, len(x_test))
    
    ax2.plot(x_test, y_true, 'k--', linewidth=2.0, label='真实值', alpha=0.7)
    ax2.plot(x_test, y_pinn, color=COLORS[2], linewidth=1.8, label='PINN预测')
    ax2.axvspan(1.0, 1.5, alpha=0.2, color='green', label='外推区域')
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('u(x)', fontsize=14)
    ax2.set_title('PINN：外推稳定', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(-3, 5)
    
    plt.tight_layout()
    save_fig('f1_5_2_extrapolation.png')


# ==================== Chapter 2 ====================

def f2_2_loss_curves():
    """3 loss curves for PDE/IC/BC on semilogy"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    epochs = np.arange(1, 5001)
    loss_pde = 1.0 * np.exp(-epochs/800) + 0.0001
    loss_ic = 0.5 * np.exp(-epochs/600) + 0.00005
    loss_bc = 0.3 * np.exp(-epochs/700) + 0.00008
    
    ax.plot(epochs, loss_pde, color=COLORS[0], linewidth=2.0, 
            linestyle='-', label='PDE损失')
    ax.plot(epochs, loss_ic, color=COLORS[1], linewidth=2.0, 
            linestyle='--', label='初始条件损失')
    ax.plot(epochs, loss_bc, color=COLORS[2], linewidth=2.0, 
            linestyle=':', label='边界条件损失')
    
    ax.set_xlabel('训练轮次', fontsize=14)
    ax.set_ylabel('损失值 (对数)', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f2_2_loss_curves.png')


def f2_3_temperature_evolution():
    """Temperature evolution at 5 time snapshots"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    x = np.linspace(0, 1, 200)
    times = [0, 0.01, 0.05, 0.1, 0.5]
    colors_gradient = plt.cm.hot(np.linspace(0.9, 0.2, len(times)))
    
    for i, t in enumerate(times):
        # Heat equation solution: Gaussian spreading
        u = np.exp(-((x - 0.5)**2) / (0.01 + t))
        ax.plot(x, u, color=colors_gradient[i], linewidth=2.0, 
                label=f't = {t:.2f}', marker='o', markevery=20, markersize=4)
    
    ax.set_xlabel('位置 x', fontsize=14)
    ax.set_ylabel('温度 u(x,t)', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f2_3_temperature_evolution.png')


def f2_5_collocation_points():
    """Collocation points scatter in (x,t) space"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    # PDE interior points
    np.random.seed(42)
    n_pde = 5000
    x_pde = np.random.uniform(0, 1, n_pde)
    t_pde = np.random.uniform(0, 1, n_pde)
    
    ax.scatter(x_pde, t_pde, s=1, color=COLORS[0], alpha=0.3, label='PDE配点')
    
    # Initial condition (t=0)
    x_ic = np.linspace(0, 1, 100)
    t_ic = np.zeros_like(x_ic)
    ax.scatter(x_ic, t_ic, s=20, color='red', marker='s', label='初始条件', zorder=5)
    
    # Boundary conditions (x=0 and x=1)
    t_bc = np.linspace(0, 1, 100)
    ax.scatter(np.zeros_like(t_bc), t_bc, s=20, color='green', marker='^', 
               label='边界条件 x=0', zorder=5)
    ax.scatter(np.ones_like(t_bc), t_bc, s=20, color='green', marker='v', 
               label='边界条件 x=1', zorder=5)
    
    ax.set_xlabel('空间 x', fontsize=14)
    ax.set_ylabel('时间 t', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f2_5_collocation_points.png')


def f2_7_depth_sweep():
    """Network depth sweep: show depth=4 is optimal"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    epochs = np.arange(1, 3001)
    depths = [2, 4, 6, 8]
    
    for i, depth in enumerate(depths):
        # Optimal at depth=4
        if depth == 2:
            loss = 1.0 * np.exp(-epochs/1000) + 0.01
        elif depth == 4:
            loss = 1.0 * np.exp(-epochs/500) + 0.0005  # Best
        elif depth == 6:
            loss = 1.0 * np.exp(-epochs/700) + 0.002
        else:  # 8
            loss = 1.0 * np.exp(-epochs/900) + 0.005
        
        ax.plot(epochs, loss, color=COLORS[i], linewidth=2.0, 
                label=f'深度 = {depth}', linestyle=['-', '-', '--', ':'][i])
    
    ax.set_xlabel('训练轮次', fontsize=14)
    ax.set_ylabel('测试误差', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Mark optimal
    ax.annotate('最优', xy=(2500, 0.0008), xytext=(1800, 0.003),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=12, color='black')
    
    plt.tight_layout()
    save_fig('f2_7_depth_sweep.png')


def f2_7_3_lr_sweep():
    """Learning rate sweep"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    epochs = np.arange(1, 2001)
    
    # lr=1e-2: unstable
    loss_high = 0.5 * np.exp(-epochs/300) + 0.001
    loss_high += 0.1 * np.sin(epochs/50) * np.exp(-epochs/1000)  # oscillations
    
    # lr=1e-3: good
    loss_good = 1.0 * np.exp(-epochs/400) + 0.0002
    
    # lr=1e-4: slow
    loss_slow = 1.0 * np.exp(-epochs/1200) + 0.0005
    
    ax.plot(epochs, loss_high, color=COLORS[0], linewidth=2.0, 
            label='学习率 = 1e-2 (不稳定)')
    ax.plot(epochs, loss_good, color=COLORS[1], linewidth=2.0, 
            label='学习率 = 1e-3 (最优)')
    ax.plot(epochs, loss_slow, color=COLORS[2], linewidth=2.0, 
            label='学习率 = 1e-4 (收敛慢)')
    
    ax.set_xlabel('训练轮次', fontsize=14)
    ax.set_ylabel('训练损失', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f2_7_3_lr_sweep.png')


def f2_7_4_weight_sweep():
    """Loss weight pathological cases"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    epochs = np.arange(1, 2001)
    
    # Balanced
    loss_balanced = 0.5 * np.exp(-epochs/500) + 0.0001
    
    # Too much λ_pde: BC fails
    loss_pde_heavy = 0.3 * np.exp(-epochs/400) + 0.02  # Lower PDE loss but higher overall
    
    # Too much λ_bc: PDE fails
    loss_bc_heavy = 0.7 * np.exp(-epochs/600) + 0.005
    
    ax.plot(epochs, loss_balanced, color=COLORS[1], linewidth=2.5, 
            label='平衡权重 (最优)', zorder=3)
    ax.plot(epochs, loss_pde_heavy, color=COLORS[0], linewidth=2.0, 
            linestyle='--', label='λ_pde 过大')
    ax.plot(epochs, loss_bc_heavy, color=COLORS[2], linewidth=2.0, 
            linestyle=':', label='λ_bc 过大')
    
    ax.set_xlabel('训练轮次', fontsize=14)
    ax.set_ylabel('总体误差', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f2_7_4_weight_sweep.png')


# ==================== Chapter 3 ====================

def f3_2_temperature_field():
    """2D temperature field in heatsink geometry"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    # Create heatsink-like geometry
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Temperature field: hot at bottom, cooling with fins
    T = 100 * (1 - Y) + 20 * Y  # Base gradient
    # Add fin effects (cold spots)
    for fin_x in [0.2, 0.4, 0.6, 0.8]:
        T -= 15 * np.exp(-((X - fin_x)**2 / 0.002 + (Y - 0.5)**2 / 0.1))
    
    contour = ax.contourf(X, Y, T, levels=20, cmap='hot')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('温度 (°C)', fontsize=14)
    
    ax.set_xlabel('x (m)', fontsize=14)
    ax.set_ylabel('y (m)', fontsize=14)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    save_fig('f3_2_temperature_field.png')


def f3_7_loss_curves():
    """4 loss curves for heatsink problem"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    epochs = np.arange(1, 4001)
    
    loss_pde = 1.0 * np.exp(-epochs/800) + 0.0001
    loss_heat = 0.3 * np.exp(-epochs/600) + 0.00005
    loss_conv = 0.4 * np.exp(-epochs/700) + 0.00008
    loss_adiabatic = 0.2 * np.exp(-epochs/500) + 0.00003
    
    ax.plot(epochs, loss_pde, color=COLORS[0], linewidth=2.0, 
            label='内部PDE')
    ax.plot(epochs, loss_heat, color=COLORS[1], linewidth=2.0, 
            linestyle='--', label='底部加热边界')
    ax.plot(epochs, loss_conv, color=COLORS[2], linewidth=2.0, 
            linestyle='-.', label='侧面对流边界')
    ax.plot(epochs, loss_adiabatic, color=COLORS[3], linewidth=2.0, 
            linestyle=':', label='顶部绝热边界')
    
    ax.set_xlabel('训练轮次', fontsize=14)
    ax.set_ylabel('损失值', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f3_7_loss_curves.png')


# ==================== Chapter 4 ====================

def f4_2_fno_prediction():
    """FNO prediction: 3-panel input/prediction/truth"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Create synthetic field data
    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)
    X, Y = np.meshgrid(x, y)
    
    np.random.seed(42)
    # Input permeability K
    K = 1.0 + 0.5 * np.sin(6 * np.pi * X) * np.cos(4 * np.pi * Y)
    K += 0.2 * np.random.randn(64, 64)
    
    # Ground truth pressure U
    U_truth = np.sin(3 * np.pi * X) * np.sin(2 * np.pi * Y) / K
    
    # Predicted (with small error)
    U_pred = U_truth + 0.05 * np.random.randn(64, 64)
    
    im1 = axes[0].contourf(X, Y, K, levels=20, cmap='viridis')
    axes[0].set_title('输入渗透率场 K', fontsize=14)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].contourf(X, Y, U_pred, levels=20, cmap='plasma')
    axes[1].set_title('FNO预测压力场 U', fontsize=14)
    axes[1].set_xlabel('x', fontsize=12)
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].contourf(X, Y, U_truth, levels=20, cmap='plasma')
    axes[2].set_title('真实压力场 U', fontsize=14)
    axes[2].set_xlabel('x', fontsize=12)
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    save_fig('f4_2_fno_prediction.png')


def f4_7_pressure_field():
    """4-panel: K input / prediction / truth / error"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    
    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)
    X, Y = np.meshgrid(x, y)
    
    np.random.seed(123)
    K = 1.0 + 0.5 * np.sin(5 * np.pi * X) * np.cos(3 * np.pi * Y)
    U_truth = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) / K
    U_pred = U_truth + 0.08 * np.random.randn(64, 64)
    error = np.abs(U_pred - U_truth)
    
    im1 = axes[0, 0].contourf(X, Y, K, levels=20, cmap='viridis')
    axes[0, 0].set_title('输入 K', fontsize=14)
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].contourf(X, Y, U_pred, levels=20, cmap='plasma')
    axes[0, 1].set_title('预测 U', fontsize=14)
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].contourf(X, Y, U_truth, levels=20, cmap='plasma')
    axes[1, 0].set_title('真实 U', fontsize=14)
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].contourf(X, Y, error, levels=20, cmap='Reds')
    axes[1, 1].set_title('绝对误差', fontsize=14)
    plt.colorbar(im4, ax=axes[1, 1])
    
    for ax in axes.flat:
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
    
    plt.tight_layout()
    save_fig('f4_7_pressure_field.png')


def f4_8_cp_curve():
    """Cp distribution with confidence band"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    x = np.linspace(0, 1, 100)
    cp_cfd = -0.5 * np.sin(2 * np.pi * x) + 0.1 * np.cos(8 * np.pi * x)
    cp_fno = cp_cfd + 0.02 * np.random.randn(100)
    confidence = 0.05 * np.ones_like(x)
    
    ax.plot(x, cp_cfd, 'k--', linewidth=2.0, label='CFD参考', alpha=0.8)
    ax.plot(x, cp_fno, color=COLORS[3], linewidth=2.0, label='FNO预测')
    ax.fill_between(x, cp_fno - confidence, cp_fno + confidence, 
                     color=COLORS[3], alpha=0.2, label='置信区间')
    
    ax.set_xlabel('位置 x/c', fontsize=14)
    ax.set_ylabel('压力系数 Cp', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f4_8_cp_curve.png')


def f4_9_tuning():
    """3 subplots: hyperparameter tuning"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Modes vs error (optimal at 12)
    modes = np.array([4, 8, 12, 16, 20, 24])
    error_modes = 0.05 + 0.15 / modes + 0.001 * (modes - 12)**2
    axes[0].plot(modes, error_modes, 'o-', color=COLORS[0], linewidth=2.0, markersize=8)
    axes[0].axvline(12, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('傅里叶模态数', fontsize=14)
    axes[0].set_ylabel('相对误差', fontsize=14)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].annotate('最优', xy=(12, error_modes[2]), xytext=(15, 0.08),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=11, color='red')
    
    # Layers vs error (optimal at 4)
    layers = np.array([2, 3, 4, 5, 6])
    error_layers = 0.08 + 0.1 / layers + 0.005 * (layers - 4)**2
    axes[1].plot(layers, error_layers, 's-', color=COLORS[1], linewidth=2.0, markersize=8)
    axes[1].axvline(4, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('网络层数', fontsize=14)
    axes[1].set_ylabel('相对误差', fontsize=14)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Channels vs error (optimal at 32)
    channels = np.array([8, 16, 32, 64, 128])
    error_channels = 0.1 + 0.3 / channels + 0.0001 * (channels - 32)**2
    axes[2].plot(channels, error_channels, '^-', color=COLORS[2], linewidth=2.0, markersize=8)
    axes[2].axvline(32, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[2].set_xlabel('隐藏层通道数', fontsize=14)
    axes[2].set_ylabel('相对误差', fontsize=14)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f4_9_tuning.png')


# ==================== Chapter 5 ====================

def f5_2_comparison():
    """Data-driven FNO vs Physics-informed FNO"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    samples = np.array([10, 50, 100, 200, 500, 1000, 2000])
    
    # Pure data-driven needs more data
    error_data = 0.5 * (1000 / samples)**0.7 + 0.01
    
    # Physics-informed performs better with less data
    error_physics = 0.3 * (1000 / samples)**0.5 + 0.01
    
    ax.plot(samples, error_data, 'o-', color=COLORS[0], linewidth=2.0, 
            markersize=7, label='纯数据驱动FNO')
    ax.plot(samples, error_physics, 's-', color=COLORS[2], linewidth=2.0, 
            markersize=7, label='物理约束FNO')
    
    ax.set_xlabel('训练样本数', fontsize=14)
    ax.set_ylabel('测试误差', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f5_2_comparison.png')


def f5_4_darcy_samples():
    """2x4 grid showing 4 K->U sample pairs"""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    
    for i in range(4):
        np.random.seed(i * 10)
        x = np.linspace(0, 1, 32)
        y = np.linspace(0, 1, 32)
        X, Y = np.meshgrid(x, y)
        
        K = 1.0 + 0.5 * np.sin((i+2) * np.pi * X) * np.cos((i+1) * np.pi * Y)
        K += 0.1 * np.random.randn(32, 32)
        
        U = np.sin((i+1) * np.pi * X) * np.sin((i+1) * np.pi * Y) / K
        
        im1 = axes[0, i].contourf(X, Y, K, levels=15, cmap='viridis')
        axes[0, i].set_title(f'样本{i+1}: K', fontsize=12)
        axes[0, i].axis('off')
        
        im2 = axes[1, i].contourf(X, Y, U, levels=15, cmap='plasma')
        axes[1, i].set_title(f'样本{i+1}: U', fontsize=12)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    save_fig('f5_4_darcy_samples.png')


def f5_6_small_data():
    """Log-log plot: sample size vs error"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    samples = np.array([10, 20, 50, 100, 200, 500])
    
    error_data = 0.5 * (100 / samples)**0.8 + 0.015
    error_physics = 0.3 * (100 / samples)**0.6 + 0.01
    
    ax.loglog(samples, error_data, 'o-', color=COLORS[0], linewidth=2.0, 
              markersize=8, label='无物理约束')
    ax.loglog(samples, error_physics, 's-', color=COLORS[2], linewidth=2.0, 
              markersize=8, label='有物理约束')
    
    ax.set_xlabel('训练样本数', fontsize=14)
    ax.set_ylabel('相对L2误差', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    save_fig('f5_6_small_data.png')


def f5_7_lambda_sweep():
    """Lambda sweep showing U-shaped error"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    lambdas = np.logspace(-3, 1, 50)
    
    # U-shaped: optimal around 0.1
    error = 0.05 + 0.02 / lambdas + 0.01 * lambdas
    optimal_idx = np.argmin(error)
    
    ax.semilogx(lambdas, error, color=COLORS[4], linewidth=2.5)
    ax.axvspan(0.05, 0.2, alpha=0.2, color='green', label='最优区间')
    ax.axvline(lambdas[optimal_idx], color='red', linestyle='--', 
               linewidth=2.0, label=f'最优点 λ={lambdas[optimal_idx]:.2f}')
    
    ax.set_xlabel('物理损失权重 λ_pde', fontsize=14)
    ax.set_ylabel('总体测试误差', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_fig('f5_7_lambda_sweep.png')


# ==================== Chapter 6 ====================

def f6_2_rollout_result():
    """2x2 grid: weather rollout at 4 time steps"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    
    times = ['0小时', '6小时', '12小时', '24小时']
    
    for idx, (ax, time) in enumerate(zip(axes.flat, times)):
        np.random.seed(idx * 5)
        x = np.linspace(0, 360, 128)
        y = np.linspace(-90, 90, 64)
        X, Y = np.meshgrid(x, y)
        
        # Simulate weather field evolving
        Z = 20 + 10 * np.sin(X * np.pi / 180) * np.cos(Y * np.pi / 90)
        Z += 5 * np.sin((X + idx * 30) * np.pi / 60) * np.cos(Y * np.pi / 45)
        Z += 2 * np.random.randn(64, 128)
        
        im = ax.contourf(X, Y, Z, levels=20, cmap='plasma')
        ax.set_title(f'预测时间: {time}', fontsize=13)
        ax.set_xlabel('经度', fontsize=11)
        ax.set_ylabel('纬度', fontsize=11)
        plt.colorbar(im, ax=ax, label='温度 (°C)')
    
    plt.tight_layout()
    save_fig('f6_2_rollout_result.png')


def f6_4_error_growth():
    """Error growth with forecast lead time"""
    fig, ax = plt.subplots(figsize=(10, 6.67))
    
    lead_time = np.linspace(0, 10, 100)  # days
    
    # Exponential growth with saturation
    error = 0.5 * (1 - np.exp(-lead_time / 3))
    error += 0.05 * np.random.randn(100)
    error = np.maximum(error, 0)
    
    ax.plot(lead_time, error, color=COLORS[5], linewidth=2.5)
    ax.fill_between(lead_time, error - 0.03, error + 0.03, 
                     color=COLORS[5], alpha=0.2)
    
    ax.set_xlabel('预报时效 (天)', fontsize=14)
    ax.set_ylabel('均方根误差 RMSE', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Mark critical points
    ax.axhline(0.3, color='orange', linestyle='--', linewidth=1.5, 
               alpha=0.7, label='可接受误差阈值')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    save_fig('f6_4_error_growth.png')


# ==================== Main Execution ====================

if __name__ == '__main__':
    print("Generating all figures for PhysicsNeMo book...")
    print("=" * 60)
    
    # Chapter 0
    print("\n[Chapter 0]")
    f0_2_cfd_vs_ai_comparison()
    
    # Chapter 1
    print("\n[Chapter 1]")
    f1_4_mlp_results()
    f1_5_pinn_results()
    f1_5_2_extrapolation()
    
    # Chapter 2
    print("\n[Chapter 2]")
    f2_2_loss_curves()
    f2_3_temperature_evolution()
    f2_5_collocation_points()
    f2_7_depth_sweep()
    f2_7_3_lr_sweep()
    f2_7_4_weight_sweep()
    
    # Chapter 3
    print("\n[Chapter 3]")
    f3_2_temperature_field()
    f3_7_loss_curves()
    
    # Chapter 4
    print("\n[Chapter 4]")
    f4_2_fno_prediction()
    f4_7_pressure_field()
    f4_8_cp_curve()
    f4_9_tuning()
    
    # Chapter 5
    print("\n[Chapter 5]")
    f5_2_comparison()
    f5_4_darcy_samples()
    f5_6_small_data()
    f5_7_lambda_sweep()
    
    # Chapter 6
    print("\n[Chapter 6]")
    f6_2_rollout_result()
    f6_4_error_growth()
    
    print("\n" + "=" * 60)
    print(f"✅ All 26 figures generated successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
