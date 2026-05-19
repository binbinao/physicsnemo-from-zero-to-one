#!/usr/bin/env python3
"""Generate conceptual / roadmap / banner figures for the book."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cjk_font import configure_matplotlib_cjk

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "book" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

configure_matplotlib_cjk()
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)
C = plt.cm.tab10.colors


def save(name: str) -> None:
    plt.savefig(OUT / name, facecolor="white")
    plt.close()
    print(f"  ✓ {name}")


def banner(title: str, subtitle: str, name: str, color=C[0]) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.1), 0.96, 0.8, boxstyle="round,pad=0.02", fc=color, ec="black", alpha=0.15))
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(0.5, 0.28, subtitle, ha="center", va="center", fontsize=12, color="#333")
    save(name)


def h_flow(labels: list[str], name: str, title: str = "") -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, n * 2.2), 3.5))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.text(n / 2, 0.92, title, ha="center", fontsize=14, fontweight="bold")
    for i, lab in enumerate(labels):
        x = i + 0.5
        ax.add_patch(mpatches.FancyBboxPatch((x - 0.38, 0.25), 0.76, 0.45, boxstyle="round,pad=0.02", fc=C[i % 10], ec="black", alpha=0.35))
        ax.text(x, 0.47, lab, ha="center", va="center", fontsize=10, wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(i + 0.62, 0.47), xytext=(i + 0.88, 0.47), arrowprops=dict(arrowstyle="->", lw=2))
    save(name)


def timeline(years: list[str], events: list[str], name: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(-0.5, len(years) - 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0, len(years) - 1], [0.5, 0.5], "k-", lw=2)
    for i, (y, e) in enumerate(zip(years, events)):
        ax.plot(i, 0.5, "o", ms=12, color=C[i % 10])
        ax.text(i, 0.62, y, ha="center", fontsize=11, fontweight="bold")
        ax.text(i, 0.28, e, ha="center", fontsize=9, wrap=True)
    save(name)


def v_blocks(items: list[tuple[str, str]], name: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, len(items) * 0.9)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(items) + 0.5)
    ax.axis("off")
    if title:
        ax.text(0.5, len(items) + 0.35, title, ha="center", fontsize=14, fontweight="bold")
    for i, (left, right) in enumerate(items):
        y = len(items) - i - 1
        ax.add_patch(mpatches.FancyBboxPatch((0.05, y + 0.08), 0.4, 0.55, boxstyle="round", fc=C[0], alpha=0.25, ec="black"))
        ax.add_patch(mpatches.FancyBboxPatch((0.55, y + 0.08), 0.4, 0.55, boxstyle="round", fc=C[1], alpha=0.25, ec="black"))
        ax.text(0.25, y + 0.35, left, ha="center", va="center", fontsize=10)
        ax.text(0.75, y + 0.35, right, ha="center", va="center", fontsize=10)
        if i < len(items) - 1:
            ax.annotate("", xy=(0.5, y + 0.05), xytext=(0.5, y - 0.05), arrowprops=dict(arrowstyle="->", lw=1.5))
    save(name)


def spring_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.plot([1, 9], [2, 2], "k-", lw=2)
    for x in np.linspace(1.5, 8, 12):
        ax.plot([x, x + 0.3], [1.7, 2.3], "b-", lw=1.5)
    ax.add_patch(mpatches.Rectangle((0.2, 1.5), 0.8, 1, fc="gray", ec="black"))
    ax.text(0.6, 2, "m", ha="center", fontsize=12)
    ax.annotate("F = -kx", xy=(5, 2.5), fontsize=14)
    ax.text(5, 0.5, "胡克定律 + 牛顿第二定律", ha="center", fontsize=11)
    save("f1_3_spring_diagram.png")


def heatsink_geom() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal")
    ax.add_patch(mpatches.Rectangle((2, 1), 4, 1, fc="red", alpha=0.4, label="芯片"))
    for i, x in enumerate([1, 2.5, 4]):
        ax.add_patch(mpatches.Rectangle((x, 2), 0.8, 2.5, fc="steelblue", alpha=0.5))
        ax.text(x + 0.4, 3.2, f"鳍片{i+1}", ha="center", fontsize=9)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)
    ax.set_title("Three-fin heat sink（示意）")
    ax.legend(loc="upper right")
    save("f3_3_geometry.png")


def csg_union() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title in zip(axes, ["矩形 A", "矩形 B", "A ∪ B"]):
        ax.set_aspect("equal")
        ax.add_patch(mpatches.Rectangle((0, 0), 2, 2, fc=C[0], alpha=0.4))
        ax.add_patch(mpatches.Rectangle((1, 1), 2, 2, fc=C[1], alpha=0.4))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_title(title)
        ax.axis("off")
    save("f3_4_csg.png")


def triangle_relation(name: str, labels: tuple[str, str, str], center: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal")
    ax.axis("off")
    pts = np.array([[0.5, 0.9], [0.1, 0.2], [0.9, 0.2]])
    tri = plt.Polygon(pts, fill=False, edgecolor="black", lw=2)
    ax.add_patch(tri)
    for p, lab in zip(pts, labels):
        ax.text(p[0], p[1] + 0.05, lab, ha="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.45, center, ha="center", fontsize=11, bbox=dict(boxstyle="round", fc="wheat"))
    save(name)


def fno_block() -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    boxes = ["输入 u(x)", "FFT", "频域卷积", "iFFT", "输出 u'(x)"]
    for i, b in enumerate(boxes):
        x = i * 2.2
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.3), 1.8, 0.5, boxstyle="round", fc=C[i % 5], alpha=0.35, ec="black"))
        ax.text(x + 0.9, 0.55, b, ha="center", va="center", fontsize=10)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.95, 0.55), xytext=(x + 2.05, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    save("f4_5_fno_block.png")


def operator_mapping() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.text(0.15, 0.6, "输入函数\na(x)", ha="center", fontsize=12, bbox=dict(boxstyle="round", fc=C[2], alpha=0.3))
    ax.text(0.5, 0.6, "神经算子 G", ha="center", fontsize=14, fontweight="bold", bbox=dict(boxstyle="round", fc=C[4], alpha=0.3))
    ax.text(0.85, 0.6, "输出函数\nu(x)", ha="center", fontsize=12, bbox=dict(boxstyle="round", fc=C[1], alpha=0.3))
    ax.annotate("", xy=(0.35, 0.6), xytext=(0.22, 0.6), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.78, 0.6), xytext=(0.58, 0.6), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.5, 0.2, "一个模型 → 一族工况", ha="center", fontsize=11, style="italic")
    save("f4_4_neural_operator.png")


def heat_intuition() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 1, 100)
    for t, alpha in zip([0, 0.1, 0.3], [0.9, 0.5, 0.2]):
        u = np.exp(-((x - 0.5) ** 2) / (0.01 + t))
        ax.plot(x, u, lw=2, label=f"t={t}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title("热扩散：尖峰逐渐展宽")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save("f2_4_physics_intuition.png")


def wechat_placeholder() -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(mpatches.Rectangle((0.1, 0.1), 0.8, 0.8, fc="white", ec="black"))
    rng = np.random.default_rng(0)
    for _ in range(40):
        ax.plot(rng.random(), rng.random(), "s", color="black", ms=rng.integers(2, 8))
    ax.text(0.5, 0.02, "公众号占位", ha="center", fontsize=8)
    root_assets = REPO_ROOT / "assets"
    root_assets.mkdir(exist_ok=True)
    plt.savefig(root_assets / "wechat_qrcode.png", facecolor="white")
    plt.close()
    print("  ✓ assets/wechat_qrcode.png")


def main() -> None:
    print("Generating diagram figures →", OUT)
    timeline(
        ["2020", "2021", "2022", "2023", "2024", "2026"],
        ["AlphaFold", "PIML 综述", "Modulus", "GraphCast", "PhysicsNeMo", "v2.0"],
        "f0_1_ai4s_timeline.png",
    )
    h_flow(
        ["ch0 前言", "ch1-3 PINN", "ch4-5 FNO", "ch6 AFNO", "ch7 部署"],
        "f0_4_book_roadmap.png",
        "全书 7 章路线",
    )
    banner("第 1 章", "弹簧振子 · MLP vs PINN", "f1_0_banner.png", C[0])
    h_flow(["钩子", "快速通道", "PINN", "SDK", "行业映射"], "f1_1_roadmap.png", "第 1 章路线图")
    spring_diagram()
    v_blocks(
        [("physicsnemo-sym", "PINN / PDE"), ("physicsnemo", "FNO / 训练")],
        "f1_7_physicsnemo_arch.png",
        "双框架架构",
    )
    h_flow(["2.0 钩子", "PDE", "三件套", "Hydra", "调参"], "f2_1_roadmap.png", "第 2 章路线图")
    heat_intuition()
    banner("第 3 章", "2D 散热片 · 几何与多约束", "f3_0_banner.png", C[2])
    h_flow(["1D 热杆", "2D 散热片", "3D 封装"], "f3_1_progression.png", "维度递进")
    heatsink_geom()
    csg_union()
    h_flow(["几何", "BC 标注", "采样", "训练"], "f3_5_boundary_conditions.png", "边界条件示意")
    h_flow(["Domain", "Constraint", "Solver", "Validator"], "f3_6_domain_constraint.png", "Sym 结构")
    h_flow(["观测温度", "反演 k", "优化设计"], "f3_8_inverse_flow.png", "反问题")
    h_flow(["CAD", "Icepak", "PINN", "优化", "复核"], "f3_9_design_loop.png", "散热设计闭环")
    banner("第 4 章", "FNO · 翼型 / Darcy 代理", "f4_0_banner.png", C[3])
    v_blocks([("ch1-3 sym", "ch4+ main")], "f4_1_framework_switch.png", "框架切换")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].text(0.5, 0.5, "PINN\n单工况", ha="center", fontsize=14, bbox=dict(boxstyle="round", fc=C[0], alpha=0.3))
    axes[1].text(0.5, 0.5, "FNO\n多工况", ha="center", fontsize=14, bbox=dict(boxstyle="round", fc=C[1], alpha=0.3))
    for a in axes:
        a.axis("off")
    save("f4_3_pinn_vs_fno.png")
    h_flow(["原始 CFD", "预处理", "张量数据集", "FNO 训练"], "f4_6_data_pipeline.png", "数据管线")
    fno_block()
    operator_mapping()
    banner("第 5 章", "Darcy · 数据 + 物理混合", "f5_0_banner.png", C[4])
    triangle_relation("f5_1_triangle.png", ("PINN", "FNO", "数据"), "PINO")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(0, 1, 50)
    k = 0.5 + 0.3 * np.sin(4 * np.pi * x)
    ax.fill_between(x, 0, k, alpha=0.4, label="渗透率 a(x)")
    ax2 = ax.twinx()
    ax2.plot(x, -np.gradient(k), "r-", label="压力梯度示意")
    ax.set_title("Darcy 渗流示意")
    ax.legend(loc="upper left")
    save("f5_3_darcy_physics.png")
    industries = ["油气", "电池", "陶瓷", "地质"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(industries, [1, 0.8, 0.6, 0.9], color=[C[i] for i in range(4)])
    ax.set_title("多孔介质应用场景")
    ax.set_ylabel("示意权重")
    save("f5_8_industry_map.png")
    banner("第 6 章", "FourCastNet Mini · 时序预报", "f6_0_banner.png", C[5])
    h_flow(["输入场", "AFNO", "输出场", "自回归"], "f6_1_afno_block.png", "AFNO block")
    h_flow(["t", "t+Δ", "t+2Δ", "..."], "f6_3_autoregressive.png", "自回归 rollout")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(np.random.rand(6, 8), cmap="Blues", aspect="auto")
    ax.set_title("天气 AI 行业应用（示意）")
    ax.set_xticks([])
    ax.set_yticks([])
    save("f6_5_weather_industry.png")
    banner("第 7 章", "DrivAerNet · 训练到部署", "f7_0_banner.png", C[6])
    h_flow(["数据", "训练", "HPO", "优化", "ONNX", "API"], "f7_1_pipeline.png", "端到端 pipeline")
    wechat_placeholder()
    print("Done.")


if __name__ == "__main__":
    main()
