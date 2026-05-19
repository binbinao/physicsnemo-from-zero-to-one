#!/usr/bin/env python3
"""PhysicsNeMo 教程 — 环境自检脚本

运行方式：
    python scripts/check_env.py              # 基础检查 + Tier 概览
    python scripts/check_env.py --chapter 4  # 按章节检查
    python scripts/check_env.py --tier 2     # 按依赖档位检查
    python scripts/check_env.py --list       # 列出章节要求

详见 docs/ENVIRONMENT.md
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WIDTH = 60

# tier -> pip one-liner (display only)
TIER_INSTALL = {
    0: 'pip install "torch>=2.3" numpy matplotlib',
    1: 'pip install "torch>=2.3" numpy matplotlib',
    2: (
        'pip install "torch>=2.3" numpy matplotlib hydra-core '
        "nvidia-physicsnemo nvidia-physicsnemo.sym"
    ),
    3: (
        'pip install "torch>=2.3" numpy matplotlib hydra-core '
        "nvidia-physicsnemo nvidia-physicsnemo.sym "
        "optuna fastapi uvicorn onnx"
    ),
}

# chapter -> requirements
CHAPTERS: dict[int, dict] = {
    1: {
        "title": "Hello PINN",
        "tier": 0,
        "gpu": False,
        "required": ["torch", "numpy", "matplotlib"],
        "optional": [],
        "sdk": ["physicsnemo", "physicsnemo.sym", "hydra"],
        "entry": "cd ch01_hello && python pinn_spring.py --epochs 1000",
    },
    2: {
        "title": "1D 热传导",
        "tier": 1,
        "gpu": False,
        "required": ["torch", "numpy", "matplotlib"],
        "optional": ["hydra"],
        "sdk": ["physicsnemo", "physicsnemo.sym"],
        "entry": "cd ch02_heat1d && python heat1d_pinn_raw.py",
    },
    3: {
        "title": "2D 散热片",
        "tier": 1,
        "gpu": True,
        "required": ["torch", "numpy", "matplotlib"],
        "optional": ["hydra"],
        "sdk": ["physicsnemo", "physicsnemo.sym"],
        "entry": "cd ch03_heatsink && python heat_sink_train.py",
    },
    4: {
        "title": "FNO / Darcy",
        "tier": 1,
        "gpu": True,
        "required": ["torch", "numpy", "matplotlib"],
        "optional": ["hydra"],
        "sdk": ["physicsnemo"],
        "entry": "cd ch04_fno_airfoil && python train_fno_mini.py --epochs 30",
    },
    5: {
        "title": "Physics-Informed FNO",
        "tier": 1,
        "gpu": True,
        "required": ["torch", "numpy", "matplotlib", "ch04_dir"],
        "optional": ["hydra"],
        "sdk": ["physicsnemo"],
        "entry": "cd ch05_darcy_hybrid && python train_data_fno.py --epochs 50",
    },
    6: {
        "title": "FourCastNet mini",
        "tier": 1,
        "gpu": True,
        "required": ["torch", "numpy", "matplotlib"],
        "optional": ["hydra"],
        "sdk": ["physicsnemo"],
        "entry": "cd ch06_fourcastnet_mini && python train_afno_mini.py --epochs 20",
    },
    7: {
        "title": "端到端流水线",
        "tier": 1,
        "gpu": True,
        "required": ["torch", "numpy", "matplotlib"],
        "optional": ["hydra", "optuna", "fastapi", "uvicorn", "onnx"],
        "sdk": ["physicsnemo"],
        "entry": "cd ch07_drivaernet_optim && python train.py --epochs 100",
    },
}

PKG_ALIASES = {
    "torch": "torch",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "hydra": "hydra",
    "physicsnemo": "physicsnemo",
    "physicsnemo.sym": "physicsnemo.sym",
    "optuna": "optuna",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "onnx": "onnx",
}


def banner(msg: str) -> None:
    print("=" * WIDTH)
    print(msg)
    print("=" * WIDTH)


def check_line(label: str, ok: bool, detail: str = "") -> bool:
    icon = "✅" if ok else "❌"
    line = f"{icon} {label}"
    if detail:
        line += f": {detail}"
    print(line)
    return ok


def import_ok(module: str) -> tuple[bool, str]:
    if module == "ch04_dir":
        path = REPO_ROOT / "ch04_fno_airfoil" / "fno_model.py"
        return path.is_file(), str(path.relative_to(REPO_ROOT))
    mod = PKG_ALIASES.get(module, module)
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "ok")
        return True, str(ver)[:40]
    except ImportError:
        return False, f"pip install（见 Tier 说明）"


def probe_torch() -> dict:
    info = {"ok": False, "version": "", "cuda": False, "device": ""}
    try:
        import torch

        info["ok"] = True
        info["version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = True
            props = torch.cuda.get_device_properties(0)
            info["device"] = f"{torch.cuda.get_device_name(0)} ({props.total_mem / 1e9:.1f}GB)"
    except ImportError:
        pass
    return info


def run_base_checks() -> bool:
    """Core stack used in every mode."""
    all_ok = True
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    all_ok &= check_line("Python >= 3.10", sys.version_info >= (3, 10), py)

    torch_info = probe_torch()
    if torch_info["ok"]:
        check_line("PyTorch", True, torch_info["version"])
        if torch_info["cuda"]:
            check_line("CUDA", True, torch_info["device"])
        else:
            check_line("CUDA", False, "CPU 模式（ch1-2 可跑；ch4+ 建议 GPU）")
    else:
        all_ok &= check_line("PyTorch", False, "pip install 'torch>=2.3'")

    for pkg in ("numpy", "matplotlib"):
        ok, detail = import_ok(pkg)
        all_ok &= check_line(pkg.capitalize(), ok, detail if ok else f"pip install {pkg}")

    return all_ok


def check_packages(names: list[str], label: str, required: bool) -> bool:
    if not names:
        return True
    print(f"\n── {label} ──")
    ok_all = True
    for name in names:
        ok, detail = import_ok(name)
        if name == "ch04_dir" and not ok:
            detail = "需要完整仓库中的 ch04_fno_airfoil/"
        if required:
            ok_all &= check_line(name, ok, detail)
        else:
            check_line(name, ok, detail if ok else "未安装（本章节可选）")
    return ok_all


def print_tier_summary() -> None:
    print("\n── 依赖档位 Tier ──")
    for tier, cmd in sorted(TIER_INSTALL.items()):
        print(f"  Tier {tier}: {cmd}")


def print_chapter_summary(ch: int) -> bool:
    meta = CHAPTERS[ch]
    banner(f"第 {ch} 章 · {meta['title']}")
    print(f"最低 Tier: {meta['tier']}  |  建议 GPU: {'是' if meta['gpu'] else '否'}")
    print(f"安装参考: {TIER_INSTALL[meta['tier']]}")

    base_ok = run_base_checks()
    req_ok = check_packages(meta["required"], "本章必需", required=True)
    check_packages(meta.get("optional", []), "本章可选（完整流程）", required=False)
    check_packages(meta.get("sdk", []), "SDK 版 (*_sdk.py) 额外需要", required=False)

    if meta["gpu"]:
        torch_info = probe_torch()
        if torch_info["ok"] and not torch_info["cuda"]:
            print("\n⚠️  本章建议在 GPU / Colab 上运行，见 docs/CLOUD_GPU_GUIDE.md")

    print("=" * WIDTH)
    ok = base_ok and req_ok
    if ok:
        print(f"✅ 第 {ch} 章环境就绪。建议入口：\n   {meta['entry']}")
    else:
        print(f"❌ 第 {ch} 章仍有缺失项；见 docs/ENVIRONMENT.md")
    return ok


def check_tier(tier: int) -> bool:
    banner(f"Tier {tier} 依赖检查")
    print(f"推荐安装:\n  {TIER_INSTALL.get(tier, '?')}\n")
    base_ok = run_base_checks()
    extra = []
    if tier >= 2:
        extra.extend(["hydra", "physicsnemo", "physicsnemo.sym"])
    if tier >= 3:
        extra.extend(["optuna", "fastapi", "uvicorn", "onnx"])
    pkg_ok = check_packages(extra, f"Tier {tier} 额外包", required=(tier >= 2))
    if tier >= 2:
        torch_info = probe_torch()
        if torch_info["ok"] and not torch_info["cuda"]:
            print("\n⚠️  Tier 2+ 通常需要 CUDA GPU")
    print("=" * WIDTH)
    ok = base_ok and (pkg_ok if tier >= 2 else True)
    print("✅ Tier 满足" if ok else "❌ 请安装缺失包后重试")
    return ok


def list_chapters() -> None:
    banner("各章环境要求速查")
    print(f"{'章':>4}  {'Tier':>4}  {'GPU':>4}  标题")
    print("-" * WIDTH)
    for ch, meta in sorted(CHAPTERS.items()):
        gpu = "建议" if meta["gpu"] else "可选"
        print(f"ch{ch:02d}  {meta['tier']:>4}  {gpu:>4}  {meta['title']}")
    print("\n用法: python scripts/check_env.py --chapter N")
    print("文档: docs/ENVIRONMENT.md · docs/HARDWARE_EXPECTATIONS.md")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PhysicsNeMo 教程环境自检")
    p.add_argument(
        "--chapter",
        type=int,
        choices=sorted(CHAPTERS.keys()),
        metavar="N",
        help="检查第 N 章（1-7）所需依赖",
    )
    p.add_argument(
        "--tier",
        type=int,
        choices=sorted(TIER_INSTALL.keys()),
        metavar="N",
        help="检查 Tier 0-3 依赖档位",
    )
    p.add_argument("--list", action="store_true", help="列出各章 Tier / GPU 要求")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        list_chapters()
        return 0

    if args.chapter is not None:
        ok = print_chapter_summary(args.chapter)
        print("\n📖 docs/ENVIRONMENT.md · docs/COMMAND_REFERENCE.md")
        return 0 if ok else 1

    if args.tier is not None:
        ok = check_tier(args.tier)
        return 0 if ok else 1

    # Default: base + overview
    banner("PhysicsNeMo 教程 - 环境自检（全局）")
    base_ok = run_base_checks()
    check_packages(["physicsnemo", "physicsnemo.sym"], "SDK（全书可选）", required=False)
    check_packages(["hydra"], "Hydra（ch2/4/5/6 等可选）", required=False)
    check_packages(["optuna", "fastapi"], "ch7 完整流水线（可选）", required=False)
    print_tier_summary()
    print("=" * WIDTH)
    print("📖 docs/ENVIRONMENT.md")
    print("📖 docs/QUICKSTART_DAY1.md（第 1 天仅需 Tier 0）")
    print("💡 python scripts/check_env.py --chapter 4  # 按章检查")
    print("💡 python scripts/check_env.py --list")
    print("=" * WIDTH)
    if base_ok:
        print("🎉 核心依赖通过！ch01 入口：")
        print("   cd ch01_hello && python pinn_spring.py --epochs 1000")
    else:
        print("⚠️  请先安装缺失的核心依赖。")
    print()
    return 0 if base_ok else 1


if __name__ == "__main__":
    sys.exit(main())
