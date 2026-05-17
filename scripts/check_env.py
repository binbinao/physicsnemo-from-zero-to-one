#!/usr/bin/env python3
"""PhysicsNeMo 教程 — 环境自检脚本

运行方式：
    python scripts/check_env.py

检查 Python / PyTorch / CUDA / PhysicsNeMo 是否就绪。
"""
import sys
import importlib

WIDTH = 60

def banner(msg):
    print("=" * WIDTH)
    print(msg)
    print("=" * WIDTH)

def check(label, ok, detail=""):
    icon = "✅" if ok else "❌"
    line = f"{icon} {label}"
    if detail:
        line += f": {detail}"
    print(line)
    return ok

def main():
    banner("PhysicsNeMo 教程 - 环境自检")
    all_ok = True

    # Python 版本
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    all_ok &= check("Python", sys.version_info >= (3, 10), py)

    # PyTorch
    try:
        import torch
        tv = torch.__version__
        all_ok &= check("PyTorch", True, tv)
    except ImportError:
        all_ok &= check("PyTorch", False, "未安装 → pip install 'torch>=2.3'")
        torch = None

    # CUDA
    if torch is not None:
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            mem = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB"
            check("CUDA available", True, f"{dev} (显存 {mem})")
        else:
            check("CUDA available", False, "CPU 模式（ch01-ch02 可跑，后续章节建议 GPU）")

    # PhysicsNeMo (optional)
    try:
        pm = importlib.import_module("physicsnemo")
        ver = getattr(pm, "__version__", "unknown")
        check("PhysicsNeMo", True, ver)
    except ImportError:
        check("PhysicsNeMo", False, "未安装（ch01-ch03 裸 PyTorch 版可跑，SDK 版需要）")

    # PhysicsNeMo-Sym (optional)
    try:
        importlib.import_module("physicsnemo.sym")
        check("PhysicsNeMo-Sym", True, "已安装")
    except (ImportError, ModuleNotFoundError):
        check("PhysicsNeMo-Sym", False, "未安装（SDK 版需要，裸 PyTorch 版不需要）")

    # numpy / matplotlib
    for pkg, name in [("numpy", "NumPy"), ("matplotlib", "Matplotlib")]:
        try:
            m = importlib.import_module(pkg)
            v = getattr(m, "__version__", "")
            all_ok &= check(name, True, v)
        except ImportError:
            all_ok &= check(name, False, f"pip install {pkg}")

    print("=" * WIDTH)
    if all_ok:
        print("🎉 核心依赖全部通过！可以开始第 1 章了：")
        print("   cd ch01_hello && python mlp_spring.py --epochs 1000")
    else:
        print("⚠️  部分依赖缺失，请按上方提示安装后重试。")
    print()

if __name__ == "__main__":
    main()
