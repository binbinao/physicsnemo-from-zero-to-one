"""Ensure ch04_fno_airfoil is present before ch05 training scripts run."""
from __future__ import annotations

import sys
from pathlib import Path


def require_ch04() -> None:
    ch04_model = Path(__file__).resolve().parent.parent / "ch04_fno_airfoil" / "fno_model.py"
    if ch04_model.is_file():
        return
    print(
        "错误：找不到 ch04_fno_airfoil/fno_model.py\n"
        "第 5 章依赖第 4 章代码。请 clone 完整仓库，且不要删除 ch04 目录。\n"
        "详见 ch05_darcy_hybrid/README.md",
        file=sys.stderr,
    )
    sys.exit(1)
