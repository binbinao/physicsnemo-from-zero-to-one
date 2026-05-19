#!/usr/bin/env python3
"""Verify book figure scripts can load bundled CJK font."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "book" / "scripts"))

from cjk_font import FONT_PATH, configure_matplotlib_cjk  # noqa: E402


def main() -> int:
    if not FONT_PATH.is_file():
        print(f"❌ Missing font: {FONT_PATH}")
        print("   Run: python scripts/download_book_fonts.py")
        return 1

    family = configure_matplotlib_cjk()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 1))
    ax.text(0.5, 0.5, "测试综述", ha="center", fontsize=14)
    fig.canvas.draw()
    plt.close(fig)
    print(f"✅ CJK font OK: {family} ({FONT_PATH.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
