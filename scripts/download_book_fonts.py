#!/usr/bin/env python3
"""Download bundled Noto Sans SC for book figure generation."""

from __future__ import annotations

import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "book" / "assets" / "fonts" / "NotoSansSC-Regular.otf"
URL = (
    "https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/"
    "Sans/SubsetOTF/SC/NotoSansSC-Regular.otf"
)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading → {OUT}")
    urllib.request.urlretrieve(URL, OUT)
    print(f"Done ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
