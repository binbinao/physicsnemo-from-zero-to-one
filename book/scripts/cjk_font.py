"""Register bundled Noto Sans SC for Matplotlib book figure scripts."""

from __future__ import annotations

from pathlib import Path

FONT_PATH = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NotoSansSC-Regular.otf"


def configure_matplotlib_cjk() -> str:
    """Load bundled CJK font; return family name for rcParams."""
    if not FONT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing CJK font: {FONT_PATH}\n"
            "See book/assets/fonts/README.md for download instructions."
        )

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return family
