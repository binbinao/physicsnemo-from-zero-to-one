#!/usr/bin/env python3
"""Plot training curves produced by scripts/tb_logger.py (no TensorBoard needed).

Every training script writes ``metrics.csv`` next to its TensorBoard event
files, so curves can be reproduced offline with matplotlib only.

Examples
--------
    python scripts/plot_tb_curves.py --logdir outputs/tb
    python scripts/plot_tb_curves.py --logdir outputs/tb/pinn_spring --out loss.png
    python scripts/plot_tb_curves.py --logdir outputs/tb --metrics loss/total,val/mae
    python scripts/plot_tb_curves.py --logdir outputs/tb --smooth 0.6 --linear-y

Use ``tensorboard --logdir <dir>`` instead if the optional ``tensorboard``
package is installed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def find_runs(log_dir: Path) -> list[tuple[str, Path]]:
    """Return [(run_name, metrics.csv path), ...] under ``log_dir``."""
    direct = log_dir / "metrics.csv"
    if direct.is_file():
        return [(log_dir.name or "run", direct)]

    found = sorted(log_dir.rglob("metrics.csv"))
    runs: list[tuple[str, Path]] = []
    for csv_path in found:
        run_dir = csv_path.parent
        try:
            name = str(run_dir.relative_to(log_dir))
        except ValueError:
            name = run_dir.name
        runs.append((name or "run", csv_path))
    return runs


def read_csv(path: Path) -> tuple[list[int], dict[str, list[float]]]:
    steps: list[int] = []
    series: dict[str, list[float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw_step = row.get("step")
            if raw_step in (None, ""):
                continue
            try:
                steps.append(int(float(raw_step)))
            except ValueError:
                continue
            for key, value in row.items():
                if key == "step" or value in (None, ""):
                    continue
                try:
                    series.setdefault(key, []).append(float(value))
                except ValueError:
                    series.setdefault(key, []).append(float("nan"))
    return steps, series


def smooth(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average (alpha in [0, 1]; higher = smoother)."""
    out = []
    acc = None
    for value in values:
        acc = value if acc is None else alpha * acc + (1 - alpha) * value
        out.append(acc)
    return out


def plot(runs: list[tuple[str, Path]], out_path: Path, metrics=None,
         alpha: float | None = None, log_y: bool = True, title: str = "") -> None:
    parsed: list[tuple[str, list[int], dict[str, list[float]]]] = []
    for name, csv_path in runs:
        steps, series = read_csv(csv_path)
        if steps:
            parsed.append((name, steps, series))
    if not parsed:
        raise SystemExit(f"No metrics.csv with data found under {runs[0][1].parent if runs else '.'}")

    all_keys: list[str] = []
    for _, _, series in parsed:
        for key in series:
            if key not in all_keys:
                all_keys.append(key)
    if metrics:
        wanted = [m.strip() for m in metrics.split(",") if m.strip()]
        missing = [m for m in wanted if m not in all_keys]
        if missing:
            print(f"[warn] metrics not found: {', '.join(missing)}")
        all_keys = [m for m in wanted if m in all_keys] or all_keys
    all_keys.sort()

    n = len(all_keys)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.4 * nrows),
                             squeeze=False)
    flat = axes.ravel()

    for idx, key in enumerate(all_keys):
        ax = flat[idx]
        for name, steps, series in parsed:
            values = series.get(key)
            if not values or len(values) != len(steps):
                continue
            y = smooth(values, alpha) if alpha else values
            label = name if len(parsed) > 1 else key
            ax.plot(steps, y, linewidth=1.4, label=label)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        if len(parsed) > 1:
            ax.legend(fontsize=7)

    for idx in range(n, len(flat)):
        flat[idx].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}  (runs={len(parsed)}, metrics={n})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training curves from metrics.csv written by tb_logger")
    parser.add_argument("--logdir", required=True,
                        help="Run dir, or a parent dir containing runs")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: <logdir>/curves.png)")
    parser.add_argument("--metrics", default=None,
                        help="Comma-separated metric names (default: all)")
    parser.add_argument("--smooth", type=float, default=None,
                        help="Exponential smoothing factor in [0, 1)")
    parser.add_argument("--linear-y", action="store_true",
                        help="Use a linear y axis instead of log scale")
    parser.add_argument("--title", default="", help="Figure title")
    args = parser.parse_args()

    log_dir = Path(args.logdir).expanduser().resolve()
    if not log_dir.exists():
        raise SystemExit(f"Log dir not found: {log_dir}")

    runs = find_runs(log_dir)
    if not runs:
        raise SystemExit(
            f"No metrics.csv under {log_dir}. "
            "Run a training script first, or pass the run directory directly.")

    out_path = Path(args.out).expanduser().resolve() if args.out \
        else log_dir / "curves.png"
    plot(runs, out_path, metrics=args.metrics, alpha=args.smooth,
         log_y=not args.linear_y, title=args.title)


if __name__ == "__main__":
    sys.exit(main())
