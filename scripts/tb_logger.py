#!/usr/bin/env python3
"""Unified training-curve logger: TensorBoard event files + CSV (always on).

All training scripts in this repo use this module so that curves are recorded
by default and can be inspected during or after training.

Usage in a training script
--------------------------
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from tb_logger import TBLogger

    TBLogger.add_args(parser)              # adds --tb-dir / --no-tb / --tb-tag
    args = parser.parse_args()
    tb = TBLogger.from_args(args, Path(__file__).stem)
    for step in range(steps):
        ...
        tb.log(step, {"loss/total": loss.item(), "lr": lr})
    tb.close()                             # prints the actual log dir

Behaviour
---------
* Writes ``events.out.tfevents.*`` (tensorboardX) **and** ``metrics.csv``.
* If tensorboardX is missing, it warns once and falls back to CSV only, so
  training never breaks because of logging.
* Disabled via ``--no-tb`` (argparse scripts) or env ``PN_TB=0`` (works
  everywhere, including Hydra entry points); root dir via env ``PN_TB_DIR``.

See docs/ENVIRONMENT.md ("Training curves") and
docs/TENSORBOARD_INTEGRATION_PLAN.md.
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TB_ROOT = REPO_ROOT / "outputs" / "tb"

# tensorboardX import is lazy so that the CSV fallback works without it.
_WARNED_NO_TB = False


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _env_disabled() -> bool:
    return os.environ.get("PN_TB", "1").strip().lower() in ("0", "false", "no", "off")


def _truthy(value: str) -> bool:
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def _cfg_get(obj, key, default=None):
    """Read ``key`` from an argparse Namespace or a DictConfig/dict."""
    if obj is None:
        return default
    try:
        return obj[key]
    except (TypeError, KeyError, AttributeError):
        return getattr(obj, key.replace("-", "_"), default)


class TBLogger:
    """Record scalars to TensorBoard and CSV at the same time."""

    def __init__(self, log_dir, enabled: bool = True, run_name: str | None = None):
        # PN_TB=0 turns logging off everywhere, including explicit log dirs
        self.enabled = bool(enabled) and log_dir is not None and not _env_disabled()
        self.log_dir = Path(log_dir) if self.enabled else None
        self.run_name = run_name or (self.log_dir.name if self.log_dir else "")
        self._writer = None
        self._rows: list[dict] = []
        self._fields: list[str] = ["step"]
        self._csv_path = None

        if not self.enabled:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.log_dir / "metrics.csv"
        self._writer = self._make_writer()

    # ------------------------------------------------------------------ setup
    def _make_writer(self):
        global _WARNED_NO_TB
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            if not _WARNED_NO_TB:
                print("[tb] tensorboardX not installed - logging to CSV only "
                      "(pip install tensorboardX)")
                _WARNED_NO_TB = True
            return None
        return SummaryWriter(str(self.log_dir))

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add the standard TensorBoard flags to an argparse parser."""
        parser.add_argument("--tb-dir", default=None,
                            help="TensorBoard log dir (default: <out_dir>/tb "
                                 "or outputs/tb/<script>/<timestamp>)")
        parser.add_argument("--tb-tag", default=None,
                            help="Custom run name under the log root")
        parser.add_argument("--tb-no-timestamp", action="store_true",
                            help="Do not create a timestamped sub-directory")
        parser.add_argument("--no-tb", action="store_true",
                            help="Disable TensorBoard/CSV curve logging")
        return parser

    @classmethod
    def from_args(cls, args, script_stem: str, out_dir=None) -> "TBLogger":
        """Build a logger from parsed args.

        Directory resolution order:
          1. ``--tb-dir`` (explicit)
          2. ``out_dir`` argument (or args.output_dir / args.out_dir) -> ``<dir>/tb``
          3. ``<repo>/outputs/tb/<script_stem>[/<tag>]/<timestamp>``
        """
        if getattr(args, "no_tb", False) or _env_disabled():
            return cls(None, enabled=False)

        explicit = _cfg_get(args, "tb_dir", None)
        if explicit:
            return cls(Path(explicit).expanduser().resolve(),
                       run_name=_cfg_get(args, "tb_tag", None))

        if out_dir is None:
            out_dir = _cfg_get(args, "output_dir", None) or _cfg_get(args, "out_dir", None)
        root = os.environ.get("PN_TB_DIR")

        if out_dir and not root:
            # 时间戳子目录让每次运行成为独立 run，避免曲线在 TensorBoard 中叠加
            base = Path(out_dir).expanduser().resolve() / "tb"
            if _cfg_get(args, "tb_no_timestamp", False):
                return cls(base)
            return cls(base / _stamp())

        base = Path(root).expanduser().resolve() if root else DEFAULT_TB_ROOT
        parts = [base, script_stem]
        tag = _cfg_get(args, "tb_tag", None)
        if tag:
            parts.append(tag)
        if not _cfg_get(args, "tb_no_timestamp", False):
            parts.append(_stamp())
        return cls(Path(*parts))

    @classmethod
    def under(cls, out_dir, timestamp: bool = True, tag: str | None = None) -> "TBLogger":
        """Create a logger in ``<out_dir>/tb[/<tag>][/<timestamp>]``.

        For entry points that already know their output directory (Hydra,
        config-driven scripts) and cannot rely on argparse flags. Env
        ``PN_TB=0`` disables logging here too (Hydra 档唯一的关闭方式).
        """
        if _env_disabled():
            return cls(None, enabled=False)
        base = Path(out_dir).expanduser().resolve() / "tb"
        if tag:
            base = base / tag
        if timestamp:
            base = base / _stamp()
        return cls(base)

    # ------------------------------------------------------------------ write
    def log(self, step, metrics: dict) -> None:
        """Log a dict of scalars at ``step``."""
        if not self.enabled:
            return
        row = {"step": int(step)}
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                continue
            if self._writer is not None:
                self._writer.add_scalar(str(key), row[key], int(step))

        added = False
        for key in row:
            if key not in self._fields:
                self._fields.append(key)
                added = True
        self._rows.append(row)
        # Rewriting only happens when a new metric appears (rare).
        if added and len(self._rows) > 1:
            self._flush()
        else:
            self._append(row)

    def log_figure(self, tag: str, fig, step) -> None:
        """Attach a matplotlib figure (ignored when tensorboardX is absent)."""
        if not self.enabled or self._writer is None:
            return
        try:
            self._writer.add_figure(str(tag), fig, int(step))
        except Exception as exc:  # never break training for logging
            print(f"[tb] add_figure failed: {exc}")

    def _append(self, row: dict) -> None:
        if self._csv_path is None:
            return
        new_file = not self._csv_path.exists()
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fields,
                                    extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def _flush(self) -> None:
        if self._csv_path is None:
            return
        with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fields,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    # ------------------------------------------------------------------ close
    def close(self) -> None:
        if not self.enabled:
            return
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._flush()
        print(f"TensorBoard log dir: {self.log_dir}")
        print(f"  view:  tensorboard --logdir {self.log_dir}")
        print(f"  plot:  python scripts/plot_tb_curves.py --logdir {self.log_dir}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def current_lr(optimizer) -> float:
    """Helper: read the current LR (no script records it today)."""
    try:
        return float(optimizer.param_groups[0]["lr"])
    except (AttributeError, IndexError, KeyError):
        return float("nan")
