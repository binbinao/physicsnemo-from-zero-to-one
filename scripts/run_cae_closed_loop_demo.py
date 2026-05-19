#!/usr/bin/env python3
"""
End-to-end CAE closed-loop demo WITHOUT commercial CFD licenses.

  python scripts/run_cae_closed_loop_demo.py
  python scripts/run_cae_closed_loop_demo.py --skip-train --skip-ch03

Steps: ch07 train → optimize → hifi queue → mock CFD batch → ingest → V&V summary
Optional: ch03 joint inversion smoke (--fast; see CAE_CLOSED_LOOP_DEMO.md)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-ch03", action="store_true")
    p.add_argument("--train-epochs", type=int, default=80)
    p.add_argument("--opt-trials", type=int, default=80)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    ch07 = ROOT / "ch07_drivaernet_optim"
    ckpt = ch07 / "outputs" / "best.pt"

    if not args.skip_train:
        run([PY, "data/generate_toy_cars.py", "--n_samples", "800"], cwd=ch07)
        run([PY, "train.py", "--epochs", str(args.train_epochs)], cwd=ch07)

    if not ckpt.exists():
        print(f"Missing {ckpt}; run without --skip-train first")
        sys.exit(1)

    run(
        [PY, "optimize.py", "--checkpoint", "outputs/best.pt", "--n_trials", str(args.opt_trials)],
        cwd=ch07,
    )
    run(
        [
            PY,
            "optimize_multi.py",
            "--checkpoint",
            "outputs/best.pt",
            "--n_trials",
            str(args.opt_trials),
            "--export-hifi",
            "--top-k",
            str(args.top_k),
        ],
        cwd=ch07,
    )

    csv_path = ch07 / "outputs" / "hifi_queue.csv"
    run(
        [
            PY,
            str(ROOT / "tools/cfd_batch/import_hifi_queue.py"),
            "--csv",
            str(csv_path),
            "--mode",
            "mock",
        ],
    )
    run([PY, str(ROOT / "tools/cfd_batch/run_batch.py"), "--mode", "mock"])
    run([PY, str(ROOT / "tools/cfd_batch/ingest_results.py")])

    if not args.skip_ch03:
        run(
            [PY, "heat_sink_inverse_joint.py", "--target_temp", "40", "--fast"],
            cwd=ROOT / "ch03_heatsink",
        )

    summary = ROOT / "results/cfd_runs/validation_summary.csv"
    print(f"\n=== Closed-loop demo complete ===\nSee {summary} for surrogate vs mock-CFD Cd.")
    print("Fill docs/VV_REPORT_TEMPLATE.md with the summary table.")


if __name__ == "__main__":
    main()
