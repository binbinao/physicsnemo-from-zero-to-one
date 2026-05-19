#!/usr/bin/env python3
"""Import ch07 hifi_queue.csv into CFD batch manifest.json."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOCK_CMD = f'python "{ROOT / "tools/cfd_batch/mock_cfd_solver.py"}"'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to hifi_queue.csv")
    p.add_argument("--out", default="tools/cfd_batch/manifest.json")
    p.add_argument(
        "--mode",
        default="mock",
        choices=["mock", "openfoam", "fluent", "template"],
        help="mock: in-repo synthetic CFD; others: placeholder cmd",
    )
    p.add_argument("--runs_root", default="results/cfd_runs")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    cases = []
    for row in rows:
        cid = f"case_{int(row.get('rank', len(cases) + 1)):04d}"
        params = {k: float(v) for k, v in row.items() if k not in ("rank", "predicted_cd")}
        pred = float(row.get("predicted_cd", 0))
        work = str(Path(args.runs_root) / cid)
        if args.mode == "mock":
            cmd = MOCK_CMD
            solver = "mock_cfd"
        elif args.mode == "openfoam":
            cmd = "# bash Allrun  # see templates/openfoam/"
            solver = "openfoam"
        elif args.mode == "fluent":
            cmd = "# fluent 3ddp -g < run_case.jou"
            solver = "fluent"
        else:
            cmd = f"# TODO: fill solver command for {cid}"
            solver = "custom"
        cases.append({
            "case_id": cid,
            "params": params,
            "predicted_cd": pred,
            "work_dir": work,
            "solver_cmd": cmd,
            "status": "pending",
        })

    manifest = {"version": "1", "solver": solver, "mode": args.mode, "cases": cases}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(cases)} cases, mode={args.mode})")
    if args.mode == "mock":
        print(f"Run: python tools/cfd_batch/run_batch.py --mode mock")


if __name__ == "__main__":
    main()
