#!/usr/bin/env python3
"""Import ch07 hifi_queue.csv into CFD batch manifest.json."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to hifi_queue.csv")
    p.add_argument("--out", default="tools/cfd_batch/manifest.json")
    p.add_argument("--solver", default="openfoam", choices=["openfoam", "fluent", "starccm", "custom"])
    p.add_argument("--runs_root", default="results/cfd_runs")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    cases = []
    for row in rows:
        cid = f"case_{int(row.get('rank', len(cases) + 1)):04d}"
        params = {k: float(v) for k, v in row.items() if k not in ("rank", "predicted_cd")}
        pred = float(row.get("predicted_cd", 0))
        work = str(Path(args.runs_root) / cid)
        cases.append({
            "case_id": cid,
            "params": params,
            "predicted_cd": pred,
            "work_dir": work,
            "solver_cmd": f"# TODO: fill solver command for {cid}",
            "status": "pending",
        })

    manifest = {"version": "1", "solver": args.solver, "cases": cases}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
