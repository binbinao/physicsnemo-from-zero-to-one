#!/usr/bin/env python3
"""Ingest per-case CFD results into summary CSV for V&V (C39)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_cd(case_dir: Path) -> float | None:
    f = case_dir / "result_cd.txt"
    if not f.exists():
        return None
    line = f.read_text(encoding="utf-8").strip().splitlines()[0]
    for token in line.replace(",", " ").split():
        try:
            return float(token.split("=")[-1] if "=" in token else token)
        except ValueError:
            continue
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="tools/cfd_batch/manifest.json")
    p.add_argument("--out", default="results/cfd_runs/validation_summary.csv")
    args = p.parse_args()

    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = []
    for case in data["cases"]:
        work = Path(case["work_dir"])
        cfd_cd = read_cd(work)
        rows.append({
            "case_id": case["case_id"],
            "predicted_cd": case.get("predicted_cd"),
            "cfd_cd": cfd_cd,
            "abs_err": abs(cfd_cd - case["predicted_cd"]) if cfd_cd is not None else "",
            **{f"p_{k}": v for k, v in case["params"].items()},
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {out} ({sum(1 for r in rows if r['cfd_cd']!='')} with cfd_cd)")


if __name__ == "__main__":
    main()
