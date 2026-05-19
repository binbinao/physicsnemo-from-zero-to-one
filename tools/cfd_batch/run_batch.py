#!/usr/bin/env python3
"""Run CFD batch from manifest (dry-run, mock, or custom execute)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOCK_SOLVER = Path(__file__).resolve().parent / "mock_cfd_solver.py"


def run_case(case: dict, mode: str, root: Path) -> None:
    work = root / case["work_dir"]
    work.mkdir(parents=True, exist_ok=True)
    (work / "design_params.json").write_text(
        json.dumps(case["params"], indent=2), encoding="utf-8"
    )
    cid = case["case_id"]
    print(f"\n=== {cid} predicted_cd={case.get('predicted_cd')} ===")

    if mode == "dry-run":
        print(f"  [dry-run] {case.get('solver_cmd', '# no cmd')}")
        return

    if mode == "mock":
        subprocess.run([sys.executable, str(MOCK_SOLVER)], cwd=work, check=True)
        return

    cmd = case.get("solver_cmd", "")
    if not cmd or cmd.strip().startswith("#"):
        print(f"  [skip] empty solver_cmd for {cid}")
        return
    subprocess.run(cmd, shell=True, cwd=work, check=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="tools/cfd_batch/manifest.json")
    p.add_argument(
        "--mode",
        choices=["dry-run", "mock", "execute"],
        default="dry-run",
        help="mock = in-repo synthetic CFD; execute = run solver_cmd from manifest",
    )
    args = p.parse_args()
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    for case in data["cases"]:
        run_case(case, args.mode, ROOT)
    print(f"\nDone ({len(data['cases'])} cases, mode={args.mode})")
    if args.mode == "mock":
        print("Next: python tools/cfd_batch/ingest_results.py")


if __name__ == "__main__":
    main()
