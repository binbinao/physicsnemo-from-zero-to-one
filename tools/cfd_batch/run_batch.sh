#!/usr/bin/env bash
# CFD batch driver (dry-run by default). C36
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MANIFEST="${1:-$ROOT/tools/cfd_batch/manifest.json}"
EXECUTE="${EXECUTE:-0}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST"
  echo "Run: python tools/cfd_batch/import_hifi_queue.py --csv ch07_drivaernet_optim/outputs/hifi_queue.csv"
  exit 1
fi

python3 - <<'PY' "$MANIFEST" "$ROOT" "$EXECUTE"
import json, os, subprocess, sys
manifest_path, root, execute = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
data = json.load(open(manifest_path))
for case in data["cases"]:
    cid = case["case_id"]
    work = os.path.join(root, case["work_dir"])
    os.makedirs(work, exist_ok=True)
    cmd = case.get("solver_cmd", "echo noop")
    print(f"\n=== {cid} (predicted_cd={case.get('predicted_cd')}) ===")
    print(f"work_dir: {work}")
  # Write params for traceability
    with open(os.path.join(work, "design_params.json"), "w") as f:
        json.dump(case["params"], f, indent=2)
    if execute and not cmd.strip().startswith("#"):
        subprocess.run(cmd, shell=True, cwd=work, check=False)
    else:
        print(f"[dry-run] {cmd}")
PY

echo ""
echo "Set EXECUTE=1 and fill solver_cmd in manifest to run for real."
