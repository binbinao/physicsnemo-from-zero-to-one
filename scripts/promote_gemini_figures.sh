#!/usr/bin/env bash
# Copy approved gemini trial PNGs into book/assets/ for publication.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEMINI="$ROOT/book/assets/gemini"
ASSETS="$ROOT/book/assets"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 f1_0_banner.png [f3_0_banner.png ...]"
  echo "   or: $0 --all-p0"
  exit 1
fi

if [[ "${1:-}" == "--all-p0" ]]; then
  set -- f1_0_banner.png f3_0_banner.png f4_0_banner.png f5_0_banner.png f6_0_banner.png f7_0_banner.png f0_4_book_roadmap.png
fi

for f in "$@"; do
  src="$GEMINI/$f"
  [[ -f "$src" ]] || { echo "missing $src"; exit 1; }
  cp -f "$src" "$ASSETS/$f"
  echo "promoted $f"
done
