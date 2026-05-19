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

if [[ "${1:-}" == "--all-p2" ]]; then
  set -- f1_3_spring_diagram.png f1_1_roadmap.png f2_1_roadmap.png f3_3_geometry.png f3_4_csg.png f0_1_ai4s_timeline.png
fi

if [[ "${1:-}" == "--all-banners" ]]; then
  set -- f1_0_banner.png f3_0_banner.png f4_0_banner.png f5_0_banner.png f6_0_banner.png f7_0_banner.png
fi

if [[ "${1:-}" == "--all-p3" ]]; then
  set -- \
    f2_4_physics_intuition.png \
    f3_1_progression.png \
    f3_5_boundary_conditions.png \
    f3_8_inverse_flow.png \
    f4_4_neural_operator.png \
    f4_6_data_pipeline.png \
    f5_1_triangle.png \
    f5_3_darcy_physics.png \
    f6_1_afno_block.png \
    f6_3_autoregressive.png \
    f7_1_pipeline.png \
    f1_7_physicsnemo_arch.png
fi

for f in "$@"; do
  src="$GEMINI/$f"
  [[ -f "$src" ]] || { echo "missing $src"; exit 1; }
  cp -f "$src" "$ASSETS/$f"
  echo "promoted $f"
done
