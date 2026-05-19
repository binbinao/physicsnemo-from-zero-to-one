#!/usr/bin/env bash
# Creates review-round-2 issues. Run from repo root.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BODY_DIR="$REPO_ROOT/docs/issues/review-round2/bodies"
mkdir -p "$BODY_DIR"

create() {
  local num="$1" priority="$2" title="$3" body_file="$4"
  echo "$body" > "$BODY_DIR/issue-${num}.md"
  gh issue create \
    --title "[Review-R2][${priority}] ${title}" \
    --body-file "$BODY_DIR/issue-${num}.md" \
    --label "beginner,onboarding,enhancement" \
  | tee -a "$REPO_ROOT/docs/issues/review-round2/created_urls.txt"
}

# shellcheck disable=SC2154
body() { cat > "$BODY_DIR/issue-${1}.md"; create "$1" "$2" "$3" ""; }
