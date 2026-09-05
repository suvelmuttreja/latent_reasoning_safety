#!/usr/bin/env bash
# Copy a finished stage's compact JSON/metrics records from scratch1 to home1 (usage: BRANCH_ROOT STAGE).
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BRANCH_ROOT STAGE" >&2
  exit 2
fi
case "$1" in
  fallback_4b_skip0|matched_4b_cot) branch_root="$1" ;;
  *) echo "unsupported branch root: $1" >&2; exit 2 ;;
esac
case "$2" in
  1|2|3) stage="$2" ;;
  *) echo "unsupported stage: $2" >&2; exit 2 ;;
esac
: "${WORK_ROOT:?WORK_ROOT is required}"
: "${CODE_ROOT:?CODE_ROOT is required}"

source_dir="$WORK_ROOT/results/$branch_root/stage$stage"
home_dir="$CODE_ROOT/artifacts/discovery/results/$branch_root/stage$stage"
files=(metadata.json update_metrics.jsonl durability_receipt.json)
mkdir -p "$home_dir"
for name in "${files[@]}"; do
  test -s "$source_dir/$name"
  rsync -a "$source_dir/$name" "$home_dir/$name"
  cmp -s "$source_dir/$name" "$home_dir/$name"
  sha256sum "$source_dir/$name" "$home_dir/$name"
done
