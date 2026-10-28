#!/usr/bin/env bash
# Run the workflow that produces everything in results/.
#
#   bash run_all.sh          steps 3-8 from the shipped posterior (about 1 min)
#   bash run_all.sh --full   steps 1-8: refit and recompute the posterior (about 30 min)
#
# Uses the Python of the active environment (override with PYTHON=/path/to/python).
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

case "${1:-}" in
  --full) first=1 ;;
  "") first=3 ;;
  *) echo "usage: bash run_all.sh [--full]" >&2; exit 2 ;;
esac

for script in workflow/0*.py; do
  step=$(basename "$script" | cut -c1-2)
  if (( 10#$step >= first )); then
    echo "=== $script"
    "$PYTHON" "$script"
  fi
done
echo "=== done: results/ is up to date"
