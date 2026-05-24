#!/bin/bash
# Top-level orchestrator: run the full experiment workflow for every
# dataset, then aggregate. Safe to re-run; per-config caching skips
# completed outputs.
#
# Recommended environment:
#   export LLMS="qwen3:14b,llama3.3:70b-instruct-q4_K_M"
#   export PRIMING="reasoning-first"
#
# Usage:  bash scripts/run_all.sh [dataset1 dataset2 ...]
#         defaults to: hotpot 2wiki musique
set -euo pipefail

DATASETS=("${@:-hotpot 2wiki musique}")

mkdir -p results
LOGDIR=results/logs
mkdir -p "$LOGDIR"

for DS in "${DATASETS[@]}"; do
  echo "======================================================================"
  echo " Dataset: $DS"
  echo "======================================================================"
  LOG="$LOGDIR/$DS.log"
  bash scripts/run_dataset.sh "$DS" 2>&1 | tee "$LOG"
done

echo
echo "Aggregating all results → results/final_tables.md"
python src/aggregate_results.py --results results --out results/final_tables.md
echo "DONE.  Open results/final_tables.md."
