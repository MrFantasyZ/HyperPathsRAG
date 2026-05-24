#!/bin/bash
# Hyperparameter sensitivity sweep (Appendix C).
#
# For one dataset, sweep each of the four hyperparameters across five
# values, retrieve + answer, and store one score.json per cell.
#
# Usage:  bash scripts/run_sensitivity.sh <dataset>
set -euo pipefail

DS="${1:?usage: $0 <hotpot|2wiki|musique>}"
case "$DS" in
  hotpot)  DATA="data/hotpot_hard_100.json" ;;
  2wiki)   DATA="data/2wiki_hard_100.json"  ;;
  musique) DATA="data/musique_hard_100.json";;
  *)       echo "Unknown dataset: $DS" >&2 ; exit 1 ;;
esac

RESULTS=results
KG_PKL="$RESULTS/kg_stats/$DS/kg.pkl"
LLM="${LLM:-qwen3:14b}"

if [[ ! -f "$KG_PKL" ]]; then
  echo "KG not built yet for $DS; run scripts/run_dataset.sh $DS first." >&2
  exit 1
fi

run_sweep() {
  local PARAM=$1; local FLAG=$2; shift 2
  for V in "$@"; do
    local DIR="$RESULTS/appendix_C_sensitivity/$DS/$PARAM/$V"
    mkdir -p "$DIR"
    local RET="$DIR/retrieval.json"; local ANS="$DIR/answers.json"
    if [[ -f "$DIR/score.json" ]]; then echo "  ↳ $PARAM=$V skip"; continue; fi
    echo "  $PARAM=$V"
    python src/retrieve.py --data "$DATA" --kg "$KG_PKL" \
        "$FLAG" "$V" --ablation "sens-$PARAM-$V" \
        --out "$RET"
    python src/evaluate.py --results "$RET" --out "$ANS" --model "$LLM" \
        --priming reasoning-first
    python -c "
import json
d = json.load(open(r'''$ANS''', encoding='utf-8'))
out = {k: d[k] for k in ('n','em','f1','model','priming','tokens_per_q')}
open(r'''$DIR/score.json''','w',encoding='utf-8').write(json.dumps(out, indent=2))
"
  done
}

echo "λ (BM25 weight)"
run_sweep lambda --bm25-lambda 0.0 0.1 0.3 0.5 1.0

echo "τ (synonym thr.)"
run_sweep tau --virtual-match-thr 0.70 0.80 0.85 0.90 0.95

echo "δ (top-K gap)"
run_sweep delta --rel-gap-thr 0.01 0.03 0.05 0.08 0.12

echo "K_min"
run_sweep k-min --rel-k-min 1 2 2 3 3

echo "K_max"
run_sweep k-max --rel-k-max 3 4 5 6 8

echo "DONE — sensitivity sweep for $DS"
