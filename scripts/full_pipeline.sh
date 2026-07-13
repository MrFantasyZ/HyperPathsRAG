#!/bin/bash
# Usage: bash scripts/full_pipeline.sh <data_file> <run_name>
# Example: bash scripts/full_pipeline.sh data/data_10q.json my_run
set -euo pipefail

DATA="${1:-data/data_10q.json}"
RUN="${2:-run}"

mkdir -p "$RUN"

echo "[1/4] KG event extraction → $RUN/llm_inspection.json"
python src/inspect_llm.py \
    --data "$DATA" \
    --out  "$RUN/llm_inspection.json"

echo "[2/4] Building hypergraph KG → $RUN/kg_output/"
python src/build_kg.py \
    --inspection "$RUN/llm_inspection.json" \
    --out-dir    "$RUN/kg_output"

echo "[3/4] Retrieval → $RUN/retrieval_results.json"
python src/retrieve.py \
    --data "$DATA" \
    --kg   "$RUN/kg_output/kg.pkl" \
    --out  "$RUN/retrieval_results.json"

echo "[4/4] Answer + EM/F1 → $RUN/evaluation_results.json"
python src/evaluate.py \
    --results "$RUN/retrieval_results.json" \
    --out     "$RUN/evaluation_results.json"

echo
echo "DONE. Summary:"
python -c "
import json
ev = json.loads(open('$RUN/evaluation_results.json', encoding='utf-8').read())
print(f'  EM={ev[\"em\"]}%  F1={ev[\"f1\"]}%  ({ev[\"n\"]} questions)')
"
