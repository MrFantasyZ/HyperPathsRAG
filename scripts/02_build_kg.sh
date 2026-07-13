#!/bin/bash
# Stage 2 — Build hypergraph KG: dual entity embeddings, title-prefixed
# relation embeddings, synonym edges, BM25 indices.
# Cost: ~1-2 min on A100 80GB for ~800 events.
# Usage: bash scripts/02_build_kg.sh <inspection_json> <out_dir>
set -euo pipefail

INSP="${1:-llm_inspection.json}"
OUT="${2:-kg_output}"

python src/build_kg.py \
    --inspection "$INSP" \
    --out-dir    "$OUT"

echo "KG saved → $OUT/kg.pkl"
echo "Stats   → $OUT/stats.json"
cat "$OUT/stats.json"
