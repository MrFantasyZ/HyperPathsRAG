#!/bin/bash
# Stage 3 — Retrieve contexts per question using the variant pipeline.
# Cost: ~1-3 sec per question on A100 (BGE embedder is the bottleneck).
# Usage: bash scripts/03_retrieve.sh <data_file> <kg_pkl> <out_file>
set -euo pipefail

DATA="${1:-data/data_10q.json}"
KG="${2:-kg_output/kg.pkl}"
OUT="${3:-retrieval_results.json}"

python src/retrieve.py \
    --data "$DATA" \
    --kg   "$KG" \
    --out  "$OUT"

echo "Retrieval saved → $OUT"
