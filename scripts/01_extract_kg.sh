#!/bin/bash
# KG event extraction. Stage 1 — slowest stage (LLM call per chunk).
# Usage: bash scripts/01_extract_kg.sh <data_file> <out_file>
# Example: bash scripts/01_extract_kg.sh data/musique_top20_hardest.json out/llm_inspection.json
set -euo pipefail

DATA="${1:-data/data_10q.json}"
OUT="${2:-llm_inspection.json}"

python src/inspect_llm.py --data "$DATA" --out "$OUT"

# ─── HPC multi-GPU sharding template (uncomment to use) ─────────────────────
# # 1. Launch N ollama servers on different GPUs and ports:
# for i in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$i OLLAMA_HOST=0.0.0.0:$((11434+i)) \
#         OLLAMA_KEEP_ALIVE=2h nohup ollama serve > "ollama_$i.log" 2>&1 &
# done
# sleep 10
#
# # 2. Shard input by question index modulo N:
# N=4
# for i in $(seq 0 $((N-1))); do
#     python src/inspect_llm.py \
#         --data "$DATA" --shard "$i/$N" \
#         --port $((11434+i)) \
#         --out "shard_${i}.json" &
# done
# wait
#
# # 3. Merge shards:
# python src/inspect_llm.py --merge "shard_*.json" --out "$OUT"
# rm shard_*.json
