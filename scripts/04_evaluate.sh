#!/bin/bash
# Stage 4 — CoT answering + EM/F1.
# Cost: ~10-30 sec per question on Qwen3-14B (A100). For Llama-3.3-70B
# (Q4_K_M) expect ~30-60 sec per question.
# Usage: bash scripts/04_evaluate.sh <retrieval_json> <out_file>
set -euo pipefail

RETR="${1:-retrieval_results.json}"
OUT="${2:-evaluation_results.json}"

python src/evaluate.py --results "$RETR" --out "$OUT"

python -c "
import json
ev = json.loads(open('$OUT', encoding='utf-8').read())
print()
print(f'EM={ev[\"em\"]}%  F1={ev[\"f1\"]}%  (n={ev[\"n\"]})')
print()
print('Per-question results:')
for i, r in enumerate(ev['results']):
    m = 'OK' if r['em'] else ('~~' if r['f1']>0.3 else '!!')
    print(f'  [{m}] Q{i+1}  gold=[{r[\"gold\"][:30]}]  pred=[{r[\"pred\"][:30]}]')
"
