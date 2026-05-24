#!/bin/bash
# Per-dataset full workflow.  Skips any (config, output) pair whose JSON
# already exists so the script is safely re-runnable after partial failure.
#
# Usage:  bash scripts/run_dataset.sh <dataset>
#         dataset ∈ {hotpot, 2wiki, musique}
#
# Outputs go under results/ (see RESULTS_LAYOUT.md).
set -euo pipefail

DS="${1:?usage: $0 <hotpot|2wiki|musique>}"
case "$DS" in
  hotpot)  DATA="data/hotpot_hard_100.json" ;;
  2wiki)   DATA="data/2wiki_hard_100.json"  ;;
  musique) DATA="data/musique_hard_100.json";;
  *)       echo "Unknown dataset: $DS" >&2 ; exit 1 ;;
esac

# Optional LLM list — comma-separated. Defaults to the three answer LLMs
# from the paper (Table 2).
LLMS="${LLMS:-qwen3:14b}"          # default: single LLM; override via env
PRIMING="${PRIMING:-reasoning-first}"

RESULTS=results
LLM_INSPECTION="$RESULTS/llm_inspection/$DS.json"
KG_DIR="$RESULTS/kg_stats/$DS"
KG_PKL="$KG_DIR/kg.pkl"

mkdir -p "$RESULTS/kg_stats/$DS" \
         "$RESULTS/main_table/$DS/HyperPathsRAG" \
         "$RESULTS/ablation/$DS" \
         "$RESULTS/efficiency/$DS/HyperPathsRAG" \
         "$RESULTS/appendix_B_embedding/$DS" \
         "$RESULTS/appendix_E_failures/$DS" \
         "$RESULTS/appendix_G_cot/$DS" \
         "$RESULTS/appendix_I_ragas/$DS"

# ─── Phase 1: KG extraction + build (once per dataset) ───────────────────────
if [[ ! -f "$LLM_INSPECTION" ]]; then
  echo "[1/6] Extracting events with LLM → $LLM_INSPECTION"
  python src/inspect_llm.py --data "$DATA" --out "$LLM_INSPECTION"
else
  echo "[1/6] Skipping LLM extraction — $LLM_INSPECTION exists"
fi

if [[ ! -f "$KG_PKL" ]]; then
  echo "[2/6] Building hypergraph KG → $KG_DIR"
  python src/build_kg.py --inspection "$LLM_INSPECTION" --out-dir "$KG_DIR"
else
  echo "[2/6] Skipping KG build — $KG_PKL exists"
fi

# ─── Phase 2: Main retrieval (HyperPathsRAG full) ────────────────────────────
RET_MAIN="$RESULTS/main_table/$DS/HyperPathsRAG/retrieval.json"
TIM_MAIN="$RESULTS/efficiency/$DS/HyperPathsRAG/timing.json"
if [[ ! -f "$RET_MAIN" ]]; then
  echo "[3/6] HyperPathsRAG retrieval → $RET_MAIN"
  python src/retrieve.py \
      --data "$DATA" --kg "$KG_PKL" \
      --ablation full \
      --out "$RET_MAIN" \
      --save-timing "$TIM_MAIN"
else
  echo "[3/6] Skipping main retrieval — $RET_MAIN exists"
fi

# ─── Phase 3: Multi-LLM × multi-priming answering (Table 2 + Appendix G) ─────
echo "[4/6] Answering with each LLM in $LLMS"
IFS=',' read -ra LLM_ARR <<< "$LLMS"
for LLM in "${LLM_ARR[@]}"; do
  # Sanitise LLM name for filesystem (Ollama uses ':')
  LLM_TAG=$(echo "$LLM" | tr ':/' '__')
  ANS_DIR="$RESULTS/main_table/$DS/HyperPathsRAG/$LLM_TAG"
  mkdir -p "$ANS_DIR"
  ANS="$ANS_DIR/answers.json"
  if [[ -f "$ANS" ]]; then
    echo "  ↳ $LLM: skip (exists)"
    continue
  fi
  python src/evaluate.py \
      --results "$RET_MAIN" \
      --out "$ANS" \
      --model "$LLM" \
      --priming "$PRIMING" \
      --save-tokens "$ANS_DIR/tokens.json"
  # Tee score-only summary into score.json for the aggregator
  python -c "
import json
d = json.load(open(r'''$ANS''', encoding='utf-8'))
out = {k: d[k] for k in ('n','em','f1','model','priming','tokens_per_q')}
open(r'''$ANS_DIR/score.json''','w',encoding='utf-8').write(json.dumps(out, indent=2))
"
done

# ─── Phase 4: Ablation suite (Table 3) ───────────────────────────────────────
# We sweep the algorithmic ablations exposed via CLI flags. The actual
# code-level ablations (no-multitarget, no-chaincomp, no-order,
# no-hypergraph) live in separate branches of retrieve.py / build_kg.py
# that the user implements; this script invokes them via the --ablation
# tag and the corresponding flag where possible.
echo "[5/6] Ablation sweep (HyperPathsRAG variants)"
PRIMARY_LLM="${LLM_ARR[0]}"
PRIMARY_TAG=$(echo "$PRIMARY_LLM" | tr ':/' '__')
declare -A ABL_FLAGS=(
  [no-bm25]="--bm25-lambda 0.0"
  [no-ntary]="--ablation no-ntary"             # algorithmic: enable in retrieve.py
  [no-multitarget]="--ablation no-multitarget" # algorithmic: enable in retrieve.py
  [no-chaincomp]="--ablation no-chaincomp"     # algorithmic: enable in retrieve.py
  [no-order]="--ablation no-order"             # algorithmic: enable in retrieve.py
  [no-hypergraph]="--ablation no-hypergraph"   # build_kg.py binary-triple variant
)
for ABL in "${!ABL_FLAGS[@]}"; do
  ABL_DIR="$RESULTS/ablation/$DS/$ABL/$PRIMARY_TAG"
  mkdir -p "$ABL_DIR"
  RET_ABL="$ABL_DIR/retrieval.json"
  ANS_ABL="$ABL_DIR/answers.json"
  if [[ -f "$ANS_ABL" ]]; then
    echo "  ↳ $ABL: skip (exists)"
    continue
  fi
  python src/retrieve.py \
      --data "$DATA" --kg "$KG_PKL" \
      $(echo "${ABL_FLAGS[$ABL]}") \
      --out "$RET_ABL" || { echo "  ! retrieve.py failed for $ABL"; continue; }
  python src/evaluate.py \
      --results "$RET_ABL" \
      --out "$ANS_ABL" \
      --model "$PRIMARY_LLM" \
      --priming "$PRIMING"
  python -c "
import json
d = json.load(open(r'''$ANS_ABL''', encoding='utf-8'))
out = {k: d[k] for k in ('n','em','f1','model','priming','tokens_per_q')}
open(r'''$ABL_DIR/score.json''','w',encoding='utf-8').write(json.dumps(out, indent=2))
"
done

# ─── Phase 5: Appendix experiments ───────────────────────────────────────────
echo "[6/6] Appendix sub-experiments"

# Appendix B — entity embedding probe (uses the built KG embeddings)
APP_B="$RESULTS/appendix_B_embedding/$DS/embedding_probe.json"
if [[ ! -f "$APP_B" ]]; then
  python src/embedding_probe.py --out "$APP_B" || echo "  ! embedding_probe failed"
else
  echo "  ↳ Appendix B: skip"
fi

# Appendix E — failure mode categorisation (samples 50 incorrect answers)
APP_E="$RESULTS/appendix_E_failures/$DS/failures.json"
MAIN_ANS="$RESULTS/main_table/$DS/HyperPathsRAG/$PRIMARY_TAG/answers.json"
if [[ ! -f "$APP_E" && -f "$MAIN_ANS" ]]; then
  python src/categorize_errors.py \
      --eval "$MAIN_ANS" --retrieval "$RET_MAIN" \
      --out "$APP_E" --n 50 \
      || echo "  ! categorize_errors failed"
fi

# Appendix G — CoT priming order (reuses main retrieval; just re-answers)
echo "  Appendix G — priming order"
for PR in answer-first answer-only; do
  PR_DIR="$RESULTS/appendix_G_cot/$DS/$PR"
  mkdir -p "$PR_DIR"
  PR_ANS="$PR_DIR/answers.json"
  if [[ -f "$PR_ANS" ]]; then continue; fi
  python src/evaluate.py \
      --results "$RET_MAIN" \
      --out "$PR_ANS" \
      --model "$PRIMARY_LLM" \
      --priming "$PR" \
      || echo "  ! priming=$PR failed"
  python -c "
import json
d = json.load(open(r'''$PR_ANS''', encoding='utf-8'))
out = {k: d[k] for k in ('n','em','f1','model','priming','tokens_per_q')}
open(r'''$PR_DIR/score.json''','w',encoding='utf-8').write(json.dumps(out, indent=2))
"
done

# Appendix I — RAGAS faithfulness + context relevance
APP_I="$RESULTS/appendix_I_ragas/$DS/HyperPathsRAG/ragas.json"
mkdir -p "$RESULTS/appendix_I_ragas/$DS/HyperPathsRAG"
if [[ ! -f "$APP_I" && -f "$MAIN_ANS" ]]; then
  python src/ragas_eval.py \
      --eval "$MAIN_ANS" --retrieval "$RET_MAIN" \
      --out "$APP_I" --limit 100 \
      || echo "  ! ragas_eval failed"
fi

echo
echo "DONE  ($DS).  Aggregate with:  python src/aggregate_results.py"
