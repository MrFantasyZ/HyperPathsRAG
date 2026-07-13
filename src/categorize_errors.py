"""
Failure-mode Categorisation — Appendix E.

Reads an evaluation_results.json produced by evaluate.py, samples 50
incorrect predictions (EM == 0), and asks an LLM to classify each into
one of six failure categories used in the paper:

  1. KG-missing-fact          : the chunk-to-fact extraction silently
                                dropped the bridging fact needed for
                                the answer.
  2. Decomposition under-coverage:
                                the decomposer omitted a sub-question
                                or merged two hops into one.
  3. Entity disambiguation    : a wrong synonym merge or a missed
                                synonym at Step 4.
  4. Long-tail relation       : the correct relation was retrieved but
                                ranked too low to enter the top-K.
  5. LLM answering error      : correct context, but the answering LLM
                                mis-reasoned.
  6. Annotation noise         : the gold answer is itself ambiguous or
                                wrong.

The classifier is the same Ollama LLM used elsewhere (Qwen3-14B by
default); the classification prompt instructs it to emit one of the
six labels plus a one-line justification.

Usage:
  python src/categorize_errors.py \
      --eval results/main_table/hotpot/HyperPathsRAG/qwen3-14b/answers.json \
      --retrieval results/main_table/hotpot/HyperPathsRAG/qwen3-14b/retrieval.json \
      --out results/appendix_E_failures/hotpot/failures.json \
      --n 50
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import requests

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"

CATEGORIES = [
    "kg_missing_fact",
    "decomposition_under_coverage",
    "entity_disambiguation",
    "long_tail_relation",
    "llm_answering_error",
    "annotation_noise",
]

CLASSIFIER_SYSTEM = """You are an experienced multi-hop QA reviewer. You will be given an error case from a RAG system together with its retrieved context, decomposition, and predicted answer. Classify the failure into EXACTLY ONE of these six categories:

1. kg_missing_fact          - the bridging fact that the answer depends on is NOT present anywhere in the retrieved context; the knowledge graph never extracted that fact from the corpus.
2. decomposition_under_coverage - the question was decomposed into sub-questions that miss a hop; one of the named entities in the question never appears with virtual=false in any sub-question, OR two distinct reasoning hops were merged into a single sub-question.
3. entity_disambiguation    - the retrieved context shows two distinct entities that should have been distinguished but were merged, or two surface forms of the same entity that should have been linked but weren't.
4. long_tail_relation       - the correct bridging relation IS present in the retrieved context but does not lead the LLM to the correct answer because adjacent noisier relations dominate the context.
5. llm_answering_error      - the retrieved context is correct and complete, but the answering LLM reasons wrongly (e.g. picks an intermediate hop entity, fails a numeric comparison, or ignores a constraint).
6. annotation_noise         - the gold answer itself is ambiguous, plural, or clearly wrong given the question.

Output STRICTLY in this format on two lines:
Category: <one_of_the_six_labels>
Reason: <one-line justification, ≤30 words>"""


def classify_one(question: str, gold: str, pred: str, context: str,
                 subqs: list, model: str = OLLAMA_MODEL) -> dict:
    subq_str = "\n".join(
        f"  q{i+1} ({'/'.join(e.get('attribute','?') for e in sq.get('entities',[]))})"
        f": {sq.get('relation','?')}"
        for i, sq in enumerate(subqs or [])
    ) or "  (decomposition unavailable)"
    user_msg = (
        f"Question: {question}\n"
        f"Gold answer:      {gold}\n"
        f"Predicted answer: {pred}\n\n"
        f"Sub-questions emitted by the decomposer:\n{subq_str}\n\n"
        f"Retrieved context (verbatim):\n{context}\n"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False, "think": True, "keep_alive": "1h",
        "options": {"temperature": 0, "num_predict": 256},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    raw = r.json()["message"].get("content", "")
    cat = None
    reason = ""
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("category:"):
            cat = line.split(":", 1)[1].strip().lower().replace(" ", "_")
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    if cat not in CATEGORIES:
        # Fall back: search for any of the canonical labels in the raw text
        for c in CATEGORIES:
            if c in raw.lower():
                cat = c
                break
    return {"category": cat or "unparseable", "reason": reason, "raw": raw[:400]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval",       type=str, required=True,
                    help="Evaluation JSON (output of evaluate.py)")
    ap.add_argument("--retrieval",  type=str, required=True,
                    help="Retrieval JSON (output of retrieve.py)")
    ap.add_argument("--out",        type=str, required=True,
                    help="Output JSON path")
    ap.add_argument("--n",          type=int, default=50,
                    help="Number of errors to sample (default: 50)")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--model",      type=str, default=OLLAMA_MODEL)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)],
                        force=True)

    eval_data = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    retr_data = json.loads(Path(args.retrieval).read_text(encoding="utf-8"))
    retr_by_q = {r.get("question"): r for r in retr_data}

    errors = [r for r in eval_data["results"] if r["em"] == 0]
    logging.info("Loaded %d total, %d errors", eval_data["n"], len(errors))

    random.seed(args.seed)
    sample = random.sample(errors, min(args.n, len(errors)))

    classifications = []
    for i, e in enumerate(sample, 1):
        retr = retr_by_q.get(e["question"], {})
        ctx = retr.get("context") or ""
        subqs = retr.get("subqs") or []
        cls = classify_one(e["question"], e["gold"], e["pred"], ctx, subqs,
                           model=args.model)
        cls.update({"qid": e["qid"], "question": e["question"],
                    "gold": e["gold"], "pred": e["pred"]})
        classifications.append(cls)
        logging.info("[%d/%d] %s — %s", i, len(sample), cls["category"], cls["reason"][:120])

    distribution = Counter(c["category"] for c in classifications)
    total = sum(distribution.values())
    summary = {
        "n_classified": total,
        "distribution_fraction": {k: round(v / max(total, 1), 4)
                                  for k, v in distribution.items()},
        "distribution_count":    dict(distribution),
        "classifier_model":      args.model,
        "categories":            CATEGORIES,
        "cases":                 classifications,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    logging.info("Saved failure categorisation → %s", args.out)
    logging.info("Distribution: %s", summary["distribution_fraction"])


if __name__ == "__main__":
    main()
