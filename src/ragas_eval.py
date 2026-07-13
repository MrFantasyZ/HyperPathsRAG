"""
RAGAS-style Faithfulness + Context Relevance — Appendix I.

Scores a (question, retrieved_context, predicted_answer) tuple along two
axes commonly used in RAG evaluation:

  - faithfulness        : does the predicted answer make claims that the
                          retrieved context actually supports?
  - context_relevance   : how much of the retrieved context is relevant
                          to the question?

The reference implementation is the `ragas` Python package (uses an
OpenAI / Anthropic / Ollama LLM as judge). To stay self-contained on an
air-gapped HPC, this script implements both metrics as direct
LLM-as-a-judge prompts hitting the same Ollama daemon used elsewhere.

Output: a per-question array of {faithfulness, context_relevance} plus
the dataset means.

Usage:
  python src/ragas_eval.py \
      --eval results/main_table/hotpot/HyperPathsRAG/qwen3-14b/answers.json \
      --retrieval results/main_table/hotpot/HyperPathsRAG/qwen3-14b/retrieval.json \
      --out results/appendix_I_ragas/hotpot/HyperPathsRAG/ragas.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import requests

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"

FAITHFULNESS_SYSTEM = """You are an evidence verification assistant. You receive:
- a multi-hop question
- a generated answer
- a retrieved context (knowledge-graph excerpts)

Score the faithfulness of the answer to the context on a 0.0--1.0 scale.

Definition:
  faithfulness = (number of factual claims in the answer that are
                  supported by the context) / (total claims in the answer)

If the answer is a single named entity, treat the entire answer as one
claim; mark it supported when the context contains a relation chain
that, when followed, terminates at this entity. Score 1.0 means fully
grounded; 0.0 means the answer contradicts the context or is unsupported.

Output STRICTLY this JSON on one line:
{"faithfulness": <float 0-1>, "supported_claims": <int>, "total_claims": <int>}"""

CTX_RELEVANCE_SYSTEM = """You are an evidence pruning assistant. You receive:
- a multi-hop question
- a retrieved context (knowledge-graph excerpts)

Score the context relevance on a 0.0--1.0 scale.

Definition:
  context_relevance = (number of context sentences that contribute to
                       answering the question) / (total sentences)

A sentence "contributes" if it provides one of the bridging facts of
the question's reasoning chain (named entity grounding, hop relation,
or final answer). 1.0 means every sentence is on-chain; 0.0 means no
sentence is relevant.

Output STRICTLY this JSON on one line:
{"context_relevance": <float 0-1>, "relevant_sentences": <int>, "total_sentences": <int>}"""


def _judge(system: str, user: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False, "think": False, "keep_alive": "1h",
        "options": {"temperature": 0, "num_predict": 128},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    raw = r.json()["message"].get("content", "")
    # Best-effort JSON extraction
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return {"raw": raw, "parse_error": True}
    try:
        return json.loads(m.group(0))
    except Exception as e:
        return {"raw": raw, "parse_error": True, "exc": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval",       type=str, required=True)
    ap.add_argument("--retrieval",  type=str, required=True)
    ap.add_argument("--out",        type=str, required=True)
    ap.add_argument("--model",      type=str, default=OLLAMA_MODEL,
                    help="Judge LLM (default: qwen3:14b).")
    ap.add_argument("--limit",      type=int, default=-1,
                    help="Optional cap on number of questions for cost control.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)],
                        force=True)

    eval_data = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    retr_data = json.loads(Path(args.retrieval).read_text(encoding="utf-8"))
    retr_by_q = {r.get("question"): r for r in retr_data}

    rows = eval_data["results"]
    if args.limit > 0:
        rows = rows[: args.limit]

    per_q = []
    faith_total = 0.0
    rel_total   = 0.0
    n_faith = n_rel = 0
    for i, e in enumerate(rows, 1):
        retr = retr_by_q.get(e["question"], {})
        ctx = retr.get("context") or ""
        if not ctx or not e["pred"]:
            continue

        u_faith = (f"Question: {e['question']}\n\nAnswer: {e['pred']}\n\n"
                   f"Context:\n{ctx}\n")
        u_rel   = f"Question: {e['question']}\n\nContext:\n{ctx}\n"

        j_faith = _judge(FAITHFULNESS_SYSTEM,    u_faith, args.model)
        j_rel   = _judge(CTX_RELEVANCE_SYSTEM,   u_rel,   args.model)

        f_val = j_faith.get("faithfulness")
        r_val = j_rel.get("context_relevance")
        if isinstance(f_val, (int, float)):
            faith_total += float(f_val); n_faith += 1
        if isinstance(r_val, (int, float)):
            rel_total += float(r_val); n_rel += 1

        per_q.append({
            "qid":               e["qid"],
            "faithfulness":      f_val,
            "context_relevance": r_val,
            "faith_judge":       j_faith,
            "rel_judge":         j_rel,
        })
        logging.info("[%d/%d] faith=%s ctx_rel=%s", i, len(rows), f_val, r_val)

    summary = {
        "n_questions":          len(rows),
        "faithfulness_mean":    round(faith_total / max(n_faith, 1), 4) if n_faith else None,
        "context_relevance_mean": round(rel_total / max(n_rel, 1), 4) if n_rel else None,
        "n_faith_parsed":       n_faith,
        "n_rel_parsed":         n_rel,
        "judge_model":          args.model,
        "per_question":         per_q,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    logging.info("Saved RAGAS scores → %s", args.out)
    logging.info("  faithfulness=%s  context_relevance=%s",
                 summary["faithfulness_mean"], summary["context_relevance_mean"])


if __name__ == "__main__":
    main()
