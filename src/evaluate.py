"""
End-to-end QA evaluation: feed retrieved context + question to Qwen3 → answer
→ compute Exact Match (EM) and F1 vs ground truth.

Usage:
    python evaluate.py [--results retrieval_results.json] [--out evaluation_results.json]

Pipeline:
  1. Load retrieval_results.json (output of retrieve.py)
  2. For each question, call Qwen3-14B with the retrieved context to generate an answer
  3. Compute SQuAD-style EM and F1 on (predicted vs expected_answer)
  4. Save per-question results + aggregate scores
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import requests

# ── config ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL  = "qwen3:14b"

logger = logging.getLogger(__name__)


# ── LLM answer generation ────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────
# ROLLBACK: previous stable prompt that yielded 70% EM (the "70_baseline").
# To roll back, swap _ANSWER_SYSTEM with _ANSWER_SYSTEM_70_BASELINE below.
# ───────────────────────────────────────────────────────────────────────
_ANSWER_SYSTEM_70_BASELINE = """You are a question-answering assistant for multi-hop QA. You will be given a context (excerpts from a knowledge graph) and a question.

Task: answer the question. The context contains the evidence you need, often spread across multiple sentences — connect them by following entity names that appear in more than one sentence.

Reasoning rules:
- The context may not state the answer literally. Often you must chain 2-4 facts together: e.g. if the question asks about "person X's country" and the context says "X received the USSR State Prize" and "the USSR practiced colonialism", then X's country is the Soviet Union / USSR.
- Cross-reference entity names: when the question describes someone indirectly (e.g. "the man who reformed institution Y"), look in the context for a sentence that links a named person to action on Y.
- If two answers are both plausible, pick the one most directly supported by the context's wording.
- **You MUST traverse the FULL reasoning chain end-to-end. Do not stop at an intermediate hop**, even if that intermediate entity is plausible. Always ask: "Did I reach the FINAL hop the question is asking about?"
- For "what is the meaning of word X in language Y" style questions: look for sentences like "X (referred to / equivalent to / meaning) Z" and answer Z (the referent), NOT X itself or its translation. The translation is an intermediate hop.
- Only output "unknown" if you have genuinely tried multi-hop reasoning and the context contradicts itself or omits a critical bridging fact.

Output format: emit a SHORT reasoning chain followed by the final answer, on these two lines exactly:
```
Reasoning: <hop 1> -> <hop 2> -> ... -> <final entity>
Answer: <final answer>
```

The reasoning chain MUST be one line of `A -> B -> C -> D` arrows, ≤15 words total. The Answer line must contain ONLY the final answer as the shortest correct noun phrase (no explanation, no quotes, no markdown):
- Person → name only ("Martin Luther")
- Date   → date only ("1 December 2010")
- Place  → place name only ("Rachel, Nevada")
- Organization → official name ("the Politburo")
- Usually 1-6 words.

Example 1:
Context: ... "Hollis was Small Heath's goalkeeper" ... "Birmingham City beat Aston Villa 2-1 on 1 December 2010" ...
Question: When was the last time George Hollis's team beat the 1894-95 FA Cup winner?
Reasoning: Hollis -> Small Heath (= Birmingham City) -> beat Aston Villa -> 1 December 2010
Answer: 1 December 2010

Example 2:
Context: ... "BCCI is based in India" ... "Area A became India in 1947" ... "The Arabic equivalent of 'Hindu', Al-Hind, referred to the country of India" ...
Question: What is the meaning of the word that is also the majority religion in the area that became India when the country where BCCI is based was created in the Arabic dictionary?
Reasoning: BCCI -> India -> 1947 -> British India -> Hindu -> Al-Hind -> the country of India
Answer: the country of India"""


_ANSWER_SYSTEM = """You are a question-answering assistant for multi-hop QA. You will be given a context (excerpts from a knowledge graph) and a question.

Task: answer the question. The context contains the evidence you need, often spread across multiple sentences — connect them by following entity names that appear in more than one sentence.

Reasoning rules:
- The context may not state the answer literally. Often you must chain 2-4 facts together: e.g. if the question asks about "person X's country" and the context says "X received the USSR State Prize" and "the USSR practiced colonialism", then X's country is the Soviet Union / USSR.
- Cross-reference entity names: when the question describes someone indirectly (e.g. "the man who reformed institution Y"), look in the context for a sentence that links a named person to action on Y.
- **You MUST traverse the FULL reasoning chain end-to-end. Do not stop at an intermediate hop**, even if that intermediate entity is plausible. Always ask: "Did I reach the FINAL hop the question is asking about?"
- For "what is the meaning of word X in language Y" style questions: look for sentences like "X (referred to / equivalent to / meaning) Z" and answer Z (the referent), NOT X itself or its translation. The translation is an intermediate hop.
- **Candidate disambiguation** — when the context offers MULTIPLE candidates for the same hop (e.g. several "lowest batting average" mentions, several "earliest film", several "first man to do X"):
  1. Re-read the question's full constraint chain (e.g. "lowest batting average IN THE LEAGUE THAT a specific team plays in"); a candidate that does not satisfy ALL constraints is wrong even if its sentence sounds the most prominent.
  2. If the question asks for a superlative (lowest / highest / oldest / first), compare the ACTUAL NUMERIC values or DATES across candidates — do not just match the keyword in the most-recent sentence. E.g. ".170 < .179" so the .170 candidate is "lower" even if the .179 candidate's sentence repeats the word "lowest" more prominently.
  3. Prefer the candidate whose attributes (career vs single-season, league, time period) match the question's wording, not the candidate whose sentence merely contains the most question keywords.
- Only output "unknown" if you have genuinely tried multi-hop reasoning and the context contradicts itself or omits a critical bridging fact.

Output format: emit a SHORT reasoning chain followed by the final answer, on these two lines exactly:
```
Reasoning: <hop 1> -> <hop 2> -> ... -> <final entity>
Answer: <final answer>
```

The reasoning chain MUST be one line of `A -> B -> C -> D` arrows, ≤15 words total. The Answer line must contain ONLY the final answer as the shortest correct noun phrase (no explanation, no quotes, no markdown):
- Person → name only ("Martin Luther")
- Date   → date only ("1 December 2010")
- Place  → place name only ("Rachel, Nevada")
- Organization → official name ("the Politburo")
- Usually 1-6 words.

Example 1:
Context: ... "Hollis was Small Heath's goalkeeper" ... "Birmingham City beat Aston Villa 2-1 on 1 December 2010" ...
Question: When was the last time George Hollis's team beat the 1894-95 FA Cup winner?
Reasoning: Hollis -> Small Heath (= Birmingham City) -> beat Aston Villa -> 1 December 2010
Answer: 1 December 2010

Example 2:
Context: ... "BCCI is based in India" ... "Area A became India in 1947" ... "The Arabic equivalent of 'Hindu', Al-Hind, referred to the country of India" ...
Question: What is the meaning of the word that is also the majority religion in the area that became India when the country where BCCI is based was created in the Arabic dictionary?
Reasoning: BCCI -> India -> 1947 -> British India -> Hindu -> Al-Hind -> the country of India
Answer: the country of India

Example 3 (candidate disambiguation with numeric comparison):
Context: ... "(Los Angeles Dodgers) Dodgers won 21 National League pennants." ... "(Batting average) Rob Deer hit .179 in 1991 — lowest single-season batting average for a qualified player." ... "(Batting average) Bill Bergen recorded a .170 career average in 3028 at-bats — lowest career batting average among players with 2500+ at-bats." ...
Question: Who has the lowest batting average in the league that the team that won the most titles plays in?
Reasoning: Dodgers -> National League -> compare career averages -> .170 (Bergen) < .179 (Deer) -> Bill Bergen
Answer: Bill Bergen"""


def _build_user_msg(context: str, question: str, priming: str) -> str:
    """Build the user prompt according to CoT priming order (Appendix G).

    priming ∈ {reasoning-first, answer-first, answer-only}:
      reasoning-first  → primes "Reasoning:" (default, paper main result)
      answer-first     → primes "Answer:"   (ablation: LLM emits guess first)
      answer-only      → no CoT priming     (ablation: forces direct answer)
    """
    base = f"Context:\n{context}\n\nQuestion: {question}\n\n"
    if priming == "reasoning-first":
        return base + "Reasoning:"
    if priming == "answer-first":
        return base + "Answer:"
    if priming == "answer-only":
        return base + "Give the final answer only, no reasoning.\nAnswer:"
    return base + "Reasoning:"


def _call_once(context: str, question: str, enable_think: bool, num_predict: int,
               model: str | None = None, priming: str = "reasoning-first") -> tuple[str, str, dict]:
    """Single LLM call. Returns (content, done_reason, token_stats).

    token_stats has keys {prompt_tokens, completion_tokens, total_tokens}.
    """
    user_msg = _build_user_msg(context, question, priming)
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "stream":     False,
        "think":      enable_think,
        "keep_alive": "1h",
        "options":    {"temperature": 0, "num_predict": num_predict},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    resp = r.json()
    # Ollama reports prompt_eval_count / eval_count; OpenAI proxy may
    # return usage{prompt_tokens, completion_tokens}. We unify both.
    pt = resp.get("prompt_eval_count") or resp.get("usage", {}).get("prompt_tokens") or 0
    ct = resp.get("eval_count")        or resp.get("usage", {}).get("completion_tokens") or 0
    tokens = {"prompt_tokens": int(pt), "completion_tokens": int(ct),
              "total_tokens": int(pt) + int(ct)}
    return resp["message"].get("content", "").strip(), resp.get("done_reason", ""), tokens


def _extract_answer(raw: str) -> tuple[str, str]:
    """Parse the LLM's multi-paragraph CoT output into (reasoning, answer).

    Expected format:
        Para 1 reasoning: A -> B
        Para 1 answer: X
        Para 2 reasoning: C -> D
        Para 2 answer: Y
        Final reasoning: <synthesis>
        Answer: <final>

    Parsing strategy:
      1. The FINAL answer is the line `Answer: ...` (NOT a `Para N answer:`
         line, since those start with "Para N").
      2. For the reasoning log, concatenate the per-paragraph and final
         reasoning lines (capped to a readable length).
      3. Robust fallbacks: if the LLM emits the old single-paragraph format
         (`Reasoning: ... / Answer: ...`), we still extract correctly.
    """
    txt = re.sub(r"```[a-zA-Z]*|```", "", raw).strip()
    answer = ""
    reasoning_parts: list[str] = []

    # Find FINAL Answer: line (lines starting with "answer:" or "final answer:",
    # NOT "Para N answer:"). Take the last one.
    final_answer_matches = re.findall(
        r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", txt
    )
    if final_answer_matches:
        answer = final_answer_matches[-1].strip()

    # Reasoning capture: aggregate all "Reasoning:" / "Para N reasoning:" /
    # "Final reasoning:" lines.
    for m in re.finditer(
        r"(?im)^\s*(?:(?:para\s+\d+|final)\s+)?reasoning\s*:\s*(.+?)\s*$", txt
    ):
        reasoning_parts.append(m.group(1).strip())

    # Fallback: no Answer line found → take last non-empty line.
    if not answer:
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if lines:
            answer = re.sub(
                r"^(answer|final answer|para\s+\d+\s+answer)\s*:\s*", "",
                lines[-1], flags=re.IGNORECASE
            ).strip()

    answer = answer.strip(' "\'`')
    reasoning = " || ".join(reasoning_parts)[:300]
    return reasoning, answer


def call_ollama(context: str, question: str, enable_think: bool = True,
                model: str | None = None, priming: str = "reasoning-first") -> tuple[str, str, dict]:
    """Returns (reasoning, answer, token_stats)."""
    raw, done, tokens = _call_once(context, question, enable_think=enable_think,
                                    num_predict=4096, model=model, priming=priming)
    if not raw and enable_think:
        logger.warning("Empty content (done_reason=%s) — retrying with think=False", done)
        raw, _, tokens2 = _call_once(context, question, enable_think=False,
                                      num_predict=1024, model=model, priming=priming)
        # Add tokens from the retry to the running total
        tokens = {k: tokens.get(k, 0) + tokens2.get(k, 0) for k in tokens2}
    reasoning, answer = _extract_answer(raw)
    return reasoning, answer, tokens


# ── SQuAD-style metrics ──────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lowercase, strip articles + punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def em_score(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    prec = n_common / len(p_toks)
    rec  = n_common / len(g_toks)
    return 2 * prec * rec / (prec + rec)


# ── driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "retrieval_results.json"))
    ap.add_argument("--out",     default=str(ROOT / "evaluation_results.json"))
    ap.add_argument("--no-think", action="store_true",
                    help="Disable Qwen3 thinking for faster but lower-quality answers")
    ap.add_argument("--model", type=str, default=None,
                    help="Override Ollama model name (e.g. qwen3:14b, "
                         "llama3.3:70b-instruct-q4_K_M). Defaults to OLLAMA_MODEL.")
    ap.add_argument("--priming", type=str, default="reasoning-first",
                    choices=["reasoning-first", "answer-first", "answer-only"],
                    help="CoT priming order (Appendix G ablation).")
    ap.add_argument("--save-tokens", type=str, default=None,
                    help="Optional path for per-question token-usage summary "
                         "(populates Table 4 Efficiency tokens column).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)],
                        force=True)

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    n = len(results)
    logger.info("Loaded %d retrieval results from %s", n, args.results)

    eval_rows = []
    em_total, f1_total = 0.0, 0.0
    t_start = time.time()

    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for i, r in enumerate(results):
        question = r["question"]
        gold     = (r.get("expected_answer") or "").strip()
        context  = (r.get("context") or "").strip()

        t0 = time.time()
        reasoning = ""
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not context:
            pred = ""
            err  = "no_context"
        else:
            try:
                reasoning, pred, tokens = call_ollama(
                    context, question,
                    enable_think=not args.no_think,
                    model=args.model,
                    priming=args.priming,
                )
                err  = ""
            except Exception as e:
                pred = ""
                err  = f"llm_call_failed: {e}"
                logger.exception("Q%d LLM call failed", i + 1)
        elapsed = time.time() - t0

        em = em_score(pred, gold)
        f1 = f1_score(pred, gold)
        em_total += em
        f1_total += f1
        for k in token_totals:
            token_totals[k] += tokens.get(k, 0)

        mark = "✅" if em else ("◐" if f1 > 0.3 else "❌")
        logger.info("%s Q%d  EM=%d F1=%.2f  (%.1fs, %dt)  gold='%s'  pred='%s'",
                    mark, i + 1, em, f1, elapsed, tokens.get("total_tokens", 0),
                    gold[:50], pred[:80])
        if reasoning:
            logger.info("       reasoning: %s", reasoning[:200])

        eval_rows.append({
            "qid":            r.get("question", ""),
            "question":       question,
            "gold":           gold,
            "pred":           pred,
            "reasoning":      reasoning,
            "em":             em,
            "f1":             round(f1, 3),
            "context_len":    len(context),
            "n_paths":        r.get("n_paths", 0),
            "elapsed_sec":    round(elapsed, 1),
            "tokens":         tokens,
            "error":          err,
        })

    em_pct = em_total / n * 100 if n > 0 else 0
    f1_pct = f1_total / n * 100 if n > 0 else 0
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY (%d questions, total %.1fs)", n, time.time() - t_start)
    logger.info("  EM:  %.1f%%  (%d/%d)", em_pct, int(em_total), n)
    logger.info("  F1:  %.1f%%", f1_pct)
    logger.info("  Tokens (mean/q): prompt=%.0f completion=%.0f total=%.0f",
                token_totals["prompt_tokens"] / max(n, 1),
                token_totals["completion_tokens"] / max(n, 1),
                token_totals["total_tokens"] / max(n, 1))
    logger.info("=" * 60)

    out_summary = {
        "n":           n,
        "em":          round(em_pct, 2),
        "f1":          round(f1_pct, 2),
        "model":       args.model or OLLAMA_MODEL,
        "priming":     args.priming,
        "tokens_per_q": {
            "prompt_tokens":     round(token_totals["prompt_tokens"] / max(n, 1), 1),
            "completion_tokens": round(token_totals["completion_tokens"] / max(n, 1), 1),
            "total_tokens":      round(token_totals["total_tokens"] / max(n, 1), 1),
        },
        "results":     eval_rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(out_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved per-question evaluation → %s", args.out)

    if args.save_tokens:
        Path(args.save_tokens).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_tokens).write_text(
            json.dumps({
                "n":           n,
                "model":       args.model or OLLAMA_MODEL,
                "priming":     args.priming,
                "tokens_per_q": out_summary["tokens_per_q"],
                "tokens_total": dict(token_totals),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved token summary → %s", args.save_tokens)


if __name__ == "__main__":
    main()
