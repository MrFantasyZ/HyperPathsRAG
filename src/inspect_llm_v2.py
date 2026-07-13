"""
Two-call LLM extractor (v2) — replaces the single-call extractor in
inspect_llm.py with two focused LLM calls per chunk:

  Call A  ENTITY REFINEMENT  (input: text + NER candidates)
          Output: a refined entity list with explicit attribute + definition.
          Sub-tasks:
            A1. filter generic / over-extracted candidates
            A2. assign an attribute label from a controlled vocabulary
            A3. write a 2-5 word definition grounded in the chunk

  Call B  CHUNK-TO-FACT  (input: text + refined entity list from Call A)
          Output: a JSON array of {entities: [names], relation: "..."}
          Sub-tasks:
            B1. rewrite the chunk so every pronoun / vague back-reference
                points to a canonical entity name from the refined list
            B2. decouple the rewritten chunk into minimal independent facts
                (one atomic event per fact)

Why split:
  - Mixing entity-typing with text rewriting confuses the model: in pilot
    runs on Hillcrest / Sar-El chunks the single-call extractor either
    omitted attributes for filtered entities or skipped facts altogether.
  - Splitting also lets us cache Call A's output independently (the
    refined entity list only depends on the chunk + NER, not on later
    fact-decomposition heuristics), which is useful when iterating on
    Call B's prompt.

CLI:
  python inspect_llm_v2.py --test-text "<chunk text>" [--title "<chunk title>"]
                            [--ner-candidates "ent1,ent2,..."]
  python inspect_llm_v2.py --data <data.json> --question-idx <i>
                            [--limit-chunks <n>] [--out <out.json>]

The first form is for quick A/B comparison against the legacy extractor;
the second feeds a MuSiQue/HotpotQA-style data JSON and produces an
events JSON compatible with build_kg.py's load_events().
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
# Re-use chunking + NER + JSON parsing from the legacy extractor so that
# the only behavioural difference is the prompt structure.
from inspect_llm import (                                        # type: ignore
    chunk_passage, build_chunks, load_subset,
    extract_ner,
    _parse_json,
    _is_table_or_list,
    OLLAMA_URL, OLLAMA_MODEL,
)


# ── Controlled attribute vocabulary ─────────────────────────────────────────
# Aligned with retrieve.ATTR_FAMILIES keys so that retrieve-time virtual-slot
# matching can apply attribute-family filters without further mapping.
ATTRIBUTE_VOCAB = [
    # people / roles
    "person", "role", "profession",
    # organisations / collectives
    "organization", "company", "team", "club", "government", "institution",
    "group",
    # geography
    "country", "region", "area", "city", "location", "place",
    # time
    "year", "date", "period", "century",
    # creative works
    "film", "book", "song", "tv-show", "creative-work",
    # events / contests
    "event", "sports-event", "tournament", "match", "competition", "war",
    "treaty",
    # concepts / ideology / language
    "religion", "denomination", "ideology", "language", "word", "term",
    "meaning", "concept",
    # quantities
    "number", "quantity", "measurement",
    # fallback
    "object", "other",
]


# ── Prompt A — Entity refinement ────────────────────────────────────────────
_SYSTEM_A = """/no_think
You are an entity refinement assistant. You are given a text passage and a list of candidate entities pre-extracted by a NER model. Your ONLY job in this call is to produce a clean, typed, defined entity list — DO NOT decompose facts or rewrite the text.

Perform three sub-tasks on the entity list:

SUB-TASK A1 — FILTER GENERIC / OVER-EXTRACTED CANDIDATES. Remove:
  - Generic common nouns that denote a CLASS rather than a specific REFERENT
    (e.g. "state", "country", "city", "person", "year", "month", "language",
     "religion", "thing", "object", "place" used standalone). These are
     attribute words, not entities.
  - Bare nationality / adjectival forms when the corresponding country is
    already in the list, or when no specific country is named at all
    (e.g. drop "American", "Spanish", "European" — keep "United States",
     "Spain", "European Union").
  - Common nouns subsumed by a more specific named entity in the SAME list
    (drop "swimming pool" when "Moskva Pool" is present; drop "the company"
     when a named company is also extracted).
  - Pure-substring fragments of a longer entity already in the list.
  - Vague back-reference phrases without a specific referent
    ("place of birth", "birthplace", "the village", "the institution"
     unless the phrase IS the canonical name).
  - Single-token numerics with no semantic value as a standalone entity
    (drop "17", "1985" UNLESS the chunk text uses them as a true entity
     mention like the year a country was founded; in that case KEEP).

SUB-TASK A2 — ASSIGN ATTRIBUTE. For every kept entity, choose ONE label
from this controlled vocabulary (exact lowercase string):
{attribute_vocab}
Pick the most specific label that fits. Use "other" only as a last resort.

SUB-TASK A3 — WRITE DEFINITION. For every kept entity, write a 2-5 word
definition grounded ONLY in the passage's own context. Examples:
  Salvador Dali  → "Spanish Surrealist painter"
  Sony Music     → "American music corporation"
  1839           → "Springfield-Illinois capital year"
The definition disambiguates the entity from common namesakes
(Apple-the-company vs Apple-the-fruit).

OUTPUT FORMAT — return ONLY a valid JSON array, no markdown fences, no
explanation:
[
  {{"name": "<entity surface form>",
    "attribute": "<one label from vocabulary>",
    "definition": "<2-5 word definition>"}},
  ...
]

CONSTRAINTS:
- Use the entity's natural surface form (e.g. "India", "Mao Zedong"), NOT
  a tagged form like "India (country)" or "Mao Zedong (person)".
- "name" must appear (verbatim or as a clear alias) in the passage.
- If no meaningful entities remain after filtering, return [].
- DO NOT decompose the passage into facts in this call — that is Call B."""


# ── Prompt B — Chunk-to-Fact ────────────────────────────────────────────────
_SYSTEM_B = """/no_think
You are a fact decomposition assistant. You are given:
  (i)  the original text passage, and
  (ii) a REFINED entity list (each entity has a canonical "name", an
       "attribute", and a 2-5 word "definition").
The refined list is the ONLY entity source you may use — do NOT invent
entities outside it.

Perform two sub-tasks on the passage:

SUB-TASK B1 — REWRITE FOR EXPLICIT REFERENCE. Produce an internal rewrite
of the passage in which every pronoun, alias, or vague back-reference
points to its canonical entity name from the refined list. Examples:
  "He"                            → "<canonical name>"
  "the painter"                   → "<canonical name>"
  "the company"                   → "<canonical company name>"
  "the Arabic equivalent"         → "the Arabic equivalent of <word>"
  "the former" / "the latter"     → "<actual referent>"
The rewrite is for YOUR OWN use to produce self-contained facts in
sub-task B2; do not output the rewrite separately.

**STRICTLY FORBIDDEN — even if it "sounds cleaner":**
- DO NOT paraphrase, summarise, or shorten the sentence.
- DO NOT change vocabulary or word order beyond inserting the canonical
  name (e.g. "referred to the country of India" must stay as-is, NOT
  "referred to India (country)").
- DO NOT append NER-style type tags in parentheses.
- DO NOT merge or split sentences beyond what SUB-TASK B2 requires.

SUB-TASK B2 — DECOUPLE THE PASSAGE INTO MINIMAL FACTS.
SPLITTING RULES:
- SPLIT when a subject has multiple UNRELATED attributes
  (born in X / works at Y / nationality Z → three separate facts).
- DO NOT SPLIT when entities have a COMPARATIVE or DEPENDENT relationship
  ("born earlier than", "higher than", "together founded") — keep as one fact.
- DO NOT SPLIT when multiple entities JOINTLY participate in the same
  single event (co-founders, co-signatories) — keep as one fact.

For every fact, list the participating entities by their CANONICAL NAME
from the refined list (an entity used in the fact MUST appear in the
refined list — otherwise omit the fact or rephrase to use only refined
entities).

OUTPUT FORMAT — return ONLY a valid JSON array, no markdown fences, no
explanation:
[
  {{"entities": ["<canonical name 1>", "<canonical name 2>", ...],
    "relation": "<self-contained fact sentence from the rewritten passage>"}},
  ...
]

CONSTRAINTS:
- The "relation" sentence MUST mention EVERY entity in its "entities"
  list, and EVERY entity it mentions MUST be in the list (no extras).
- Use canonical names — DO NOT introduce surface variants not in the
  refined list (e.g. write "Sony Music Entertainment", not "SME", unless
  "SME" is the canonical name).
- Do not invent facts not present in the passage.
- If no fact can be extracted, return []."""


_USER_A = 'Text: "{text}"\nNER candidate entities: {entities}'
_USER_A_NO_NER = 'Text: "{text}"'
_USER_B = 'Text: "{text}"\nRefined entity list: {refined}'


def _call_ollama(text: str, system: str, user_content: str,
                 max_tokens: int = 8192, temperature: float = 0.0) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        "stream":     False,
        "keep_alive": "1h",
        "options":    {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx":     16384,
            "num_keep":    4096,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ── Call A wrapper ──────────────────────────────────────────────────────────

def call_refine_entities(text: str, ner_candidates: list[str] | None) -> list[dict]:
    """Run Call A. Returns refined entity list [{name, attribute, definition}]."""
    if ner_candidates:
        user_msg = _USER_A.format(
            text=text, entities=json.dumps(ner_candidates, ensure_ascii=False))
    else:
        user_msg = _USER_A_NO_NER.format(text=text)
    sys_msg = _SYSTEM_A.format(attribute_vocab=", ".join(ATTRIBUTE_VOCAB))
    raw = _call_ollama(text, sys_msg, user_msg)
    parsed = _parse_json(raw)
    if not parsed or not isinstance(parsed, list):
        return []
    # Validate shape and lower-case the attribute
    out: list[dict] = []
    for ent in parsed:
        if not isinstance(ent, dict):
            continue
        name = (ent.get("name") or "").strip()
        if not name:
            continue
        attribute = (ent.get("attribute") or "other").strip().lower()
        if attribute not in ATTRIBUTE_VOCAB:
            attribute = "other"
        definition = (ent.get("definition") or "").strip()
        out.append({"name": name, "attribute": attribute, "definition": definition})
    return out


# ── Call B wrapper ──────────────────────────────────────────────────────────

def call_decouple_facts(text: str, refined_entities: list[dict]) -> list[dict]:
    """Run Call B. Returns fact list [{entities: [names], relation: "..."}]."""
    refined_compact = [
        {"name": e["name"], "attribute": e["attribute"], "definition": e["definition"]}
        for e in refined_entities
    ]
    user_msg = _USER_B.format(
        text=text, refined=json.dumps(refined_compact, ensure_ascii=False))
    raw = _call_ollama(text, _SYSTEM_B, user_msg)
    if os.environ.get("HP_V2_DEBUG"):
        print(f"  [DEBUG Call B raw len={len(raw)}]\n--- raw start ---\n{raw[:1500]}\n--- raw end ---", flush=True)
    parsed = _parse_json(raw)
    if os.environ.get("HP_V2_DEBUG"):
        print(f"  [DEBUG parsed type={type(parsed).__name__} len={len(parsed) if parsed else 0}]", flush=True)
    if not parsed or not isinstance(parsed, list):
        return []
    name_to_attr = {e["name"]: e for e in refined_entities}
    out: list[dict] = []
    for fact in parsed:
        if not isinstance(fact, dict):
            continue
        relation = (fact.get("relation") or "").strip()
        if not relation:
            continue
        entities_in_fact: list[dict] = []
        for n in fact.get("entities") or []:
            n_str = (n or "").strip()
            if not n_str:
                continue
            # Re-attach the refined attribute + definition for downstream KG
            ent_rec = name_to_attr.get(n_str)
            if ent_rec is None:
                # Tolerate aliases that aren't in the refined list (LLM
                # sometimes uses a slightly different surface form).
                entities_in_fact.append({"name": n_str,
                                         "attribute": "other",
                                         "definition": ""})
            else:
                entities_in_fact.append({
                    "name":       ent_rec["name"],
                    "attribute":  ent_rec["attribute"],
                    "definition": ent_rec["definition"],
                })
        if not entities_in_fact:
            continue
        out.append({"entities": entities_in_fact, "relation": relation})
    return out


# ── Combined chunk extraction (Call A → Call B) ─────────────────────────────

def extract_chunk_v2(text: str, title: str | None = None,
                     ner_candidates: list[str] | None = None,
                     use_ner: bool = True,
                     log_fn=print) -> dict:
    """Run both calls on one chunk. Returns a dict with both intermediate
    outputs (refined entities) and final facts, for diagnostic / KG use."""
    if ner_candidates is None and use_ner:
        try:
            ner_candidates = extract_ner(text)
        except Exception as e:
            log_fn(f"  [WARN] NER failed: {e}")
            ner_candidates = []
    ner_candidates = ner_candidates or []

    t0 = time.time()
    refined = call_refine_entities(text, ner_candidates)
    t_a = time.time() - t0
    log_fn(f"  Call A: {len(ner_candidates)} candidates → {len(refined)} refined "
           f"({t_a:.1f}s)")

    if not refined:
        return {
            "title":           title or "",
            "text":            text,
            "ner_candidates":  ner_candidates,
            "refined_entities": [],
            "events":          [],
            "timing":          {"call_a": round(t_a, 2), "call_b": 0.0},
        }

    t0 = time.time()
    facts = call_decouple_facts(text, refined)
    t_b = time.time() - t0
    log_fn(f"  Call B: {len(refined)} refined entities → {len(facts)} facts "
           f"({t_b:.1f}s)")

    return {
        "title":            title or "",
        "text":             text,
        "ner_candidates":   ner_candidates,
        "refined_entities": refined,
        "events":           facts,
        "timing":           {"call_a": round(t_a, 2), "call_b": round(t_b, 2)},
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test-text", type=str,
                   help="Quick one-shot mode: extract from a single literal "
                        "chunk text given on the command line.")
    g.add_argument("--data", type=str,
                   help="Path to a MuSiQue/HotpotQA-style data JSON; will "
                        "extract every chunk of the selected question(s).")
    ap.add_argument("--title", type=str, default="",
                    help="(test-text mode) chunk title.")
    ap.add_argument("--ner-candidates", type=str, default=None,
                    help="(test-text mode) comma-separated NER candidates "
                         "to use instead of running NuNER-Zero.")
    ap.add_argument("--no-ner", action="store_true",
                    help="(test-text mode) skip NER entirely; let the LLM "
                         "find entities on its own.")
    ap.add_argument("--question-idx", type=int, default=None,
                    help="(data mode) only process this question index.")
    ap.add_argument("--limit-chunks", type=int, default=-1,
                    help="(data mode) cap chunks processed per question.")
    ap.add_argument("--out", type=str, default=None,
                    help="Where to write the JSON output "
                         "(default: stdout for test-text, "
                         "kg_output/llm_inspection_v2.json for data mode).")
    args = ap.parse_args()

    if args.test_text:
        ner_cands = None
        if args.ner_candidates:
            ner_cands = [s.strip() for s in args.ner_candidates.split(",") if s.strip()]
        result = extract_chunk_v2(
            text=args.test_text,
            title=args.title or None,
            ner_candidates=ner_cands,
            use_ner=(not args.no_ner) and (ner_cands is None),
        )
        out = json.dumps(result, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(out, encoding="utf-8")
            print(f"Saved → {args.out}")
        else:
            print(out)
        return

    # Data mode
    questions = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.question_idx is not None:
        questions = [questions[args.question_idx]]
    chunks = build_chunks(questions)
    if args.limit_chunks > 0:
        chunks = chunks[: args.limit_chunks]
    print(f"Processing {len(chunks)} chunks (Call A + Call B per chunk)…")
    results = []
    for i, ch in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] title={ch['title']!r} text_chars={len(ch['text'])}")
        try:
            results.append(extract_chunk_v2(text=ch["text"], title=ch["title"]))
        except Exception as e:
            print(f"  CHUNK FAILED: {e}")
            results.append({
                "title": ch["title"], "text": ch["text"],
                "ner_candidates": [], "refined_entities": [], "events": [],
                "error": str(e),
            })
    out_path = args.out or (ROOT / "llm_inspection_v2.json")
    Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
