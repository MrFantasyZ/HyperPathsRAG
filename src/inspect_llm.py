"""
LLM-only inspection script — Qwen3-14B handles entity extraction + definition +
event decoupling in a single prompt per chunk.

Output: llm_inspection.txt  (chunk text + extracted events)
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

ROOT      = Path(__file__).parent
DATA_FILE = ROOT.parent / "semantic_anchor_rag/experiment_output/multi_dataset/data/combined_300.json"
OUT_FILE  = ROOT / "llm_inspection.txt"

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"

CHUNK_MAX_SENTS = 5
CHUNK_MAX_CHARS = 800
CHUNK_MIN_SENTS = 8

# ── chunking ──────────────────────────────────────────────────────────────────

def chunk_passage(title: str, sentences: list[str]) -> list[dict]:
    total_chars = sum(len(s) for s in sentences)
    needs_split = len(sentences) > CHUNK_MIN_SENTS or total_chars > CHUNK_MAX_CHARS
    chunks = []
    if not needs_split:
        text = " ".join(sentences).strip()
        if text:
            chunks.append({"title": title, "text": text})
    else:
        buf, buf_chars = [], 0
        for sent in sentences:
            buf.append(sent)
            buf_chars += len(sent)
            if len(buf) >= CHUNK_MAX_SENTS or buf_chars >= CHUNK_MAX_CHARS:
                text = " ".join(buf).strip()
                if text:
                    chunks.append({"title": title, "text": text})
                buf, buf_chars = [], 0
        if buf:
            text = " ".join(buf).strip()
            if text:
                chunks.append({"title": title, "text": text})
    return chunks


def build_chunks(questions: list[dict]) -> list[dict]:
    chunks, seen = [], set()
    for q in questions:
        for title, sents in q.get("context", []):
            for chunk in chunk_passage(title, sents):
                if chunk["text"] not in seen:
                    seen.add(chunk["text"])
                    chunks.append(chunk)
    return chunks


def load_subset(n: int = 10) -> list[dict]:
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    sizes = [(sum(len(s) for _, sents in q.get("context", []) for s in sents), i)
             for i, q in enumerate(data)]
    sizes.sort(reverse=True)
    return [data[i] for _, i in sizes[:n]]


# ── LLM ───────────────────────────────────────────────────────────────────────

_SYSTEM = """/no_think
You are an information extraction assistant. You are given a text passage and a list of candidate entities pre-extracted by a NER model. The NER model guarantees high recall but may over-extract (e.g., it extracts both "swimming pool" and "Moskva Pool" when only "Moskva Pool" is the meaningful named entity).

Complete these four tasks:

TASK 1 — Entity filtering & selection: from the candidate entity list, keep only meaningful, specific named entities. Remove:
- Generic common nouns when a more specific named entity covering the same concept is already in the list (e.g. remove "swimming pool" if "Moskva Pool" is present)
- Fragments that are sub-parts of a longer entity already in the list
- Overly vague words with no specific referent (e.g. "construction", "war", "government" with no proper name)

TASK 2 — Entity definition: for each selected entity, write a brief 2-5 word definition using only information from the passage

TASK 3 — Event decoupling: decompose the passage into minimal, independent sub-events
SPLITTING RULES:
- SPLIT when a subject has multiple unrelated attributes (born in X / works at Y / nationality Z → three separate events)
- DO NOT SPLIT when entities have a comparative or dependent relationship ("born earlier than", "higher than", "together founded") → keep as one event
- DO NOT SPLIT when multiple entities jointly participate in the same single event (co-founders, co-signatories) → keep as one event

TASK 4 — MINIMAL coreference resolution (DO NOT REWRITE OR NORMALIZE):
**The relation MUST be a copy of the original sentence with ONLY the following changes:**

A) Pronouns must be resolved: replace pronouns (he, she, it, they, his, her, its, their, this, that, these, those) with the explicit entity name they refer to.

B) Vague noun phrases must be resolved. **THIS IS THE MOST FREQUENTLY MISSED CASE.** Scan every noun phrase that starts with "The …" and ask: "if I read this single sentence in isolation, do I know exactly what entity it refers to?" If not, you MUST resolve it by inserting the explicit antecedent.

   Sub-cases (NON-exhaustive — apply the principle, not just the patterns):

   B1) Plain back-reference: "The construction", "The organization", "The company", "The book", "The treaty", "The man", "The institution" — resolve to its named antecedent.

   B2) **Translation / equivalent / variant back-references** (especially common and easy to miss):
       "The Arabic equivalent", "The Latin name", "The local term", "The translation", "The version", "The German word",
       "The Sanskrit form", "The English name", "The original spelling", ...
       Each of these is a NOUN PHRASE whose head ("equivalent / name / term / translation") refers to some entity mentioned earlier — usually a word or name in another language. You MUST insert "of <antecedent>" so the resolved noun phrase is self-explanatory.

       Worked example:
         Original passage: "The word Hindu is derived from Sindhu, ... . The Arabic equivalent Al-Hind likewise referred to the country of India."
         BAD relation:  "The Arabic equivalent Al-Hind likewise referred to the country of India."
                         ← Ambiguous when read in isolation: "the Arabic equivalent" of WHAT?
         GOOD relation: "The Arabic equivalent of the word Hindu, Al-Hind, likewise referred to the country of India."
                         ← Now self-explanatory: equivalent of the word "Hindu". The original wording is otherwise preserved.

   B3) Comparative / ordinal back-references: "the former", "the latter", "the above-mentioned", "the aforementioned", "the first", "the second" — resolve to the actual referent.

   B4) Implicit possessive / genitive back-references: "X's [something]" where X is not specified in this sentence — resolve X.

**STRICTLY FORBIDDEN — even if it "sounds cleaner":**
- DO NOT paraphrase, summarize, or shorten the sentence.
- DO NOT change vocabulary or word order (e.g. "referred to the country of India" must stay as-is, NOT "referred to India (country)").
- DO NOT append NER-style type tags in parentheses (e.g. write "India", NOT "India (country)"; write "Mao Zedong", NOT "Mao Zedong (person)").
- DO NOT merge or split sentences beyond what TASK 3 splitting requires.
- The original article wording, definite articles ("the"), prepositions, and noun phrases MUST be preserved verbatim. ONLY add the "of <antecedent>" insertion where rule B requires it.

Example (Palace of the Soviets):
  Text: "Construction started in 1937 and was terminated by the German invasion in 1941. Its steel frame was disassembled for bridges."
  NER candidates: ["construction (facility)", "1937 (date)", "German invasion (event)", "1941 (date)", "steel frame (product)", "bridges (facility)"]
  Context: passage is about the Palace of the Soviets
  Good filtering: keep [1937, German invasion, 1941] — remove "construction" (too vague), "steel frame" and "bridges" (generic)
  Good relation: "Construction of the Palace of the Soviets started in 1937 and was terminated by the German invasion in 1941."
                 ← "Construction" is the vague noun phrase, resolved to "Construction of the Palace of the Soviets". Everything else is the original sentence verbatim.
  Good relation: "Its steel frame was disassembled for bridges." → "The steel frame of the Palace of the Soviets was disassembled for bridges."
                 ← Only the pronoun "Its" is resolved. Rest is verbatim.
  BAD relation:  "Palace of the Soviets construction (facility) was halted by German invasion (event) in 1941 (date)."  ← BAD: rephrased + NER tags added.

OUTPUT FORMAT — return ONLY a valid JSON array, no markdown fences, no explanation:
[
  {
    "entities": [
      {"name": "<entity surface form>", "definition": "<2-5 word definition>"}
    ],
    "relation": "<self-contained restatement of this sub-event>"
  },
  ...
]

CONSTRAINTS:
- Each event's "entities" list must contain ONLY the entities that are directly mentioned in that event's "relation" sentence — do NOT include all candidates in every event
- Only use entities present in the candidate list or clearly named in the passage text
- Every entity mentioned in a relation must appear in that event's "entities" list
- Do not invent entities absent from the text
- **The "relation" field MUST be the original sentence verbatim, with ONLY pronouns and vague back-references resolved. Do NOT paraphrase or add type tags.**
- The "name" field in entities must be the entity's natural surface form (e.g. "India", "Mao Zedong"), NOT a tagged form like "India (country)" or "Mao Zedong (person)".
- If no meaningful named entities or relations exist, return []"""

_SYSTEM_TABLE = """/no_think
You are an information extraction assistant. The input text is a structured table or list. The chunk TITLE is provided and is OFTEN CRUCIAL CONTEXT — it identifies what the table is about (the participants, the event, the relationship).

There are TWO types of tables. Decide which one applies before extracting:

==================== TYPE A: SINGLE-SUBJECT TABLE (infobox / stats / career record) ====================
Each row/field describes ONE entity (a person, team, organization). The title usually IS that entity.
For each attribute (birth date, position, nationality, award, tenure, score), create ONE event:
"<subject> <attribute-phrase> <value>" (e.g. "Derek Jeter was born on June 26, 1974").

==================== TYPE B: RELATIONAL TABLE (match list / head-to-head / co-occurrence list) ====================
**Recognise this type when the TITLE names a recurring relationship between two parties**, such as:
  - "Second City derby" → matches between Birmingham City and Aston Villa
  - "El Clásico" → matches between Real Madrid and Barcelona
  - "Iron Bowl" → games between Auburn Tigers and Alabama Crimson Tide
  - "X–Y rivalry", "X vs Y matches", "List of X-Y games", ...
or when the column header lists only ONE participant (e.g., "Home team") and the OTHER participant is implied by the title.

For each row in a TYPE-B table you MUST extract MULTIPLE relational events:

  Suppose the title is "Second City derby" (= Birmingham City vs Aston Villa).
  The row gives: Date=1 December 2010, Venue=St Andrew's, Home team=Birmingham City, Score=2-1, Competition=League Cup, Round=Quarter Final, Attendance=27,679.
  The IMPLICIT away team is Aston Villa (the OTHER party from the title).
  Home scored 2, Away scored 1, so HOME won.

  Extract:
  1. The participation fact:
     "Birmingham City played Aston Villa on 1 December 2010 at St Andrew's in the League Cup Quarter Final."
  2. The result fact (winning team is the SUBJECT of "beat"):
     "Birmingham City beat Aston Villa 2-1 on 1 December 2010 in the League Cup Quarter Final."
       (Use "Home team drew with Away team N-N on …" if scores tie. Use "Away team beat Home team A-B on …" if Away > Home.)
  3. Optionally: the attendance/venue fact:
     "The 1 December 2010 League Cup Quarter Final between Birmingham City and Aston Villa at St Andrew's had an attendance of 27,679."

==================== INPUT FORMAT ====================
You will receive two fields:
  Title: "<chunk title>"
  Text:  "<table or list text>"

==================== OUTPUT FORMAT ====================
Return ONLY a valid JSON array, no markdown fences, no explanation:
[
  {"entities": [{"name": "<entity>", "definition": "<2-5 word definition>"}, ...],
   "relation": "<self-contained fact sentence>"},
  ...
]

==================== EXAMPLES ====================

Example A (single-subject infobox):
  Title: "Derek Jeter"
  Text:  "Derek Jeter  Shortstop  Born: (1974-06-26) June 26, 1974  New York Yankees  Awards: 5× World Series"
  Output:
  [
    {"entities": [{"name": "Derek Jeter", "definition": "baseball shortstop"}, {"name": "June 26, 1974", "definition": "Derek Jeter birth date"}], "relation": "Derek Jeter was born on June 26, 1974"},
    {"entities": [{"name": "Derek Jeter", "definition": "baseball shortstop"}, {"name": "New York Yankees", "definition": "MLB baseball team"}], "relation": "Derek Jeter played for the New York Yankees"},
    {"entities": [{"name": "Derek Jeter", "definition": "baseball shortstop"}], "relation": "Derek Jeter played as a shortstop"},
    {"entities": [{"name": "Derek Jeter", "definition": "baseball shortstop"}, {"name": "World Series", "definition": "MLB championship title"}], "relation": "Derek Jeter won 5 World Series titles"}
  ]

Example B (relational match list — note the IMPLICIT away team from the title):
  Title: "Second City derby"   (Birmingham City vs Aston Villa)
  Text:  "Date Venue Home team Score Competition Round Attendance 1 December 2010 St Andrew's Birmingham City 2 -- 1 League Cup Quarter Final 27,679"
  Output:
  [
    {"entities": [{"name": "Birmingham City", "definition": "English football club"}, {"name": "Aston Villa", "definition": "English football club"}, {"name": "1 December 2010", "definition": "match date"}, {"name": "St Andrew's", "definition": "Birmingham City home ground"}, {"name": "League Cup", "definition": "English football cup competition"}], "relation": "Birmingham City played Aston Villa on 1 December 2010 at St Andrew's in the League Cup Quarter Final"},
    {"entities": [{"name": "Birmingham City", "definition": "English football club"}, {"name": "Aston Villa", "definition": "English football club"}, {"name": "1 December 2010", "definition": "match date"}, {"name": "League Cup", "definition": "English football cup competition"}], "relation": "Birmingham City beat Aston Villa 2-1 on 1 December 2010 in the League Cup Quarter Final"},
    {"entities": [{"name": "Birmingham City", "definition": "English football club"}, {"name": "Aston Villa", "definition": "English football club"}, {"name": "27,679", "definition": "match attendance"}], "relation": "The 1 December 2010 League Cup Quarter Final between Birmingham City and Aston Villa at St Andrew's had an attendance of 27,679"}
  ]

CONSTRAINTS:
- Each relation must be a complete, self-contained sentence with no pronouns and no vague references.
- Use the natural surface form of entity names, NEVER tagged forms like "Birmingham City (organization)".
- For TYPE B, ALWAYS extract the participation event AND the result (beat / drew / lost) event for every row.
- Do not invent facts not present in the text.
- If the text is empty or unparseable, return []."""

_USER          = 'Text: "{text}"\nCandidate entities from NER: {entities}'
_USER_NO_NER   = 'Text: "{text}"'   # fallback when NER produced nothing


def _is_table_or_list(text: str) -> bool:
    """Heuristic: detect structured table/list/infobox text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    # Infobox date pattern: Born: (YYYY-MM-DD)
    if re.search(r"Born\s*:\s*\(\d{4}-\d{2}-\d{2}\)", text):
        return True
    # Header-like patterns common in sports/stats tables
    if re.search(r"\b(Nat|Tenure|Honours|Founded|Position|Manager)\b", text) and len(lines) >= 3:
        return True
    # Many short lines with no verb (list-like)
    short_lines = sum(1 for l in lines if len(l) < 80 and not re.search(r"\b(is|was|were|has|have|had|are)\b", l, re.I))
    if len(lines) >= 4 and short_lines / len(lines) > 0.7:
        return True
    return False


# ── NuNER Zero ────────────────────────────────────────────────────────────────

NER_LABELS = [
    "person", "organization", "location", "date", "time",
    "work of art", "event", "product", "law", "language",
    "nationality", "religion", "title", "facility", "number", "country",
]

_ner_model = None

def get_ner_model():
    global _ner_model
    if _ner_model is None:
        from gliner import GLiNER
        print("Loading NuNER Zero …", flush=True)
        _ner_model = GLiNER.from_pretrained("numind/NuNER_Zero")
        _ner_model.eval()
    return _ner_model


def _merge_adjacent(preds: list[dict]) -> list[dict]:
    if not preds:
        return preds
    preds = sorted(preds, key=lambda x: x["start"])
    merged = [dict(preds[0])]
    for cur in preds[1:]:
        prev = merged[-1]
        if cur["label"] == prev["label"] and cur["start"] - prev["end"] <= 1:
            prev["end"]   = cur["end"]
            prev["text"]  = (prev["text"] + " " + cur["text"]).strip()
            prev["score"] = max(prev["score"], cur["score"])
        else:
            merged.append(dict(cur))
    return merged


def extract_ner(text: str, threshold: float = 0.4) -> list[str]:
    """Return list of 'entity_text (label)' strings."""
    model = get_ner_model()
    preds = model.predict_entities(text, NER_LABELS, threshold=threshold)
    preds = _merge_adjacent(preds)
    seen, result = set(), []
    for p in preds:
        name = p["text"].strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            result.append(f"{name} ({p['label']})")
    return result


# ── Ollama LLM ────────────────────────────────────────────────────────────────

def _call_ollama(text: str, system: str, entities: list[str] | None = None,
                 max_tokens: int = 8192, temperature: float = 0,
                 title: str | None = None) -> str:
    import requests
    if entities:
        user_content = _USER.format(text=text, entities=json.dumps(entities, ensure_ascii=False))
    elif title:
        # Table prompt path: include chunk title as critical relational context
        user_content = f'Title: "{title}"\nText: "{text}"'
    else:
        user_content = _USER_NO_NER.format(text=text)
    payload = {
        "model":      OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        "stream":     False,
        "keep_alive": "1h",          # keep model + KV cache in VRAM between calls
        "options":    {
            "temperature": temperature,
            "num_predict": max_tokens,
            # Explicit context window: must fit system_prompt (~4k tokens) +
            # user_content (~2k) + num_predict output (≤8k). Default 2048 is
            # WAY too small and silently truncates our long prompts, which
            # breaks Ollama's prefix-KV-cache reuse (truncated prefix differs
            # from previous calls → cache miss → full recompute every chunk).
            "num_ctx":     16384,
            # Always keep the FULL system prompt in the KV cache so subsequent
            # chunks share the same cached prefix and only the user message
            # is freshly computed. 4096 is comfortably above our system prompt size.
            "num_keep":    4096,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _parse_json(raw: str) -> list[dict] | None:
    """Strip think tags + fences, then parse JSON array."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return None


_SYSTEM_RETRY = """/no_think
Extract named entities and events from the text. Return ONLY a valid JSON array, no markdown:
[{"entities": [{"name": "<entity natural surface form, no type tags>", "definition": "<2-5 word definition>"}], "relation": "<original sentence VERBATIM with only pronouns and vague back-references resolved>"}]
Rules:
- Split unrelated attributes into separate events; keep joint/comparative events together.
- Each event's entities list contains only entities mentioned in that relation.
- The relation MUST preserve the original wording. Only resolve pronouns / vague back-references. DO NOT paraphrase or add NER type tags like "(country)", "(person)"."""

def _retry_thinking(text: str, system: str, entities: list[str] | None = None,
                    title: str | None = None) -> tuple[str, list[dict] | None]:
    """Retry with a simpler prompt (avoids long thinking chains consuming token budget)."""
    raw = _call_ollama(text, system=_SYSTEM_RETRY, entities=entities,
                       max_tokens=4096, temperature=0, title=title)
    return raw, _parse_json(raw)


def process_chunk(chunk: dict) -> dict:
    """Return {title, text, ner_entities, events, raw, status, elapsed, prompt_type}."""
    t0 = time.time()
    is_table = _is_table_or_list(chunk["text"])
    prompt_type = "table" if is_table else "normal"

    if is_table:
        # Table/list: skip NER, use dedicated prompt
        system   = _SYSTEM_TABLE
        entities = None
    else:
        # Normal text: NER first, then LLM with candidate list
        system   = _SYSTEM
        entities = extract_ner(chunk["text"])

    # For table chunks, pass the chunk title as critical context (often names
    # the implicit participants of a relational table).
    title_arg = chunk.get("title") if is_table else None
    raw    = _call_ollama(chunk["text"], system=system, entities=entities, title=title_arg)
    events = _parse_json(raw)
    status = "ok"

    if events is None:
        raw2, events = _retry_thinking(chunk["text"], system=system, entities=entities, title=title_arg)
        raw = raw2
        status = "retry_ok" if events is not None else "failed"

    if events is None:
        events = []

    # validate structure
    valid = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ents = ev.get("entities", [])
        rel  = ev.get("relation", "").strip()
        if not rel:
            continue
        clean_ents = [
            {"name": e["name"].strip(), "definition": e["definition"].strip()}
            for e in ents
            if isinstance(e, dict) and e.get("name") and e.get("definition")
        ]
        if clean_ents:
            valid.append({"entities": clean_ents, "relation": rel})

    if valid:
        status = "ok" if status == "ok" else "retry_ok"
    elif status == "ok":
        status = "empty"

    return {
        "title":        chunk["title"],
        "text":         chunk["text"],
        "ner_entities": entities or [],
        "events":       valid,
        "raw":          raw,
        "status":       status,
        "prompt_type":  prompt_type,
        "elapsed":      round(time.time() - t0, 1),
    }


# ── report ────────────────────────────────────────────────────────────────────

def write_report(results: list[dict], out_file: Path) -> None:
    counts = {"ok": 0, "retry_ok": 0, "empty": 0, "failed": 0}
    for r in results:
        counts[r["status"]] += 1

    lines = []
    lines.append("LLM Inspection Report — Qwen3-14B  (entity extraction + definition + event decoupling)")
    lines.append(f"Total chunks : {len(results)}")
    lines.append(f"Status       : ok={counts['ok']}  retry_ok={counts['retry_ok']}  "
                 f"empty={counts['empty']}  failed={counts['failed']}")
    lines.append("=" * 80)
    lines.append("")

    for i, r in enumerate(results):
        flag  = {"ok": "", "retry_ok": "[RETRY]", "empty": "[EMPTY]", "failed": "[FAILED]"}[r["status"]]
        ptype = "[TABLE]" if r.get("prompt_type") == "table" else ""
        lines.append(f"[Chunk {i+1}/{len(results)} | {r['elapsed']}s | {flag}{ptype} | Title: {r['title']}]")
        lines.append(f"TEXT: {r['text']}")
        if r["ner_entities"]:
            lines.append(f"NER candidates: {' | '.join(r['ner_entities'])}")
        lines.append("")

        if r["events"]:
            for j, ev in enumerate(r["events"]):
                ent_str = " | ".join(f"{e['name']} ({e['definition']})" for e in ev["entities"])
                lines.append(f"  Event {j+1}: {ev['relation']}")
                lines.append(f"    Entities: {ent_str}")
        else:
            lines.append("  <no events extracted>")
            if r["status"] == "failed":
                # show raw output truncated for debugging
                raw_preview = r["raw"][:300].replace("\n", " ")
                lines.append(f"  RAW: {raw_preview}…")

        lines.append("-" * 80)
        lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written → {out_file}")
    print(f"Summary: {counts}")


def save_json(results: list[dict], out_file: Path) -> None:
    serializable = []
    for r in results:
        serializable.append({k: v for k, v in r.items() if k != "raw"})
    out_file.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="Path to input JSON (list of questions with context). "
                         "Defaults to combined_300.json relative to source layout.")
    ap.add_argument("--out", type=str, default=None,
                    help="Path to output llm_inspection.json. Default: "
                         "<source-dir>/llm_inspection.json")
    ap.add_argument("--n", type=int, default=None,
                    help="If set, only the top-N largest questions are extracted "
                         "(legacy debug mode). Default: ALL questions in --data.")
    args = ap.parse_args()

    if args.data:
        DATA_FILE = Path(args.data)
    if args.out:
        OUT_FILE = Path(args.out)
        if OUT_FILE.suffix == ".json":
            OUT_FILE = OUT_FILE.with_suffix(".txt")  # write_report needs .txt

    print(f"Loading from {DATA_FILE} …")
    if args.n is not None:
        questions = load_subset(args.n)
    else:
        questions = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} questions.")
    chunks    = build_chunks(questions)
    print(f"Built {len(chunks)} chunks — calling Qwen3-14B …\n")

    results = []
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {chunk['text'][:70]}…", flush=True)
        result = process_chunk(chunk)
        results.append(result)
        print(f"    → {len(result['events'])} events  ({result['elapsed']}s)  [{result['status']}]", flush=True)

    write_report(results, OUT_FILE)
    save_json(results, OUT_FILE.with_suffix(".json"))
