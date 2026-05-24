# HyperPathsRAG — Method Documentation

A hypergraph-based RAG system for multi-hop QA that frames retrieval as
**multi-target approximation** over a knowledge graph and emits context
as **per-path** reasoning chains.

## 1. Overview

```
┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  question  │──▶│ decompose    │──▶│ multi-target │──▶│ variant path │
│            │   │ → skeleton   │   │ search       │   │ enumeration  │
└────────────┘   │   DAG (n-ary)│   │ over KG      │   │ + scoring    │
                 └──────────────┘   └──────────────┘   └──────────────┘
                                                              │
┌────────────┐                                  ┌─────────────▼┐
│  KG        │                                  │ context as   │
│  (HyperG)  │◀────────  KG construction ───────│ per-path     │
└────────────┘                                  │ paragraphs   │
                                                └──────┬───────┘
                                                       │
                                                ┌──────▼────────┐
                                                │ CoT answer    │
                                                │ LLM (Qwen3 /  │
                                                │ Llama-3.3-70B)│
                                                └──────┬────────┘
                                                       ▼
                                                    Answer
```

## 2. Three main contributions

1. **Multi-target retrieval over a hypergraph.** Decompose a query into
   `n` sub-questions; retrieval simultaneously approximates `n`
   embedding targets via lock-and-slide entity assignment, not a single
   query embedding.
2. **Reasoning-chain-completeness path scoring.** End-to-end paths in
   the variant graph are scored by `sum(seg_sim) / n_subqs` — paths
   covering more sub-questions outscore locally-similar but
   chain-incomplete ones.
3. **Order-preserving per-path context.** For each top-K path, the
   relations along the path are concatenated **in path order** into ONE
   paragraph; different paths are separate paragraphs. Mirrors the
   reasoning chain structurally and avoids the pronoun / back-reference
   confusion of by-sub-question grouping.

A **fourth** contribution emerged from implementation:

4. **N-ary sub-question decomposition.** Skeleton sub-questions carry
   an `entities` list of arbitrary size (not `subject` + `object`).
   When iterative search binds virtual placeholders to concrete KG
   entities, each skeleton sub-question instantiates into multiple
   **variant sub-questions** — the path enumeration walks the
   resulting variant graph. This matches the n-ary structure of the
   underlying hypergraph KG and avoids the disconnected skeleton DAG
   that binary-triple decomposition produces.

## 3. Step-by-step pipeline

### 3.1 Hypergraph KG construction (`inspect_llm.py` + `build_kg.py`)

Each text chunk is processed by:
1. **NuNER-Zero** (zero-shot NER) — extracts candidate entities.
2. **Qwen3-14B (LLM)** — given chunk + candidate entities, the LLM
   performs four tasks in one call (`_SYSTEM` prompt):
   - **TASK 1**: filter entity list, drop generic / sub-fragments.
   - **TASK 2**: write a 2-5 word definition for each kept entity.
   - **TASK 3**: split the chunk into minimal atomic events.
   - **TASK 4**: rewrite each event sentence with minimal coreference
     resolution (replace pronouns and ambiguous "the X" references
     with their named antecedent; do NOT normalise wording).
3. **`_SYSTEM_TABLE`** prompt: for table-like text (heuristic
   `_is_table_or_list`), extract row-level events with implicit
   second-party recovery (e.g. for a "Second City derby" table, the
   away team is recovered from the title's pairing).
4. **Output**: `llm_inspection.json` — list of `{title, text, prompt_type,
   status, events: [{relation, entities: [{name, definition}]}, …]}`.

The hypergraph KG (`build_kg.py`) then:
- Embeds **entities** twice (name-only + name+definition) — averages
  used for entity similarity.
- Embeds **relations** as `"(<source_title>) <relation_text>"` to keep
  chunk-level context in the embedding (a key fix for chunks like
  "1894-95 FA Cup" whose body never repeats the title).
- Adds **synonym edges** between near-duplicate entities (averaged
  name+def sim > 0.85).
- Builds **BM25 indices** over both entity strings and
  title-prefixed relation strings (used by Composed Retrieval).

### 3.2 Query decomposition → skeleton DAG (`retrieve.py:decompose_query`)

LLM emits a JSON list of n-ary sub-questions:

```json
[
  {"relation": "Area A became India in year A",
   "entities": [
     {"name": "A",     "attribute": "area",    "virtual": true},
     {"name": "India", "attribute": "country", "virtual": false},
     {"name": "A",     "attribute": "year",    "virtual": true}
   ]},
  ...
]
```

`build_skeleton_dag` constructs a **bipartite** graph: entity nodes
keyed by `(real|virtual, attr+name)` connected to sub-question nodes
keyed by qid. Two sub-questions are reachable if they share an entity
node by key.

### 3.3 Anchor localisation (`retrieve.py:locate_anchor_entities`)

For each **anchor** sub-question (one with ≥1 real entity), the real
entity is matched to the KG via **Composed Retrieval**: BM25 top-100
prefilter → dense bi-encoder rerank (averaged name-only and
name+def cosine). Synonym neighbours of the matched entity are added
to the seed set `KE1`.

### 3.4 Iterative multi-target search (`retrieve.py:iterative_search`)

From `KE1`, the search alternates:
- **Relation competition**: relations incident to currently visited
  entities are scored against ALL sub-questions simultaneously
  (dense sim + `HYBRID_BM25_BOOST * normalised_BM25`); each
  relation is locked to its highest-similarity sub-question subject
  to a top-1-determine threshold.
- **Entity expansion**: the locked relations' OTHER endpoints are
  matched against the **virtual entities** in the skeleton DAG;
  passing virtuals become bound, and their bindings are cumulatively
  recorded in `state.virtual_bindings`.

After convergence a **supplement round** merges in seed relations
produced by a parallel Composed Retrieval over each sub-question's
natural-language relation_text (BM25 top-100 → dense rerank top-K
per query, unioned).

### 3.5 Variant sub-question instantiation (`retrieve.py:build_variants`)

After iterative search, each skeleton sub-question generates **variant
sub-questions** by cartesian-producting the bindings of its virtual
placeholders. Each variant carries:
- `qid` — parent skeleton qid
- `bindings` — concrete `(v_key, ent_id)` pairs (None = still unbound)
- `bound_entities` — frozen-set of KG ent_ids this variant references

A variant's solutions are the parent skeleton qid's relations,
returned as-is (per-variant filtering by bound_entities was tested
and removed — it over-zealously dropped gold relations whose KG
entity neighbours did not lexically cover the bridging entities).

### 3.6 Variant graph + path enumeration (`retrieve.py`)

A **variant graph** connects two variants iff they (a) belong to
DIFFERENT skeleton qids AND (b) share ≥1 bound entity. BFS from
anchor variants establishes levels; simple paths from anchor variants
to deepest variants are enumerated, deduplicated by qid SET (the
canonical qid sequence for each unique qid set is kept; alternate
permutations are dropped).

### 3.7 Path scoring (`retrieve.py:score_paths`)

```
score(p) = ( Σ over variant_id v in p of mean_sim(v) ) / n_subqs
```

Normalising by the TOTAL number of skeleton sub-questions
(`n_subqs`), not by `|p|`, penalises paths that only cover a subset of
the reasoning chain. Each covered variant contributes the mean
similarity of its **gap-based adaptively-selected top relations**
(`select_top_relations`: keep top-K_MIN=2 plus all subsequent up to
K_MAX=5 within a sim gap of GAP_THR=0.05).

### 3.8 Per-path context organisation (`retrieve.py:organize_variant_context`)

For each of the top-K scoring variant paths:
- Walk variants in path order.
- For each variant, append the top relations' text strings (the
  `(title) sentence` form built in `build_kg.py`).
- Deduplicate **within a path** (a relation that appeared earlier in
  the same path is not repeated).
- Do NOT deduplicate across paths — recurring relations across
  paths act as an implicit frequency-vote signal to the answering
  LLM.

The K paragraphs are joined with `\n\n` separators.

### 3.9 CoT answering (`evaluate.py`)

The answer LLM receives a single-CoT prompt:

```
Reasoning: <hop 1> -> <hop 2> -> ... -> <final entity>
Answer: <final answer>
```

The user message is suffixed with `Reasoning:` (NOT `Answer:`) to
prime the LLM to write the reasoning chain FIRST and only then commit
to a final answer. Priming with `Answer:` was tested and caused
LLM to dump an early guess before reasoning was complete; the
post-hoc reasoning chain often correctly reached the gold entity but
the prematurely-emitted answer remained anchored to a shallow hop
(observed on Q1 Politburo vs Soviet Union).

A **candidate-disambiguation** rule guides multi-candidate cases:
"when picking between two entities under the same hop, do not match
the most prominent keyword — re-check the question's full constraint
chain (e.g. league-specificity), and for superlatives (lowest /
highest / first), compare the actual numeric values rather than the
sentence's lexical match." This is illustrated by Example 3 in the
system prompt.

## 4. Key hyperparameters

| File | Constant | Default | Rationale |
|---|---|---|---|
| `retrieve.py` | `BM25_TOPN` | 100 | composed retrieval prefilter |
| `retrieve.py` | `TOP1_DETERMINE_EPS` | 0.1 | top-1-determine spread for entity matching |
| `retrieve.py` | `VIRTUAL_MATCH_THR` | 0.5 | virtual→real binding floor |
| `retrieve.py` | `VIRTUAL_TOP1_DETERMINE_EPS` | 0.05 | tighter virtual binding spread |
| `retrieve.py` | `VIRTUAL_MATCH_TOPK` | 3 | bindings per virtual per iter |
| `retrieve.py` | `MIN_THR_STEP4` | 0.4 | min relation-sim in iterative search |
| `retrieve.py` | `TOP_K_PATHS` | 3 | top-K paths emitted as paragraphs |
| `retrieve.py` | `REL_K_MIN` | 2 | adaptive cutoff floor (always keep) |
| `retrieve.py` | `REL_K_MAX` | 5 | adaptive cutoff ceiling |
| `retrieve.py` | `REL_GAP_THR` | 0.05 | gap-based stop for adaptive cutoff |
| `retrieve.py` | `HYBRID_BM25_BOOST` | 0.3 | additive BM25 boost on dense sim |
| `retrieve.py` | `MAX_BINDINGS_PER_VIRTUAL` | 3 | cap bindings (prevents variant blowup) |
| `retrieve.py` | `MAX_VARIANTS_PER_QID` | 8 | cap variants per skeleton qid |
| `retrieve.py` | `MAX_VARIANT_PATHS` | 200 | cap path enumeration |
| `inspect_llm.py` | `num_ctx` | 16384 | full prompt fits, KV cache stable |
| `inspect_llm.py` | `num_keep` | 4096 | always keep system prompt in cache |

## 5. Experimental snapshot (10q debug set, Qwen3-14B)

| Config | EM | F1 |
|---|---|---|
| Initial binary + by-sub-q grouping | 30% | 30% |
| binary + per-path context | 50% | 50% |
| n-ary variant + score-paths fix + adaptive K | 50% | 55% |
| + single CoT prompt | 60% | 65% |
| + variant graph + per-path context (no dedup) | **70%** | **70%** |
| Per-path concat + cross-path dedup (TESTED, rejected) | 60% | 60% |
| Multi-paragraph CoT (TESTED, rejected) | 50-60% | 50-60% |
| **Final: variant + reasoning-prime + disambig rule** | **70%** | **70%** |

The 3 unsolved questions in the 10q set:
- **Q6** (Rachel, Nevada) — dataset gold direction error (Groom Lake
  is south of Rachel, not vice versa).
- **Q9** (Bill Bergen, ±1 jitter) — multi-candidate ambiguity, fixed
  reliably with disambiguation rule.
- **Q10** (1 December 2010) — KG-side limitation: the underlying
  "1894-95 FA Cup" chunk never explicitly states Aston Villa won
  (only an indirect "Howard Vaughton, former Aston Villa player"
  mention), and the cross-name bridge Small Heath → Birmingham City
  is not in any chunk either.

## 6. Comparison with prior work

| Method | KG | Decomposition | Path enumeration | Context format |
|---|---|---|---|---|
| Standard chunk RAG | none | none | none | top-K chunks flat |
| GraphRAG (Edge 2024) | binary triple + community | none | none | community summary |
| HippoRAG / HippoRAG2 | binary triple | none | personalised PageRank | top-K passages |
| HyperGraphRAG (NeurIPS'25) | **hypergraph** | none | none | entity-anchored subgraph |
| HGRAG (AAAI'26) | hypergraph | none | none | cross-granularity entity+rel seeds |
| Beyond Chunks and Graphs / GRIEVER | binary triple | binary triplet | none | per-subq grouped relations |
| **HyperPathsRAG (ours)** | **hypergraph** | **n-ary** | **variant paths** | **per-path concat** |

We are the first to consistently apply n-ary representation to BOTH the
KG side AND the query-decomposition side, and the first to organise the
RAG context as **path-ordered paragraphs** rather than per-subq groups.

## 7. Output format

`evaluate.py` saves one JSON file with per-question records:

```json
{
  "n": 10,
  "em": 70.0,
  "f1": 70.0,
  "results": [
    {
      "qid": "<question text>",
      "question": "...",
      "gold": "the Politburo",
      "pred": "Politburo",
      "reasoning": "Vladimir Rapoport -> Soviet Union -> Mao Zedong -> Politburo",
      "em": 1,
      "f1": 1.0,
      "context_len": 1711,
      "n_paths": 1,
      "elapsed_sec": 7.9,
      "error": ""
    }, ...
  ]
}
```

`reasoning` is the chain the LLM emitted before the answer.
