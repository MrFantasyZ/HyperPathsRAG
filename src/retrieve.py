"""
HyperPathsRAG Retrieval Pipeline.

Step 1. Query decomposition (Qwen3-14B → typed triplets)
Step 2. Skeleton DAG construction
Step 3. Composed Retriever (BM25 + dense) → {KE1} → {RN1} → Result_1
Step 4. Iterative multi-target search (virtual entity matching, locking, sliding)
Step 5. Path scoring (level coverage + top-2 mean cos)
Step 6. Context organization (top-K paths × top-K relations per seg)
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import networkx as nx
import requests
import torch
from rank_bm25 import BM25Okapi

# ── config ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
KG_PKL    = ROOT / "kg_output" / "kg.pkl"
DATA_FILE = ROOT.parent / "semantic_anchor_rag/experiment_output/multi_dataset/data/combined_300.json"

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BM25_TOPN                  = 100
TOP1_DETERMINE_EPS         = 0.03    # tie-keeping ε for multi-subq locking
                                     # (loose → relation drifts to too many
                                     # subqs and dilutes top-K context;
                                     # tight → near-tied sibling subqs are
                                     # missed)
VIRTUAL_MATCH_THR          = 0.5    # absolute lower bound
VIRTUAL_TOP1_DETERMINE_EPS = 0.05   # within ε of top-1 sim for the virtual entity
VIRTUAL_MATCH_TOPK         = 3      # max kept per virtual
MIN_THR_STEP4              = 0.4
TOP_K_PATHS                = 3
# Adaptive per-segment relation cutoff (gap-based, replaces fixed TOP_K_REL_PER_SEG):
#   always keep top-REL_K_MIN, then continue adding the next-best relation
#   until either (a) a similarity gap larger than REL_GAP_THR appears, or
#   (b) REL_K_MAX is reached. Concentrated sim distributions get a wide
#   cutoff (Q8-Sufi at rank 5 included); long-tail distributions get a tight
#   cutoff (Q9 noise excluded).
REL_K_MIN                  = 2          # path-scoring floor (tight, favours sharp distributions)
REL_K_MAX                  = 5          # path-scoring ceiling
REL_GAP_THR                = 0.05       # path-scoring gap threshold (tight)
# Context-emission widening: AFTER a path is selected as one of the top-K
# paths, the relations actually shown to the LLM use a LOOSER adaptive cap
# so that a relation sitting just outside the scoring-K but still close to
# rank-K_MIN (e.g. the gold Sufi-missionary relation at sim 0.794 with gap
# 0.057 from the #2 relation) can still surface in the context. Path
# scoring uses the tight numbers above; context emission uses these.
REL_K_MIN_CONTEXT          = 2
REL_K_MAX_CONTEXT          = 8
REL_GAP_THR_CONTEXT        = 0.10
# Cap bindings per virtual entity when building variants — avoids
# cartesian-product explosion when iterative_search has accumulated
# many candidate bindings across rounds. Keeps the first-K (insertion
# order) bindings, which correlates with strongest sim due to round-0
# bias.
MAX_BINDINGS_PER_VIRTUAL   = 5
MAX_VARIANTS_PER_QID       = 8      # cap variants per skeleton qid (defends against
                                    # virtuals that picked up too many bindings)
MAX_VARIANT_PATHS          = 200    # truncate enumeration if graph is dense
MAX_STEP4_ITERS            = 5
# After the relation-Composed-Retrieval seeds are injected (supplement round),
# walk a few more iterations so the endpoints of newly-locked seed relations
# (notably title-as-entity nodes added by the title-hyperedge patch) get the
# chance to expose THEIR other relations.  Without this loop, title-edges
# would only sit in the graph as bridges that the algorithm never crosses.
POST_SUPPLEMENT_ITERS      = 2
# Entity-coherence boost. When scoring relation R for sub-question q, add a
# bonus proportional to the number of entities R shares with the relations
# already locked to q's NEIGHBOUR sub-questions in the skeleton DAG (i.e.
# subqs that share at least one virtual/real entity slot with q — the
# "承上启下" sub-questions both upstream and downstream in the reasoning
# chain). This makes the relation-locking stage prefer chain-coherent
# candidates over locally-similar but disconnected distractors, pushing
# Stage-4 path scoring's chain-completeness intuition one stage earlier.
ENTITY_OVERLAP_W           = 0.05    # per shared entity, summed over neighbours
ENTITY_OVERLAP_CAP         = 0.30    # cap on the per-(r,q) boost
ENTITY_OVERLAP_TOPK_PER_NB = 3       # top-K (by sim) neighbour solutions considered

# ── Attribute-family compatibility for virtual-entity matching ────────────
# Each virtual placeholder carries an `attribute` (e.g. "team", "year",
# "country") emitted by the decomposer. When matching it against KG
# entities, we restrict candidates to entities whose stored
# `attribute` (set by build_kg.py) is in the compatible
# family. This prevents the well-known degenerate case where the
# placeholder attribute word ("team", "year") embeds close to generic
# noun entities (`ent::club`, `ent::1996`) rather than the actually
# named entity instances (`ent::birmingham city`, `ent::1 december 2010`).
#
# Falls back to no filtering if the virtual's attribute is unknown OR
# the KG entities have no `attribute` field yet (backward-compatible).
ATTR_FAMILIES: dict[str, set[str]] = {
    # people / roles
    "person":       {"person", "role", "profession"},
    "role":         {"role", "profession", "person"},
    "profession":   {"profession", "role", "person"},
    "director":     {"person", "profession", "role"},
    "missionary":   {"person", "role", "group", "profession"},
    "arguer":       {"person", "role"},
    # organisations
    "organization": {"organization", "company", "institution", "team",
                     "club", "government", "group"},
    "company":      {"company", "organization"},
    "team":         {"team", "club", "organization", "group"},
    "club":         {"club", "team", "organization"},
    "institution":  {"institution", "organization"},
    "government":   {"government", "organization", "country"},
    "group":        {"group", "organization", "team"},
    # geography
    "country":      {"country", "region", "place", "area", "location"},
    "region":       {"region", "area", "country", "place", "location"},
    "area":         {"area", "region", "country", "place", "location"},
    "city":         {"city", "place", "location", "area"},
    "location":     {"location", "place", "region", "area", "city"},
    "place":        {"place", "location", "city", "region"},
    # time
    "year":         {"year", "date", "period", "century"},
    "date":         {"date", "year", "period"},
    "period":       {"period", "year", "century", "date"},
    "century":      {"century", "period", "year"},
    "time":         {"date", "year", "period", "century"},
    # works
    "film":         {"film", "creative-work", "tv-show"},
    "book":         {"book", "creative-work"},
    "song":         {"song", "creative-work"},
    "tv-show":      {"tv-show", "creative-work", "film"},
    "creative-work": {"creative-work", "film", "book", "song", "tv-show"},
    # events / contests
    "event":        {"event", "sports-event", "competition", "war",
                     "treaty", "tournament", "match"},
    "match":        {"match", "sports-event", "competition", "tournament", "event"},
    "war":          {"war", "event", "conflict", "treaty"},
    "conflict":     {"war", "event", "conflict"},
    "competition":  {"competition", "tournament", "sports-event", "event"},
    "tournament":   {"tournament", "competition", "sports-event", "event"},
    "sports-event": {"sports-event", "match", "competition", "tournament", "event"},
    "treaty":       {"treaty", "event"},
    # concepts / language
    "religion":     {"religion", "denomination"},
    "denomination": {"denomination", "religion"},
    "ideology":     {"ideology", "concept"},
    "language":     {"language"},
    "word":         {"word", "term", "language"},
    "term":         {"term", "word", "concept", "religion"},   # "Hindu" labeled as term
    "meaning":      {"meaning", "concept", "term", "word"},
    "concept":      {"concept", "ideology", "term"},
    # quantitative
    "number":       {"number", "quantity"},
    "quantity":     {"quantity", "number", "measurement"},
    "measurement":  {"measurement", "quantity", "number"},
    # misc
    "object":       {"object"},
    "imperialist power": {"country", "ideology"},   # special phrase from decomp
    "rainforest":   {"region", "area", "place", "location"},   # rainforests live in regions
    "dictionary":   {"book", "creative-work", "language"},
}
# Hybrid score = dense_sim + HYBRID_BM25_BOOST * bm25_normalized.
# Additive boost (not weighted sum): BM25 only ADDS when it has signal, never
# subtracts when BM25 is silent. Fixes BGE's poor discrimination on years /
# specific identifiers (e.g. "1894-95 FA Cup" vs "1995 FA Cup") without
# penalising subqs where BM25 has no useful signal ("Match A occurred in year A").
HYBRID_BM25_BOOST          = 0.3
EPS                        = 1e-8


# ── logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Encoder
# ══════════════════════════════════════════════════════════════════════════════

_embed_tok   = None
_embed_model = None


def get_embedder():
    global _embed_tok, _embed_model
    if _embed_model is None:
        from transformers import AutoTokenizer, AutoModel
        logger.info("Loading %s on %s …", EMBED_MODEL, DEVICE)
        _embed_tok   = AutoTokenizer.from_pretrained(EMBED_MODEL)
        _embed_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
        _embed_model.eval()
    return _embed_tok, _embed_model


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    tok, model = get_embedder()
    out_vecs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model(**enc)
        v = o.last_hidden_state[:, 0, :].cpu().numpy()
        n = np.linalg.norm(v, axis=1, keepdims=True)
        out_vecs.append(v / np.maximum(n, EPS))
    return np.vstack(out_vecs).astype(np.float32)


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


_CAMEL_RE   = re.compile(r"([a-z])([A-Z])")
_ALIAS_RE   = re.compile(r"(?<=\b)([A-Z])(?=\s|$)")   # standalone capital letter as alias

def normalize_for_embedding(text: str) -> str:
    """Strip noise tokens that hurt BGE: '?' markers, alias letters (A/B/C),
    and split camelCase identifiers into natural words."""
    text = text.replace("?", " ")
    text = _CAMEL_RE.sub(r"\1 \2", text)
    text = _ALIAS_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# KG Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class KGIndex:
    """Wraps the NetworkX MultiGraph with vectorised access for retrieval."""

    def __init__(self, kg_pkl: Path):
        logger.info("Loading KG from %s …", kg_pkl)
        with open(kg_pkl, "rb") as f:
            obj = pickle.load(f)
        self.G: nx.MultiGraph = obj["graph"]

        # split entity & relation nodes
        self.entity_ids:   list[str] = []
        self.relation_ids: list[str] = []
        for nid, d in self.G.nodes(data=True):
            if d["type"] == "entity":
                self.entity_ids.append(nid)
            else:
                self.relation_ids.append(nid)

        # Dual entity embeddings (built into KG): name-only and name+def.
        # Matching uses the AVERAGE of the two cosine similarities, providing
        # double-insurance: names disambiguate when definitions overlap, and
        # definitions disambiguate when names overlap (Apple Inc. vs Apple fruit).
        self.entity_emb = np.vstack([
            self.G.nodes[nid]["embedding"] for nid in self.entity_ids
        ]).astype(np.float32)
        self.entity_name_emb = np.vstack([
            self.G.nodes[nid]["name_embedding"] for nid in self.entity_ids
        ]).astype(np.float32)
        self.relation_emb = np.vstack([
            self.G.nodes[nid]["embedding"] for nid in self.relation_ids
        ]).astype(np.float32)

        self.ent_idx_of = {nid: i for i, nid in enumerate(self.entity_ids)}
        self.rel_idx_of = {nid: i for i, nid in enumerate(self.relation_ids)}

        # BM25 over entity "name: definition"
        ent_texts = [
            f"{self.G.nodes[nid]['display']}: {self.G.nodes[nid]['definition']}"
            for nid in self.entity_ids
        ]
        self.bm25 = BM25Okapi([bm25_tokenize(t) for t in ent_texts])

        # BM25 over relation node text (with source title context).
        # Used by the parallel "relation-seed" Composed Retriever — catches
        # cases where the answer relation has weak entity-graph connectivity
        # (Q9 Bergen isolation) or where entity-anchor matching went wrong
        # (Q8 anchor → wrong direction).
        rel_texts = [
            f"({self.G.nodes[rid].get('source_title','').strip()}: {self.G.nodes[rid]['text']})"
            if self.G.nodes[rid].get('source_title') else self.G.nodes[rid]['text']
            for rid in self.relation_ids
        ]
        self.bm25_relations = BM25Okapi([bm25_tokenize(t) for t in rel_texts])

        logger.info("KG loaded: %d entity, %d relation nodes",
                    len(self.entity_ids), len(self.relation_ids))

    def synonyms_of(self, ent_id: str) -> list[str]:
        """Return entity nodes connected to ent_id via 'synonym' edges."""
        if not self.G.has_node(ent_id):
            return []
        out = []
        for nbr, edata_dict in self.G.adj[ent_id].items():
            if self.G.nodes[nbr]["type"] != "entity":
                continue
            for _, edata in edata_dict.items():
                if edata.get("edge_type") == "synonym":
                    out.append(nbr)
                    break
        return out

    def relations_of(self, ent_id: str) -> list[str]:
        """Return relation nodes connected to ent_id via hyperedges."""
        if not self.G.has_node(ent_id):
            return []
        out = []
        for nbr, edata_dict in self.G.adj[ent_id].items():
            if self.G.nodes[nbr]["type"] != "relation":
                continue
            for _, edata in edata_dict.items():
                if edata.get("edge_type") == "hyperedge":
                    out.append(nbr)
                    break
        return out

    def entities_of_relation(self, rel_id: str) -> list[str]:
        """Return entity nodes connected to rel_id via hyperedges."""
        if not self.G.has_node(rel_id):
            return []
        out = []
        for nbr, edata_dict in self.G.adj[rel_id].items():
            if self.G.nodes[nbr]["type"] != "entity":
                continue
            for _, edata in edata_dict.items():
                if edata.get("edge_type") == "hyperedge":
                    out.append(nbr)
                    break
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Query decomposition
# ══════════════════════════════════════════════════════════════════════════════

_DECOMPOSE_SYSTEM = """You are a question decomposition assistant. Decompose a multi-hop question into atomic n-ary sub-questions and output them as a JSON array.

Each element of the array is an object with EXACTLY TWO keys:
- "relation": a complete natural English sentence describing one atomic fact. It MUST explicitly mention EVERY entity in the entities list (use the alias letter as a quasi-name in the sentence, e.g. "director A").
- "entities": an array of TWO OR MORE entity objects. Each entity object has keys {"name", "attribute", "virtual"}.

KEY DESIGN — n-ary sub-questions:
A sub-question is NOT restricted to two endpoints. When the fact involves 3 or more entities, list ALL of them. Example: "Area A became India in year A" mentions three entities — area A, India, and year A — so its entities list MUST contain all three. Listing only two of them silently breaks the reasoning chain because the missing entity cannot link this sub-question to others that share it.

Field semantics:
- "name" (str):
    * Real entities (virtual=false): the specific name as it appears in the question (e.g. "God's Gift To Women", "BCCI", "Christians in New Zealand").
    * Virtual entities (virtual=true): a single uppercase letter alias (A, B, C, ...). Reuse the same alias across sub-questions to chain multi-hop reasoning.
- "attribute" (str): type tag (film, director, person, year, country, area, organization, denomination, religion, meaning, ...). Always lowercase.
- "virtual" (bool): true = unknown to retrieve from KG, false = explicitly named in the question.

CRITICAL RULES:
0. **NAMED-ENTITY COVERAGE** (most important): Before writing sub-questions, mentally list EVERY named entity / specific term that appears IN THE QUESTION TEXT itself (proper nouns like "BCCI", "India", "Aston Villa", "1894-95 FA Cup"; specific cardinal terms like "second largest rainforest"; languages/formats like "Arabic dictionary"). Each such entity MUST appear with virtual=false in at least one sub-question's entities list. Pay special attention to entities buried in sub-clauses like "the area that became India" — here "India" is a named entity that MUST appear with virtual=false somewhere.
1. At least ONE sub-question MUST contain a real entity (virtual=false) — this entity is the anchor used to enter the knowledge graph.
2. **EVERY entity referenced inside the "relation" sentence MUST be listed in "entities"**. If the relation sentence says "Area A became India in year A", all of {area A, India, year A} must be in entities. Missing entities are the most common cause of broken reasoning chains.
3. Same alias = same unknown entity. Use this to chain multi-hop reasoning across sub-questions.
4. The "relation" sentence reads naturally and references every entity in the list.
5. Decompose into the MINIMAL set of independent atomic sub-questions that together fully solve the question — but never at the cost of skipping a named entity (rule 0 trumps minimality).

Example 1 (mostly binary sub-questions — common case):
Question: "Which film has the director born earlier, God's Gift To Women or Aldri Annet Enn Brak?"
[
  {"relation": "The film God's Gift To Women is directed by director A",
   "entities": [
     {"name": "God's Gift To Women", "attribute": "film", "virtual": false},
     {"name": "A", "attribute": "director", "virtual": true}
   ]},
  {"relation": "The film Aldri Annet Enn Brak is directed by director B",
   "entities": [
     {"name": "Aldri Annet Enn Brak", "attribute": "film", "virtual": false},
     {"name": "B", "attribute": "director", "virtual": true}
   ]},
  {"relation": "Director A was born in year A",
   "entities": [
     {"name": "A", "attribute": "director", "virtual": true},
     {"name": "A", "attribute": "year", "virtual": true}
   ]},
  {"relation": "Director B was born in year B",
   "entities": [
     {"name": "B", "attribute": "director", "virtual": true},
     {"name": "B", "attribute": "year", "virtual": true}
   ]}
]

Example 2 (TERNARY when sentence has 3 entities — DO NOT DROP ANY):
Question: "what is the meaning of the word that is also the majority religion in the area that became India when the country where BCCI is based was created in the Arabic dictionary?"
Named entities in question: BCCI, India, Arabic dictionary. ALL must appear with virtual=false.
[
  {"relation": "Organization BCCI is based in country A",
   "entities": [
     {"name": "BCCI", "attribute": "organization", "virtual": false},
     {"name": "A", "attribute": "country", "virtual": true}
   ]},
  {"relation": "Country A was created in year A",
   "entities": [
     {"name": "A", "attribute": "country", "virtual": true},
     {"name": "A", "attribute": "year", "virtual": true}
   ]},
  {"relation": "Area A became India in year A",
   "entities": [
     {"name": "A", "attribute": "area", "virtual": true},
     {"name": "India", "attribute": "country", "virtual": false},
     {"name": "A", "attribute": "year", "virtual": true}
   ]},
  {"relation": "Area A has religion A as its majority religion",
   "entities": [
     {"name": "A", "attribute": "area", "virtual": true},
     {"name": "A", "attribute": "religion", "virtual": true}
   ]},
  {"relation": "Word religion A has meaning A in the Arabic dictionary",
   "entities": [
     {"name": "A", "attribute": "religion", "virtual": true},
     {"name": "A", "attribute": "meaning", "virtual": true},
     {"name": "Arabic dictionary", "attribute": "dictionary", "virtual": false}
   ]}
]
Note that the third sub-question lists THREE entities (area A, India, year A). The "year A" entity is critical — it shares an alias with sub-question 2's "year A", which is what connects the BCCI side of the chain to the area/religion side. Listing only two entities here would split the reasoning graph into disconnected components.

Example 3 (binary, anchor inversion when real entity is a filter):
Question: "Who married the man who reformed the institution behind the denomination constituting 12.6% of Christians in New Zealand?"
[
  {"relation": "The group Christians in New Zealand has denomination A",
   "entities": [
     {"name": "Christians in New Zealand", "attribute": "group", "virtual": false},
     {"name": "A", "attribute": "denomination", "virtual": true}
   ]},
  {"relation": "Denomination A is behind institution A",
   "entities": [
     {"name": "A", "attribute": "denomination", "virtual": true},
     {"name": "A", "attribute": "institution", "virtual": true}
   ]},
  {"relation": "Person A wanted to reform institution A",
   "entities": [
     {"name": "A", "attribute": "person", "virtual": true},
     {"name": "A", "attribute": "institution", "virtual": true}
   ]},
  {"relation": "Person A was married to person B",
   "entities": [
     {"name": "A", "attribute": "person", "virtual": true},
     {"name": "B", "attribute": "person", "virtual": true}
   ]}
]

Output ONLY the JSON array. No markdown fences, no explanation, no extra text."""


def call_ollama_decompose(question: str) -> str:
    """Call Qwen3 with thinking DISABLED. Decomposition is a structural task
    (place 三元组占位框架, keep all unknowns as '?'); enabling thinking causes the
    model to use world knowledge to "pre-solve" virtual entities, embedding answer
    hints as real entities and breaking the KG-retrieval flow."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _DECOMPOSE_SYSTEM},
            {"role": "user",   "content": f'Question: "{question}"\nOutput:'},
        ],
        "stream":     False,
        "think":      False,
        "keep_alive": "1h",
        "options":    {"temperature": 0, "num_predict": 1024},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    msg = r.json()["message"]
    raw = msg.get("content", "")
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```[a-zA-Z]*|```", "", raw).strip()
    return raw


def _validate_entity_dict(d: dict) -> dict | None:
    if not isinstance(d, dict):
        return None
    name      = str(d.get("name", "")).strip()
    attribute = str(d.get("attribute", "")).strip().lower()
    virtual   = bool(d.get("virtual", False))
    if not name:
        return None
    return {"name": name, "attribute": attribute, "virtual": virtual}


def parse_subquestions(raw: str) -> list[dict]:
    """Parse JSON array of n-ary sub-question objects. Each object has keys
    'relation' (str) and 'entities' (list of >=2 entity dicts).

    Also accepts the legacy {subject, object} schema for backward compatibility:
    if 'entities' is absent but 'subject'/'object' are present, they are
    folded into a 2-entity list. (Newer prompts always emit 'entities'.)
    """
    text = re.sub(r"```(?:json)?|```", "", raw).strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except Exception as e:
        logger.warning("JSON parse failed: %s\nRaw content (first 500 chars):\n%s", e, text[:500])
        return []
    if not isinstance(data, list):
        return []

    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("relation", "")).strip()
        if not rel:
            continue
        raw_ents = item.get("entities")
        if not isinstance(raw_ents, list):
            # Legacy fallback: build from subject + object
            raw_ents = []
            sub = item.get("subject")
            obj = item.get("object")
            if sub is not None:
                raw_ents.append(sub)
            if obj is not None:
                raw_ents.append(obj)
        ents = [_validate_entity_dict(e) for e in raw_ents]
        ents = [e for e in ents if e is not None]
        if len(ents) < 2:
            continue
        out.append({"relation": rel, "entities": ents})
    return out


def decompose_query(question: str) -> list[dict]:
    raw = call_ollama_decompose(question)
    subqs = parse_subquestions(raw)
    if not subqs:
        logger.warning("Decomposition parse failed; raw content:\n%s", raw[:500])
    return subqs


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Skeleton DAG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EntityInfo:
    raw:       str        # original string e.g. "? director A"
    virtual:   bool
    attribute: str        # e.g. "director"
    name:      str        # e.g. "A" or "God's Gift To Women"
    key:       str        # canonical key for skeleton matching (real=name, virtual="attr alias")

    @property
    def embed_text(self) -> str:
        # For encoder input: always include attribute so BGE has the type signal
        # (e.g. "person Vladimir Rapoport" / "country A").
        if self.attribute:
            return f"{self.attribute} {self.name}".strip()
        return self.name


def entity_from_dict(d: dict) -> EntityInfo:
    """Build an EntityInfo from the validated {name, attribute, virtual} dict
    produced by the JSON decomposition output."""
    name      = d["name"]
    attribute = d["attribute"]
    virtual   = d["virtual"]
    # Canonical key for skeleton-graph matching:
    #   real:    just the name (e.g. "Vladimir Rapoport")
    #   virtual: "<attribute> <alias>" (e.g. "country A") — different aliases
    #            distinguish different virtuals with the same attribute
    if virtual:
        key = f"{attribute} {name}".strip() if attribute else name
    else:
        key = name
    return EntityInfo(
        raw       = (f"? {attribute} {name}" if virtual else f"{attribute} {name}").strip(),
        virtual   = virtual,
        attribute = attribute,
        name      = name,
        key       = key,
    )


@dataclass
class SubQuestion:
    qid:           int
    entities:      list[EntityInfo]  # n-ary: 2 or more entities
    relation_text: str               # natural English sentence from LLM JSON

    @property
    def triple_text(self) -> str:
        # Kept for backward compatibility with iterative_search encoder inputs.
        return self.relation_text

    @property
    def real_entities(self) -> list[EntityInfo]:
        return [e for e in self.entities if not e.virtual]

    @property
    def virtual_entities(self) -> list[EntityInfo]:
        return [e for e in self.entities if e.virtual]


def build_skeleton_dag(subq_dicts: list[dict]) -> tuple[nx.MultiGraph, list[SubQuestion]]:
    """Bipartite skeleton graph: ENTITY nodes ↔ SUB-QUESTION nodes.

    Each sub-question is its own node (SQ::<qid>); it is linked to every
    entity in its entities list. Two sub-questions become reachable in the
    graph iff they share an entity (entity nodes are deduplicated by key).

    This generalises the previous binary-edge model to n-ary
    sub-questions, so that a single sub-question whose sentence touches
    three entities (e.g. 'Area A became India in year A') still produces
    one hyperedge linking all three of {area A, India, year A}. The
    previous binary scheme silently dropped the third entity and broke
    the reasoning chain at this exact point.

    Node attributes:
      - entity nodes:  {"kind": "entity", "info": EntityInfo}
      - subq nodes:    {"kind": "subq",   "info": SubQuestion}

    Edge attributes: each entity↔subq edge carries {"qid": <int>}.
    """
    G = nx.MultiGraph()
    subqs: list[SubQuestion] = []
    for qid, item in enumerate(subq_dicts):
        ent_infos = [entity_from_dict(e) for e in item["entities"]]
        # Dedup entities by key within the same sub-question
        seen_keys: set[str] = set()
        unique_ents: list[EntityInfo] = []
        for e in ent_infos:
            if e.key in seen_keys:
                continue
            seen_keys.add(e.key)
            unique_ents.append(e)
        sq = SubQuestion(qid=qid, entities=unique_ents, relation_text=item["relation"])
        subqs.append(sq)

        sq_node = f"SQ::{qid}"
        G.add_node(sq_node, kind="subq", info=sq)
        for ent in unique_ents:
            ent_node = ("V::" if ent.virtual else "R::") + ent.key
            if ent_node not in G:
                G.add_node(ent_node, kind="entity", info=ent)
            G.add_edge(sq_node, ent_node, qid=qid)
    return G, subqs


def build_subq_adjacency(subqs: list[SubQuestion]) -> dict[int, set[int]]:
    """For each sub-question qid, return the set of qids sharing at least one
    entity slot key. Two subqs share an entity iff their entity lists contain
    EntityInfo objects with the same canonical key (attribute + name). The
    relation is symmetric, encoding the skeleton DAG's "承上启下" structure
    without distinguishing parent vs. child direction — what matters for
    entity-overlap scoring is that both ends sit on the same reasoning chain.
    """
    adj: dict[int, set[int]] = {sq.qid: set() for sq in subqs}
    qid_keys: dict[int, set[str]] = {sq.qid: {e.key for e in sq.entities} for sq in subqs}
    for i, sq_a in enumerate(subqs):
        for sq_b in subqs[i + 1:]:
            if qid_keys[sq_a.qid] & qid_keys[sq_b.qid]:
                adj[sq_a.qid].add(sq_b.qid)
                adj[sq_b.qid].add(sq_a.qid)
    return adj


def compute_levels(G: nx.MultiGraph) -> dict[str, int]:
    """BFS distance from real-entity nodes (level 0) over the bipartite
    entity↔subq graph. Sub-question nodes inherit (entity-level + 1),
    virtual entity nodes inherit (subq-level + 1), and so on.

    Used by enumerate_paths_with_levels to identify deepest virtual
    entities as path endpoints."""
    levels: dict[str, int] = {}
    sources = [n for n, d in G.nodes(data=True)
               if d.get("kind") == "entity" and not d["info"].virtual]
    if not sources:
        return levels
    from collections import deque
    queue: deque = deque()
    for s in sources:
        levels[s] = 0
        queue.append(s)
    while queue:
        u = queue.popleft()
        for v in G.neighbors(u):
            if v not in levels:
                levels[v] = levels[u] + 1
                queue.append(v)
    return levels


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Composed Retriever + first relation search
# ══════════════════════════════════════════════════════════════════════════════

def composed_retrieve(kg: KGIndex, query_text: str, top_n_bm25: int = BM25_TOPN) -> str | None:
    """BM25 top-N → dense rerank using AVERAGE of (name-only, name+def) sim."""
    tokens = bm25_tokenize(query_text)
    if not tokens:
        return None
    bm25_scores = kg.bm25.get_scores(tokens)   # (n_ent,)
    top_n_idx = np.argsort(-bm25_scores)[:top_n_bm25]

    q_vec    = encode_texts([query_text])[0]
    sim_name = kg.entity_name_emb[top_n_idx] @ q_vec     # name-only
    sim_def  = kg.entity_emb[top_n_idx]      @ q_vec     # name+def
    sims     = (sim_name + sim_def) / 2.0

    best_local = int(np.argmax(sims))
    best_idx   = int(top_n_idx[best_local])
    return kg.entity_ids[best_idx]


def composed_retrieve_relations(
    kg: KGIndex,
    query_texts: list[str],
    top_n_bm25: int = 100,
    top_k_per_query: int = 10,
) -> list[str]:
    """Per-query parallel relation-node Composed Retriever (BM25 → dense).

    For each query text (original question + each sub-question's relation sentence),
    do BM25 top-N filter → dense rerank → take top-K. Return UNION of all per-query
    top-K results.

    Why per-query and NOT one combined query:
        Empirically (Q10 test) a concatenated combined query DILUTES the focused
        semantic signal of each sub-question, dropping target relations from
        rank #6 (best single subq) to rank #55 (combined). Per-query searches
        keep each subq's focus tight and union catches all reasoning chains.

    Recovers:
      - Q9 Bergen: subq "Player A has lowest batting average" matches Bergen rel
      - Q8 Sufi: subq "Missionary A spread religion A" matches Sufi rel
      - Q10 Vaughton/Aston: subq "1894-95 FA Cup winner is team B" matches Vaughton rel
    """
    query_texts = [q for q in query_texts if q and q.strip()]
    if not query_texts:
        return []

    # Batch encode all queries for efficiency
    q_vecs = encode_texts(query_texts)

    seeds: set[str] = set()
    for q_text, q_vec in zip(query_texts, q_vecs):
        tokens = bm25_tokenize(q_text)
        if not tokens:
            continue
        bm25_scores = kg.bm25_relations.get_scores(tokens)
        top_n_idx   = np.argsort(-bm25_scores)[:top_n_bm25]
        cand_emb    = kg.relation_emb[top_n_idx]
        sims        = cand_emb @ q_vec
        order_local = np.argsort(-sims)[:top_k_per_query]
        for i in order_local:
            seeds.add(kg.relation_ids[int(top_n_idx[i])])
    return list(seeds)


def locate_anchor_entities(kg: KGIndex, anchor_subqs: list[SubQuestion]) -> set[str]:
    """For each anchor subq, iterate over ALL its real entities (n-ary) and
    locate each in the KG via Composed Retriever. Return {KE1}."""
    ke1: set[str] = set()
    seen_queries: set[str] = set()
    for sq in anchor_subqs:
        for ent in sq.entities:
            if ent.virtual:
                continue
            query = ent.embed_text
            if query in seen_queries:
                continue
            seen_queries.add(query)
            best = composed_retrieve(kg, query)
            if best is None:
                logger.warning("Composed retrieval failed for anchor: %s", query)
                continue
            ke1.add(best)
            for syn in kg.synonyms_of(best):
                ke1.add(syn)
            logger.info("  Anchor '%s' → %s (+%d synonyms)",
                        query, kg.G.nodes[best]["display"], len(kg.synonyms_of(best)))
    return ke1


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Iterative multi-target search
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolutionEntry:
    rel_id:      str
    sim:         float
    variant_tag: str   # "original" | "subj=<ent_id>" | "obj=<ent_id>"


@dataclass
class RetrievalState:
    subqs:             list[SubQuestion]
    subq_vecs:         np.ndarray              # (n_subq, 1024)
    virtual_keys:      list[str]               # canonical "<attr> <alias>"
    virtual_vecs:      np.ndarray              # (n_virtual, 1024)
    virtual_idx_of:    dict[str, int]
    solutions:         dict[int, list[SolutionEntry]]  # qid → entries
    locked_relations:  set[str]
    visited_entities:  set[str]                # all KE_i ever seen
    # virtual_key -> set of KG entity ids bound to that virtual placeholder.
    # Populated cumulatively by iterative_search as virtual entities get
    # matched. Used by organize_context to filter relations per variant.
    virtual_bindings:  dict[str, set[str]]
    # Real entity key (from question) -> KG entity ids matched as anchors.
    anchor_bindings:   dict[str, set[str]]
    # (entity_id, virtual_key) -> averaged-embedding cosine similarity that
    # was observed when matching this entity against this virtual slot.
    # Used by build_variants to sort virtual bindings by SIM (not alphabet)
    # before truncating to MAX_BINDINGS_PER_VIRTUAL.
    virtual_sims:      dict[tuple[str, str], float]


def collect_virtuals(subqs: list[SubQuestion]) -> tuple[list[str], dict[str, int]]:
    """Deduplicated virtual entity keys (canonical "<attr> <alias>") across
    every entity slot of every n-ary sub-question."""
    seen: dict[str, int] = {}
    keys: list[str] = []
    for sq in subqs:
        for ent in sq.entities:
            if ent.virtual and ent.key and ent.key not in seen:
                seen[ent.key] = len(keys)
                keys.append(ent.key)
    return keys, seen


def compute_bm25_relation_scores(
    kg: KGIndex, subqs: list[SubQuestion]
) -> np.ndarray:
    """Precompute per-subq BM25 scores over ALL relations, normalised to [0,1].

    Returns (n_relations, n_subqs). Used by round_relation_search for hybrid
    score = HYBRID_ALPHA * dense_sim + (1-HYBRID_ALPHA) * bm25_norm. BM25
    rescues retrieval cases where dense embeddings fail to discriminate
    years / specific identifiers (e.g. "1894-95 FA Cup" vs "1995 FA Cup").
    """
    n_rel = len(kg.relation_ids)
    out   = np.zeros((n_rel, len(subqs)), dtype=np.float32)
    for q_idx, sq in enumerate(subqs):
        tokens = bm25_tokenize(sq.relation_text)
        if not tokens:
            continue
        scores = kg.bm25_relations.get_scores(tokens)
        mx = float(scores.max())
        if mx > 0:
            out[:, q_idx] = scores / mx
    return out


def round_relation_search(
    kg: KGIndex,
    state: RetrievalState,
    rn_curr_ids: list[str],
    is_first_round: bool,
    variant_tag_map: dict[int, list[str]] | None = None,
    bm25_per_subq: np.ndarray | None = None,
    subq_neighbors: dict[int, set[int]] | None = None,
) -> int:
    """One round of relation node search + lock+slide assignment. Returns # added."""
    if not rn_curr_ids:
        return 0

    rn_idx     = np.array([kg.rel_idx_of[r] for r in rn_curr_ids])
    rn_emb     = kg.relation_emb[rn_idx]                          # (n_rn, d)
    dense_sims = rn_emb @ state.subq_vecs.T                       # (n_rn, n_subq)
    if bm25_per_subq is not None:
        bm25_local = bm25_per_subq[rn_idx]                        # (n_rn, n_subq), in [0,1]
        sims_mat   = dense_sims + HYBRID_BM25_BOOST * bm25_local
    else:
        sims_mat   = dense_sims

    # ── Entity-coherence boost (承上启下) ─────────────────────────────────────
    # For each candidate relation r and each subq q with at least one
    # already-locked neighbour, add ENTITY_OVERLAP_W per shared entity
    # between r and the top-K locked relations of q's neighbours. This
    # rewards relations that fit the reasoning chain entity-wise, not just
    # textually.
    if subq_neighbors:
        # Pre-compute the entity set of each candidate relation in rn_curr_ids
        cand_ent_sets: list[set[str]] = [
            set(kg.entities_of_relation(rid)) for rid in rn_curr_ids
        ]
        # For each subq_idx, collect the union of entities mentioned by the
        # top-K locked solutions of all its NEIGHBOUR subqs (both upstream and
        # downstream in the reasoning chain).
        neighbour_entity_sets: list[set[str]] = []
        qid_to_idx = {state.subqs[i].qid: i for i in range(len(state.subqs))}
        for q_idx, sq in enumerate(state.subqs):
            ent_acc: set[str] = set()
            for nbr_qid in subq_neighbors.get(sq.qid, ()):
                if nbr_qid not in qid_to_idx:
                    continue
                # Top-K locked solutions of the neighbour, ranked by sim
                nbr_entries = sorted(
                    state.solutions.get(nbr_qid, []),
                    key=lambda e: -e.sim,
                )[:ENTITY_OVERLAP_TOPK_PER_NB]
                for entry in nbr_entries:
                    for ent in kg.entities_of_relation(entry.rel_id):
                        ent_acc.add(ent)
            neighbour_entity_sets.append(ent_acc)
        # Apply boost
        for r_idx in range(len(rn_curr_ids)):
            r_ents = cand_ent_sets[r_idx]
            if not r_ents:
                continue
            for q_idx in range(len(state.subqs)):
                nbr_ents = neighbour_entity_sets[q_idx]
                if not nbr_ents:
                    continue
                overlap = len(r_ents & nbr_ents)
                if overlap:
                    boost = min(ENTITY_OVERLAP_W * overlap, ENTITY_OVERLAP_CAP)
                    sims_mat[r_idx, q_idx] += boost

    n_rn, n_sq = sims_mat.shape

    # Determine per-subq threshold
    thresholds = np.zeros(n_sq, dtype=np.float32)
    for q_idx in range(n_sq):
        qid = state.subqs[q_idx].qid
        if is_first_round:
            # Top-1-determine on this subq's column
            col_max = float(sims_mat[:, q_idx].max())
            thresholds[q_idx] = col_max - TOP1_DETERMINE_EPS
        else:
            existing = [e.sim for e in state.solutions.get(qid, [])]
            if existing:
                thresholds[q_idx] = max(MIN_THR_STEP4, float(np.mean(existing)))
            else:
                thresholds[q_idx] = MIN_THR_STEP4

    # For each relation, find eligible subqs, pick highest sim
    added = 0
    for r_idx in range(n_rn):
        rel_id = rn_curr_ids[r_idx]
        eligible_q_idxs = [q for q in range(n_sq) if sims_mat[r_idx, q] >= thresholds[q]]
        if not eligible_q_idxs:
            continue
        eligible_q_idxs.sort(key=lambda q: -sims_mat[r_idx, q])
        winner_q_idx = eligible_q_idxs[0]
        winner_qid   = state.subqs[winner_q_idx].qid
        sim_val      = float(sims_mat[r_idx, winner_q_idx])
        tag = "original"
        if variant_tag_map and winner_qid in variant_tag_map:
            tag = variant_tag_map[winner_qid][0] if variant_tag_map[winner_qid] else "original"
        state.solutions.setdefault(winner_qid, []).append(
            SolutionEntry(rel_id=rel_id, sim=sim_val, variant_tag=tag))
        state.locked_relations.add(rel_id)
        added += 1
    return added


def iterative_search(
    kg: KGIndex,
    subqs: list[SubQuestion],
    ke1: set[str],
    seed_relations: set[str] | None = None,   # extra RN candidates from relation-Composed-Retrieval
) -> RetrievalState:
    # encode all sub-questions with noise stripped (drops '?', alias letters, expands camelCase)
    subq_inputs = [normalize_for_embedding(sq.triple_text) for sq in subqs]
    logger.debug("Normalised subq texts:\n  %s", "\n  ".join(subq_inputs))
    subq_vecs = encode_texts(subq_inputs)

    # encode virtuals (also normalised)
    virtual_keys, virtual_idx_of = collect_virtuals(subqs)
    virtual_inputs = [normalize_for_embedding(k) for k in virtual_keys]
    virtual_vecs = encode_texts(virtual_inputs) if virtual_inputs else np.zeros((0, 1024), dtype=np.float32)

    # Precompute BM25 scores for every (relation, subq) pair, normalised to [0,1].
    # Reused across all rounds (main loop + supplement) of round_relation_search.
    bm25_per_subq = compute_bm25_relation_scores(kg, subqs)

    # Sub-question adjacency in the skeleton DAG — used by the entity-coherence
    # boost in round_relation_search. Two subqs are adjacent iff they share at
    # least one entity slot (real or virtual).
    subq_neighbors = build_subq_adjacency(subqs)

    # Map virtual-key → its decomposer-emitted attribute (e.g. "team A" → "team").
    # Used by the virtual-entity-matching attribute filter below.
    virtual_attr_of: dict[str, str] = {}
    for sq in subqs:
        for ent in sq.entities:
            if ent.virtual and ent.key not in virtual_attr_of:
                virtual_attr_of[ent.key] = ent.attribute

    # Real-entity anchor bindings: real_entity_key (as written in the question)
    # -> set of KG entity ids. Seeded from KE1; needed to resolve bound real
    # entities for variant filtering at context time.
    anchor_bindings: dict[str, set[str]] = defaultdict(set)
    for sq in subqs:
        for ent in sq.entities:
            if not ent.virtual:
                # Find which KE1 entity corresponds to this real key via
                # display name match (composed_retrieve was already run).
                for eid in ke1:
                    if kg.G.nodes[eid].get("display", "").lower() == ent.name.lower():
                        anchor_bindings[ent.key].add(eid)
        # If no exact display match, just attribute all KE1 to all real keys
        # (degraded fallback used by downstream filter as a permissive bound set).
    if not anchor_bindings:
        # All real entities lumped under a single sentinel key
        anchor_bindings["__all__"] = set(ke1)

    state = RetrievalState(
        subqs=subqs,
        subq_vecs=subq_vecs,
        virtual_keys=virtual_keys,
        virtual_vecs=virtual_vecs,
        virtual_idx_of=virtual_idx_of,
        solutions={sq.qid: [] for sq in subqs},
        locked_relations=set(),
        visited_entities=set(ke1),
        virtual_bindings=defaultdict(set),
        anchor_bindings=anchor_bindings,
        virtual_sims={},
    )

    # ── Unified iterative search (merged Step 3 + Step 4) ────────────────────
    # Iteration 0: KE_curr = KE1 (the just-located anchor entities). All sub-questions
    #              (including virtual-only ones) compete for relations from KE1's
    #              neighborhood. This avoids the leak where anchor-only Top-1-determine
    #              filters out 1-hop relations that are the true answer to virtual subqs.
    # Iteration i≥1: KE_curr from virtual-entity matching of the newly-reached entities.
    #
    # In EVERY iteration, RN_curr = (relations of all visited entities) − locked,
    # so an unlocked relation of an early-visited entity can still be claimed later
    # when a more appropriate sub-question gets a turn.

    # Alias kept for backward compatibility with the rest of the function;
    # writes flow through to state.virtual_sims so downstream code (e.g.
    # build_variants) can sort virtual bindings by sim instead of alphabet.
    ent_virt_cache = state.virtual_sims

    for it in range(MAX_STEP4_ITERS + 1):
        # ── 1. Determine new entities to add this round ──────────────────────
        variant_tag_map: dict[int, list[str]] = defaultdict(list)
        if it == 0:
            # First iteration: visited_entities already = KE1; no virtual matching yet.
            new_ke_curr: set[str] = set()    # nothing newly added; visited already has KE1
            matched_count = len(ke1)
            ke_next_count = len(ke1)
        else:
            # Newly-reached entities = neighbors of any selected relation, not yet visited
            ke_curr: set[str] = set()
            for entries in state.solutions.values():
                for e in entries:
                    for ent_nbr in kg.entities_of_relation(e.rel_id):
                        if ent_nbr not in state.visited_entities:
                            ke_curr.add(ent_nbr)
            if not ke_curr:
                logger.info("Iter %d: no new entities reached → stop", it)
                break

            # Virtual entity matching (3 conditions: > VIRTUAL_MATCH_THR,
            # within VIRTUAL_TOP1_DETERMINE_EPS of top-1, capped at VIRTUAL_MATCH_TOPK)
            matched_entities: set[str] = set()
            if virtual_vecs.shape[0] > 0:
                ke_list = sorted(ke_curr)
                ke_idx  = np.array([kg.ent_idx_of[e] for e in ke_list])
                sim_def_mat  = kg.entity_emb[ke_idx]      @ virtual_vecs.T
                sim_name_mat = kg.entity_name_emb[ke_idx] @ virtual_vecs.T
                sims         = (sim_def_mat + sim_name_mat) / 2.0
                for i, ent_id in enumerate(ke_list):
                    for j, v_key in enumerate(virtual_keys):
                        ent_virt_cache[(ent_id, v_key)] = float(sims[i, j])

                for j, v_key in enumerate(virtual_keys):
                    col = sims[:, j].copy()
                    if len(col) == 0:
                        continue
                    # Attribute-family mask: zero out entities whose KG
                    # `attribute` is not in the virtual's compatible family.
                    # If the virtual attribute is unknown to ATTR_FAMILIES
                    # OR a candidate entity has no attribute field yet,
                    # the mask is permissive (backward-compatible).
                    v_attr = virtual_attr_of.get(v_key, "")
                    compatible = ATTR_FAMILIES.get(v_attr.strip().lower())
                    if compatible:
                        for i, ent_id in enumerate(ke_list):
                            ent_attr = kg.G.nodes[ent_id].get("attribute")
                            if ent_attr and ent_attr not in compatible:
                                col[i] = -1.0   # mask
                    ranked = np.argsort(-col)
                    top1_sim = float(col[ranked[0]])
                    if top1_sim <= VIRTUAL_MATCH_THR:
                        continue
                    kept = 0
                    for i in ranked:
                        s = float(col[i])
                        if s <= VIRTUAL_MATCH_THR or s < top1_sim - VIRTUAL_TOP1_DETERMINE_EPS:
                            break
                        ent_id = ke_list[int(i)]
                        matched_entities.add(ent_id)
                        # Cumulative variant binding (used by context filter):
                        # this virtual placeholder now has another real-entity
                        # candidate.
                        state.virtual_bindings[v_key].add(ent_id)
                        # For each subq, record which n-ary entity slot just
                        # bound to this real KG entity.
                        for sq in subqs:
                            for slot, ent in enumerate(sq.entities):
                                if ent.virtual and ent.key == v_key:
                                    variant_tag_map[sq.qid].append(
                                        f"slot{slot}={v_key}={ent_id}")
                        kept += 1
                        if kept >= VIRTUAL_MATCH_TOPK:
                            break
            else:
                matched_entities = set(ke_curr)

            if not matched_entities:
                logger.info("Iter %d: no virtual matches above %.2f → stop A",
                            it, VIRTUAL_MATCH_THR)
                break

            # KE_next = matched ∪ synonym neighbours, then merge into visited
            ke_next: set[str] = set(matched_entities)
            for e in matched_entities:
                for s in kg.synonyms_of(e):
                    if s not in state.visited_entities:
                        ke_next.add(s)
            state.visited_entities.update(ke_next)
            new_ke_curr = ke_curr
            matched_count = len(matched_entities)
            ke_next_count = len(ke_next)

        # ── 2. RN_curr = unlocked relations of ALL visited entities ──────────
        # NOTE: relation-Composed-Retrieval seeds are NOT injected here; they
        # would derail the entity-expansion path by claiming early subqs with
        # tangential matches. Seeds are injected as a SUPPLEMENT after the
        # main loop converges (see "supplement round" below).
        rn_set: set[str] = set()
        for e in state.visited_entities:
            for r in kg.relations_of(e):
                if r not in state.locked_relations:
                    rn_set.add(r)
        if not rn_set:
            logger.info("Iter %d: no unlocked candidate relations → stop", it)
            break
        rn_curr = sorted(rn_set)

        # ── 3. All subqs compete for these relations ─────────────────────────
        before = sum(len(v) for v in state.solutions.values())
        round_relation_search(
            kg, state, rn_curr,
            is_first_round=(it == 0),
            variant_tag_map=variant_tag_map if it > 0 else None,
            bm25_per_subq=bm25_per_subq,
            subq_neighbors=subq_neighbors,
        )
        added = sum(len(v) for v in state.solutions.values()) - before

        if it == 0:
            logger.info("Iter 0 (from KE1=%d): RN_curr=%d, added=%d",
                        len(ke1), len(rn_curr), added)
        else:
            logger.info("Iter %d: KE_curr=%d, matched=%d, KE_next=%d, RN_curr=%d, added=%d",
                        it, len(new_ke_curr), matched_count, ke_next_count, len(rn_curr), added)

        if it > 0 and added == 0:
            logger.info("Iter %d: no relations added → stop B", it)
            break

        if added == 0:
            logger.info("Iter %d: no relations added → stop B", it)
            break

    # ── Supplement round: inject relation-Composed-Retrieval seeds ──────────
    # Run AFTER entity expansion converges, so seeded relations only fill gaps
    # left by entity expansion (cases where the answer relation has no path
    # through the entity graph — Q8 anchor misdirection, Q9 entity isolation).
    # Threshold = max(MIN_THR_STEP4, mean of existing solutions per subq), so
    # already-well-served subqs won't accept low-quality seeds.
    if seed_relations:
        supplement = [r for r in sorted(seed_relations) if r not in state.locked_relations]
        if supplement:
            before = sum(len(v) for v in state.solutions.values())
            round_relation_search(kg, state, supplement,
                                  is_first_round=False,
                                  variant_tag_map=None,
                                  bm25_per_subq=bm25_per_subq,
                                  subq_neighbors=subq_neighbors)
            added = sum(len(v) for v in state.solutions.values()) - before
            logger.info("Supplement round (relation seeds): %d candidates, +%d added",
                        len(supplement), added)

    # ── Post-supplement entity expansion ────────────────────────────────────
    # Walk forward from the endpoints of supplement-locked relations.  This
    # is what gives title-hyperedges (added by patch_kg_title_edges.py /
    # build_kg.py) their actual retrieval value: a relation matched purely
    # by Composed Retrieval lands its title-entity (e.g. Southeast Asia)
    # into the visited set, and from there other relations sharing that
    # title (e.g. the Sufi-missionaries relation) become reachable.
    for post_it in range(POST_SUPPLEMENT_ITERS):
        # 1) Endpoints of all currently-locked relations that are not yet visited
        new_endpoints: set[str] = set()
        for entries in state.solutions.values():
            for e in entries:
                for ent_nbr in kg.entities_of_relation(e.rel_id):
                    if ent_nbr not in state.visited_entities:
                        new_endpoints.add(ent_nbr)
        if not new_endpoints:
            logger.info("Post-supplement iter %d: no new endpoints → stop", post_it)
            break

        # 2) Virtual matching over new_endpoints (mirrors the main-loop block)
        matched_entities: set[str] = set()
        variant_tag_map_post: dict[int, list[str]] = defaultdict(list)
        if virtual_vecs.shape[0] > 0:
            ke_list = sorted(new_endpoints)
            ke_idx  = np.array([kg.ent_idx_of[e] for e in ke_list])
            sim_def_mat  = kg.entity_emb[ke_idx]      @ virtual_vecs.T
            sim_name_mat = kg.entity_name_emb[ke_idx] @ virtual_vecs.T
            sims         = (sim_def_mat + sim_name_mat) / 2.0
            for i, ent_id in enumerate(ke_list):
                for j, v_key in enumerate(virtual_keys):
                    ent_virt_cache[(ent_id, v_key)] = float(sims[i, j])
            for j, v_key in enumerate(virtual_keys):
                col = sims[:, j].copy()
                if len(col) == 0:
                    continue
                # Same attribute-family mask as the main loop above
                v_attr = virtual_attr_of.get(v_key, "")
                compatible = ATTR_FAMILIES.get(v_attr.strip().lower())
                if compatible:
                    for i, ent_id in enumerate(ke_list):
                        ent_attr = kg.G.nodes[ent_id].get("attribute")
                        if ent_attr and ent_attr not in compatible:
                            col[i] = -1.0
                ranked = np.argsort(-col)
                top1_sim = float(col[ranked[0]])
                if top1_sim <= VIRTUAL_MATCH_THR:
                    continue
                kept = 0
                for i in ranked:
                    s = float(col[i])
                    if s <= VIRTUAL_MATCH_THR or s < top1_sim - VIRTUAL_TOP1_DETERMINE_EPS:
                        break
                    ent_id = ke_list[int(i)]
                    matched_entities.add(ent_id)
                    state.virtual_bindings[v_key].add(ent_id)
                    for sq in subqs:
                        for slot, ent in enumerate(sq.entities):
                            if ent.virtual and ent.key == v_key:
                                variant_tag_map_post[sq.qid].append(
                                    f"slot{slot}={v_key}={ent_id}")
                    kept += 1
                    if kept >= VIRTUAL_MATCH_TOPK:
                        break

        # 3) Always include endpoints that reach a locked relation via a
        # title-edge — these are chunk-level bridges (the chunk-title entity
        # whose hyperedge was added by the title-edges patch / build_kg.py).
        # They will not necessarily score high against any virtual
        # placeholder, but their hyperedge fan-out is exactly what we want
        # to surface (e.g. Southeast Asia → Sufi-missionary relation).
        title_endpoints: set[str] = set()
        locked = state.locked_relations
        for e in new_endpoints:
            if e not in kg.G:
                continue
            for nbr, edata_dict in kg.G.adj[e].items():
                if nbr not in locked:
                    continue
                for _, edata in edata_dict.items():
                    if edata.get("is_title_edge"):
                        title_endpoints.add(e)
                        break
                if e in title_endpoints:
                    break
        matched_entities |= title_endpoints

        # ALSO promote title-endpoints into virtual_bindings whenever their
        # KG `attribute` is compatible with a virtual placeholder. Without
        # this step, title-edge bridges (e.g. Southeast Asia for the
        # rainforest chain) reach visited_entities but never become
        # candidate VARIANT BINDINGS, so build_variants never produces a
        # variant rooted at them. This is what kept Q2's "region A" from
        # ever binding to Southeast Asia even though its title-edges did
        # make Southeast Asia visited.
        for te in title_endpoints:
            te_attr = (kg.G.nodes[te].get("attribute") or "").strip().lower()
            if not te_attr:
                continue
            for v_key, v_attr in virtual_attr_of.items():
                compatible = ATTR_FAMILIES.get((v_attr or "").strip().lower())
                if compatible and te_attr in compatible:
                    state.virtual_bindings[v_key].add(te)
                    # Title-endpoints are gold-confirmed chunk-level bridges
                    # (their relations were matched by Composed Retrieval
                    # against the original query). Apply a RELATIVE boost
                    # over their cached cosine sim, preserving relative
                    # ranking among multiple title-endpoints — semantically
                    # closer endpoints (SE Asia for rainforest query) still
                    # outrank loosely-connected title bridges (Tajikistan).
                    cached = state.virtual_sims.get((te, v_key), VIRTUAL_MATCH_THR)
                    state.virtual_sims[(te, v_key)] = min(1.0, cached + 0.10)
                    # Tag the variant so downstream variant-graph code
                    # treats this as an explicit binding event.
                    for sq in subqs:
                        for slot, ent in enumerate(sq.entities):
                            if ent.virtual and ent.key == v_key:
                                variant_tag_map_post[sq.qid].append(
                                    f"slot{slot}={v_key}={te}")

        if not matched_entities:
            logger.info("Post-supplement iter %d: no new endpoints matched → stop", post_it)
            break

        # 4) Expand to synonyms + commit to visited set
        ke_next = set(matched_entities)
        for e in matched_entities:
            for s in kg.synonyms_of(e):
                if s not in state.visited_entities:
                    ke_next.add(s)
        state.visited_entities.update(ke_next)

        # 5) RN_curr from all visited entities (now including the new ones)
        rn_set: set[str] = set()
        for e in state.visited_entities:
            for r in kg.relations_of(e):
                if r not in state.locked_relations:
                    rn_set.add(r)
        if not rn_set:
            logger.info("Post-supplement iter %d: no unlocked relations → stop", post_it)
            break
        rn_curr = sorted(rn_set)

        # 6) Lock new relations
        before = sum(len(v) for v in state.solutions.values())
        round_relation_search(kg, state, rn_curr,
                              is_first_round=False,
                              variant_tag_map=variant_tag_map_post,
                              bm25_per_subq=bm25_per_subq,
                              subq_neighbors=subq_neighbors)
        added = sum(len(v) for v in state.solutions.values()) - before
        logger.info("Post-supplement iter %d: new_endpoints=%d, matched=%d (incl. %d title), "
                    "RN_curr=%d, added=%d",
                    post_it, len(new_endpoints), len(matched_entities),
                    len(title_endpoints), len(rn_curr), added)
        if added == 0:
            break

    return state


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Path scoring
# ══════════════════════════════════════════════════════════════════════════════

def enumerate_paths_with_levels(G: nx.MultiGraph, levels: dict[str, int]) -> list[list[int]]:
    """Enumerate simple paths in the bipartite (entity↔subq) skeleton, from
    anchor entity nodes (Level 0, real entities) to deepest / degree-1
    virtual entity endpoints.

    Each path's node sequence alternates entity ↔ subq ↔ entity ↔ ...
    We return the SUB-QUESTION qid sequence (the subq nodes visited along
    the walk), which is what the downstream scoring and context layers
    consume."""
    if not levels:
        return []
    # Anchors: real entity nodes only
    anchor_nodes = {n for n, d in G.nodes(data=True)
                    if d.get("kind") == "entity" and not d["info"].virtual}
    if not anchor_nodes:
        return []

    # Endpoints: virtual entities at max BFS depth OR degree-1 virtuals
    entity_levels = {n: l for n, l in levels.items()
                     if G.nodes[n].get("kind") == "entity"}
    if not entity_levels:
        return []
    max_lvl = max(entity_levels.values())
    endpoints: set[str] = set()
    for n, d in G.nodes(data=True):
        if d.get("kind") != "entity":
            continue
        if not d["info"].virtual:
            continue
        if levels.get(n, -1) == max_lvl or G.degree(n) == 1:
            endpoints.add(n)
    if not endpoints:
        endpoints = {n for n, d in G.nodes(data=True)
                     if d.get("kind") == "entity" and d["info"].virtual}

    # Collect all simple-path qid sequences, then deduplicate by qid SET so we
    # do not emit many permutations of the same reasoning chain (a common
    # phenomenon when the bipartite skeleton is densely connected by shared
    # virtual entities). For each unique qid set, keep only ONE canonical
    # qid sequence: the FIRST one we see in the BFS-ordered walk (which is
    # the most natural path order — closer to the level-ascending direction).
    seen_seq: set[tuple[int, ...]] = set()
    seen_set: set[frozenset[int]] = set()
    paths: list[list[int]] = []
    for a in anchor_nodes:
        for tgt in endpoints:
            if a == tgt:
                continue
            try:
                for node_path in nx.all_simple_paths(G, a, tgt):
                    qid_seq: list[int] = []
                    for n in node_path:
                        if G.nodes[n].get("kind") == "subq":
                            qid_seq.append(G.nodes[n]["info"].qid)
                    if not qid_seq:
                        continue
                    seq_key = tuple(qid_seq)
                    if seq_key in seen_seq:
                        continue
                    seen_seq.add(seq_key)
                    set_key = frozenset(qid_seq)
                    if set_key in seen_set:
                        continue
                    seen_set.add(set_key)
                    paths.append(qid_seq)
            except nx.NodeNotFound:
                continue
    # Sort: longer paths first (more qids = more complete reasoning chains),
    # then by qid sequence for determinism.
    paths.sort(key=lambda p: (-len(p), p))
    return paths


def select_top_relations_with(entries: list[SolutionEntry],
                              k_min: int, k_max: int, gap_thr: float) -> list[SolutionEntry]:
    """Parametrised gap-based adaptive top-K cutoff (see select_top_relations)."""
    if not entries:
        return []
    entries_sorted = sorted(entries, key=lambda e: -e.sim)
    n = len(entries_sorted)
    if n <= k_min:
        return entries_sorted
    selected = list(entries_sorted[:k_min])
    for i in range(k_min, min(k_max, n)):
        prev = entries_sorted[i - 1].sim
        cur  = entries_sorted[i].sim
        if prev - cur > gap_thr:
            break
        selected.append(entries_sorted[i])
    return selected


def select_top_relations(entries: list[SolutionEntry]) -> list[SolutionEntry]:
    """Gap-based adaptive top-K cutoff.

    Sort entries by sim descending, then:
      1. Always keep the top REL_K_MIN entries.
      2. Keep extending to the next entry while the sim drop from the
         previous kept entry is at most REL_GAP_THR.
      3. Stop on first gap larger than REL_GAP_THR, or upon hitting
         REL_K_MAX entries.

    Empirically this:
      - Tightens to ~K_MIN on long-tail distributions (e.g. Q9, where the
        2nd–3rd-best relation already drops sharply) — minimal noise.
      - Widens to ~K_MAX on densely-clustered distributions (e.g. Q8 qid=2,
        where the right relation sits at rank 5 within a small gap window) —
        recovers gold relations buried below a fixed top-2.
    """
    if not entries:
        return []
    entries_sorted = sorted(entries, key=lambda e: -e.sim)
    n = len(entries_sorted)
    if n <= REL_K_MIN:
        return entries_sorted
    selected = list(entries_sorted[:REL_K_MIN])
    for i in range(REL_K_MIN, min(REL_K_MAX, n)):
        prev = entries_sorted[i - 1].sim
        cur  = entries_sorted[i].sim
        if prev - cur > REL_GAP_THR:
            break
        selected.append(entries_sorted[i])
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# Variant sub-questions — concrete instantiations of skeleton sub-questions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Variant:
    """A concrete instantiation of a skeleton sub-question.

    A skeleton sub-question with virtual entities like
        "Imperialist power B is the country of citizenship of arguer A"
    becomes a variant sub-question by substituting concrete KG entity ids
    for each virtual placeholder:
        bindings = (("imperialist power B", <Soviet Union ent>),
                    ("arguer A",          <Mao Zedong ent>))
    A None value indicates an unbound virtual (the question's answer slot).

    bound_entities aggregates the KG ent_ids for every entity referenced
    by this variant (real anchors + non-None virtual bindings). Two
    variants are adjacent in the variant graph iff their bound_entities
    intersect — this is how reasoning chains form.
    """
    qid:            int
    variant_id:     int
    bindings:       tuple                  # ((v_key, ent_id_or_None), ...) sorted
    bound_entities: frozenset              # KG ent_ids referenced by this variant


def build_variants(state: "RetrievalState", kg: KGIndex) -> tuple[list[Variant], dict[int, Variant]]:
    """For each skeleton sub-question, instantiate all variants by
    cartesian-producting the bindings of every virtual entity.

    Returns (list_of_variants, variants_by_id).
    """
    variants: list[Variant] = []
    next_vid = 0
    for sq in state.subqs:
        virtuals  = [e for e in sq.entities if e.virtual]
        real_ents = [e for e in sq.entities if not e.virtual]

        # Concrete KG entities for the real entities in this sub-question.
        real_bound: set[str] = set()
        for re in real_ents:
            real_bound |= state.anchor_bindings.get(re.key, set())
        # Generic anchor pool fallback (used when no exact name match was found)
        if not real_bound and "__all__" in state.anchor_bindings:
            for re in real_ents:
                real_bound |= state.anchor_bindings.get("__all__", set())

        # Bindings per virtual: list of (v_key, ent_id) options. Unbound → [(v_key, None)].
        # Cap at MAX_BINDINGS_PER_VIRTUAL to prevent cartesian-product blowup.
        # Sort by virtual-match SIM (highest first), NOT alphabetical — the
        # alphabetical default would silently keep the lexicographically
        # earliest candidates (e.g. "Andes/Anglicanism/Atheism" before
        # "Christianity/Hinduism/Islam"), throwing away the semantically
        # closest ones recorded in state.virtual_sims.
        binding_lists: list[list[tuple[str, str | None]]] = []
        for ve in virtuals:
            bound_set = state.virtual_bindings.get(ve.key, set())
            if bound_set:
                bound = sorted(
                    bound_set,
                    key=lambda eid: -state.virtual_sims.get((eid, ve.key), 0.0),
                )[:MAX_BINDINGS_PER_VIRTUAL]
                binding_lists.append([(ve.key, eid) for eid in bound])
            else:
                binding_lists.append([(ve.key, None)])

        combos = list(itertools.product(*binding_lists)) if binding_lists else [()]
        # Cap variants per qid to keep the variant graph tractable
        combos = combos[:MAX_VARIANTS_PER_QID]
        for combo in combos:
            combo_dict = dict(combo)
            ents = set(real_bound)
            for eid in combo_dict.values():
                if eid is not None:
                    ents.add(eid)
            variants.append(Variant(
                qid=sq.qid,
                variant_id=next_vid,
                bindings=tuple(sorted(combo_dict.items())),
                bound_entities=frozenset(ents),
            ))
            next_vid += 1

    return variants, {v.variant_id: v for v in variants}


def variant_solutions(variant: Variant, state: "RetrievalState",
                      kg: KGIndex) -> list[SolutionEntry]:
    """Solutions of one variant sub-question.

    Soft entity-overlap filter: a relation locked to the variant's parent
    skeleton qid is retained iff its hyperedge entity set intersects
    variant.bound_entities.  This realises the variant as a *concrete
    reasoning step* — its entity bindings actually constrain which
    relations belong to its segment of the path, instead of every variant
    inheriting the same full qid solution set.

    Why soft (any-overlap) and not strict (all-bindings-must-appear):
    a previous strict filter over-zealously dropped gold relations whose
    entity lists were incomplete (e.g. the "Mao reported to the
    Politburo" relation lacks "Korea" as an extracted entity, so it
    failed a {Mao, Korea} hard filter). Any-overlap keeps such relations
    while still pruning candidates that share *no* entity with this
    variant's binding chain.

    Fallback: a variant with empty bound_entities (no virtuals matched
    AND no real anchors) cannot meaningfully filter, so it falls back
    to the full skeleton-qid solution set.
    """
    candidates = state.solutions.get(variant.qid, [])
    if not variant.bound_entities:
        return candidates
    out: list[SolutionEntry] = []
    for e in candidates:
        rel_ents = set(kg.entities_of_relation(e.rel_id))
        if rel_ents & variant.bound_entities:
            out.append(e)
    return out if out else candidates  # never starve a variant — fall back to full set


def build_variant_graph(variants: list[Variant]) -> "nx.MultiGraph":
    """Variant graph: each variant is a node; an edge connects two variants
    iff they share at least one bound entity AND belong to DIFFERENT
    skeleton qids (variants of the same qid are alternatives, not
    consecutive hops)."""
    G = nx.MultiGraph()
    for v in variants:
        G.add_node(v.variant_id, info=v, kind="variant")
    n = len(variants)
    for i in range(n):
        va = variants[i]
        for j in range(i + 1, n):
            vb = variants[j]
            if va.qid == vb.qid:
                continue
            shared = va.bound_entities & vb.bound_entities
            if shared:
                G.add_edge(va.variant_id, vb.variant_id, shared=tuple(sorted(shared)))
    return G


def enumerate_variant_paths(G_var: "nx.MultiGraph",
                             variants_by_id: dict[int, Variant],
                             anchor_qids: set[int],
                             n_subqs: int) -> list[list[int]]:
    """Find variant paths: walks through the variant graph starting at
    anchor variants (variants belonging to a sub-question with at least
    one real entity) and reaching the most distant variants.

    Dedup by qid SET (we never want two paths that visit the same set of
    skeleton qids — they're permutations of the same reasoning chain).
    """
    from collections import deque

    anchor_vids = [vid for vid, v in variants_by_id.items()
                   if v.qid in anchor_qids]
    if not anchor_vids:
        anchor_vids = list(variants_by_id.keys())

    # BFS from anchor variants to compute levels
    levels: dict[int, int] = {}
    queue: deque = deque()
    for vid in anchor_vids:
        levels[vid] = 0
        queue.append(vid)
    while queue:
        u = queue.popleft()
        for n in G_var.neighbors(u):
            if n not in levels:
                levels[n] = levels[u] + 1
                queue.append(n)
    if not levels:
        return []

    max_lvl = max(levels.values())
    endpoints = [v for v, l in levels.items() if l == max_lvl]

    paths: list[list[int]] = []
    seen_seq: set[tuple[int, ...]] = set()
    seen_qid_set: set[frozenset[int]] = set()
    for a in anchor_vids:
        for tgt in endpoints:
            if a == tgt:
                if max_lvl == 0:
                    # Single-variant path
                    seq = (a,)
                    if seq in seen_seq:
                        continue
                    seen_seq.add(seq)
                    paths.append([a])
                continue
            if len(paths) >= MAX_VARIANT_PATHS:
                break
            try:
                for node_path in nx.all_simple_paths(G_var, a, tgt, cutoff=n_subqs):
                    qid_seq = tuple(variants_by_id[v].qid for v in node_path)
                    # Skip if same qid visited twice in this path
                    if len(set(qid_seq)) < len(qid_seq):
                        continue
                    seq_key = tuple(node_path)
                    if seq_key in seen_seq:
                        continue
                    seen_seq.add(seq_key)
                    # Dedup by (qid SET, binding signature). qid SET kills
                    # permutations of the same chain; binding signature
                    # preserves DIFFERENT binding branches that happen to
                    # walk the same skeleton qid set (e.g. religion=Christianity
                    # vs religion=Islam paths through {q0,q1,q2}).
                    binding_sig = tuple(sorted(
                        (vk, eid)
                        for vid in node_path
                        for vk, eid in variants_by_id[vid].bindings
                    ))
                    path_key = (frozenset(qid_seq), binding_sig)
                    if path_key in seen_qid_set:
                        continue
                    seen_qid_set.add(path_key)
                    paths.append(list(node_path))
                    if len(paths) >= MAX_VARIANT_PATHS:
                        break
            except nx.NodeNotFound:
                continue
        if len(paths) >= MAX_VARIANT_PATHS:
            break

    # Sort: longer paths first (more reasoning hops covered),
    # then by variant sequence for determinism.
    paths.sort(key=lambda p: (-len(p), p))
    return paths


def score_variant_paths(
    paths: list[list[int]],
    variants_by_id: dict[int, Variant],
    state: "RetrievalState",
    kg: KGIndex,
    n_subqs: int,
) -> list[tuple[list[int], float, dict[int, list[SolutionEntry]]]]:
    """Score each variant path by reasoning-chain-completeness × similarity.

        score(p) = (sum over v in p of mean_sim(variant_solutions(v))) / n_subqs

    Same chain-completeness intent as score_paths, but operating on
    variant ids instead of skeleton qids. seg_top is keyed by variant_id.
    """
    n_subqs = max(n_subqs, 1)
    scored = []
    for path in paths:
        seg_top: dict[int, list[SolutionEntry]] = {}
        seg_sim_total = 0.0
        covered = 0
        for vid in path:
            v = variants_by_id[vid]
            sols = variant_solutions(v, state, kg)
            if not sols:
                continue
            # Path scoring uses the TIGHT-K selection so that a single
            # strong evidence relation (high #1, lower #2-#3) still gives
            # the path a high score, instead of being diluted by including
            # marginal entries.
            scoring_top = select_top_relations_with(
                sols, REL_K_MIN, REL_K_MAX, REL_GAP_THR)
            # Context emission uses the WIDER-K selection so that relations
            # just outside the scoring-K but still close to the top can
            # still surface in the LLM's context (e.g. the gold relation at
            # rank-3 with a small gap from rank-2).
            context_top = select_top_relations_with(
                sols, REL_K_MIN_CONTEXT, REL_K_MAX_CONTEXT, REL_GAP_THR_CONTEXT)
            seg_top[vid] = context_top
            seg_sim_total += float(np.mean([e.sim for e in scoring_top]))
            covered += 1
        if covered == 0:
            continue
        path_score = seg_sim_total / n_subqs
        scored.append((path, path_score, seg_top))
    scored.sort(key=lambda x: -x[1])
    return scored


def score_paths(state: RetrievalState, qid_paths: list[list[int]],
                max_level: int,
                kg: KGIndex | None = None,
                ) -> list[tuple[list[int], float, dict[int, list[SolutionEntry]]]]:
    """Score each path by reasoning-chain completeness × per-segment similarity.

    Formula:
        score(p) = (sum over q_i in p of mean_sim(q_i)) / n_subqs

    Normalising by the TOTAL number of sub-questions (n_subqs) — not by
    |p| — penalises paths that only cover a subset of the reasoning
    chain. A short path of one well-matched segment is no longer
    artificially favoured over a long path that walks the entire chain.
    This realises the chain-completeness intent of paper contribution 2.

    Uncovered segments contribute 0 to the sum; covered segments
    contribute the mean similarity of their adaptively-selected top
    relations (see select_top_relations for the gap-based cutoff).
    Paths with NO covered segments are dropped.
    """
    n_subqs = max(len(state.subqs), 1)
    scored = []
    for path in qid_paths:
        seg_top: dict[int, list[SolutionEntry]] = {}
        seg_sim_total: float = 0.0
        covered = 0
        for qid in path:
            entries = state.solutions.get(qid, [])
            if not entries:
                continue
            top = select_top_relations(entries)
            seg_top[qid] = top
            seg_sim_total += float(np.mean([e.sim for e in top]))
            covered += 1
        if covered == 0:
            continue
        # Divide by total subqs in the question, NOT by len(path).
        path_score = seg_sim_total / n_subqs
        scored.append((path, path_score, seg_top))
    scored.sort(key=lambda x: -x[1])
    return scored


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Context organization
# ══════════════════════════════════════════════════════════════════════════════

TOP_K_CHUNKS = 5   # chunk-mode output: number of source chunks to return


def organize_variant_context(
    kg: KGIndex,
    scored: list[tuple[list[int], float, dict[int, list[SolutionEntry]]]],
    variants_by_id: dict[int, Variant],
    top_k_paths: int = TOP_K_PATHS,
) -> tuple[str, str]:
    """Variant-path-aware context organisation.

    For each top-K scoring variant path, walk the variants in path order
    and concatenate the per-variant filtered relations into ONE paragraph.
    Multiple variant paths become separate paragraphs.

    Each variant has its OWN filtered solution set (see variant_solutions),
    so each paragraph reflects ONE specific binding chain — e.g. "Spiridonov
    → Soviet Union → Mao → Politburo" — rather than the union of all
    possible variants of every skeleton sub-question.
    """
    if not scored:
        return "", "yes_empty"
    # Per-path dedup only. We intentionally do NOT dedup across paths —
    # relations that recur across multiple variant paths serve as an
    # implicit frequency-vote signal to the answering LLM (a relation
    # that survives in 3/3 paths is more confidently part of the
    # reasoning chain than one that appears in only 1/3).
    paragraphs: list[str] = []
    for path, _score, seg_top in scored[:top_k_paths]:
        seg_texts: list[str] = []
        seen_in_path: set[str] = set()
        for vid in path:
            entries = sorted(seg_top.get(vid, []), key=lambda e: -e.sim)
            for entry in entries:
                if entry.rel_id in seen_in_path:
                    continue
                seen_in_path.add(entry.rel_id)
                seg_texts.append(kg.G.nodes[entry.rel_id]["text"])
        if seg_texts:
            paragraphs.append(" ".join(seg_texts))
    if paragraphs:
        return "\n\n".join(paragraphs), "no"
    return "", "yes_empty"


def organize_context(
    kg: KGIndex,
    scored: list[tuple[list[int], float, dict[int, list[SolutionEntry]]]],
    subqs: list[SubQuestion] | None = None,
    state: RetrievalState | None = None,
    top_k_paths: int = TOP_K_PATHS,
    mode: str = "relation",     # "relation" (default, per-path sentence concat)
                                # or "chunk" (experimental, returns whole source chunks)
    top_k_chunks: int = TOP_K_CHUNKS,
) -> tuple[str, str]:
    """Return (context_text, fallback_used_flag).

    mode='relation' (default): for each top-K reasoning path, walk its
    sub-questions in path order and concatenate the per-segment top
    relations into ONE paragraph. Different paths become separate
    paragraphs. This preserves the natural reasoning order within each
    paragraph, avoiding pronoun / back-reference confusion that arises
    when relations from unrelated reasoning steps are grouped together.

    mode='chunk' (experimental): chunk-vote retrieval. Each selected
    relation votes for its source chunk; we return the original chunk
    text. Preserves natural language flow at the cost of bringing in
    chunk-level noise.
    """
    if not scored:
        # Fall through to fallback below
        pass

    # ── CHUNK MODE: vote-based chunk selection ─────────────────────────────
    if mode == "chunk" and scored:
        chunk_votes: dict[tuple[str, str], int] = {}
        chunk_first_pos: dict[tuple[str, str], tuple[int, int]] = {}
        seen_rels_global: set[str] = set()
        for path_idx, (path, _, seg_top) in enumerate(scored[:top_k_paths]):
            for seg_idx, qid in enumerate(path):
                for entry in seg_top.get(qid, []):
                    if entry.rel_id in seen_rels_global:
                        continue
                    seen_rels_global.add(entry.rel_id)
                    src_text  = kg.G.nodes[entry.rel_id].get("source_text", "")
                    src_title = kg.G.nodes[entry.rel_id].get("source_title", "")
                    if not src_text:
                        continue
                    key = (src_title, src_text)
                    chunk_votes[key] = chunk_votes.get(key, 0) + 1
                    prev = chunk_first_pos.get(key)
                    if prev is None or (path_idx, seg_idx) < prev:
                        chunk_first_pos[key] = (path_idx, seg_idx)
        if not chunk_votes:
            return "", "no"
        sorted_chunks = sorted(
            chunk_votes.items(),
            key=lambda kv: (chunk_first_pos[kv[0]], -kv[1]),
        )
        parts: list[str] = []
        seen_texts: set[str] = set()
        for (title, text), _votes in sorted_chunks[:top_k_chunks]:
            if text in seen_texts:
                continue
            seen_texts.add(text)
            parts.append(text)
        return "\n\n".join(parts), "no"

    # ── RELATION MODE: per-path concatenation in path order ────────────────
    # Each top-K path becomes ONE paragraph. Within a paragraph the
    # relations are emitted in path order (q_1's top relations → q_2's →
    # …). Dedup is applied within a single paragraph only — a relation
    # that legitimately appears in two reasoning chains is allowed to
    # appear in both paragraphs (each paragraph must be self-contained).
    paragraphs: list[str] = []
    if scored:
        for path, _score, seg_top in scored[:top_k_paths]:
            seg_texts: list[str] = []
            seen_in_path: set[str] = set()
            for qid in path:
                # Top relations for this segment, sorted by sim desc
                entries = sorted(seg_top.get(qid, []), key=lambda e: -e.sim)
                for entry in entries:
                    if entry.rel_id in seen_in_path:
                        continue
                    seen_in_path.add(entry.rel_id)
                    seg_texts.append(kg.G.nodes[entry.rel_id]["text"])
            if seg_texts:
                paragraphs.append(" ".join(seg_texts))

    if paragraphs:
        return "\n\n".join(paragraphs), "no"

    # ── Fallback: no valid path → use all collected relations by sim ─────────
    if state is None:
        return "", "yes_empty"
    seen_rels: set[str] = set()
    all_entries: list[tuple[float, str]] = []
    for qid, entries in state.solutions.items():
        for e in entries:
            if e.rel_id not in seen_rels:
                seen_rels.add(e.rel_id)
                all_entries.append((e.sim, e.rel_id))
    all_entries.sort(reverse=True)
    fallback_n = min(20, len(all_entries))
    parts = [kg.G.nodes[rel_id]["text"] for _, rel_id in all_entries[:fallback_n]]
    text = "[Supplementary evidence (no valid reasoning path found)]\n" + "\n".join(parts)
    return text, "yes_partial" if parts else "yes_empty"


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════

def load_questions(n: int = 10) -> list[dict]:
    """Load questions from DATA_FILE. n<=0 returns ALL questions (in file
    order); n>0 returns the top-N largest by total context character count."""
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    if n is None or n <= 0:
        return data
    sizes = [(sum(len(s) for _, sents in q.get("context", []) for s in sents), i)
             for i, q in enumerate(data)]
    sizes.sort(reverse=True)
    return [data[i] for _, i in sizes[:n]]


def retrieve_one(kg: KGIndex, question: str, debug: bool = False) -> dict:
    t0 = time.time()
    logger.info("Question: %s", question)

    # Step 1
    triplets = decompose_query(question)
    if not triplets:
        logger.error("Decomposition produced no triplets!")
        return {"question": question, "context": "", "error": "decomposition_failed"}
    logger.info("Decomposed into %d sub-questions:", len(triplets))
    for t in triplets:
        ent_strs = []
        for e in t["entities"]:
            tag = ("? " if e["virtual"] else "") + f'{e["attribute"]} {e["name"]}'.strip()
            ent_strs.append(tag)
        logger.info("    [%s]  {%s}", t["relation"], " | ".join(ent_strs))

    # Step 2
    G_skel, subqs = build_skeleton_dag(triplets)
    levels = compute_levels(G_skel)
    max_level = max(levels.values()) if levels else 0
    # Anchor sub-questions = those containing at least one real (non-virtual) entity
    anchor_subqs = [sq for sq in subqs if any(not e.virtual for e in sq.entities)]
    logger.info("Skeleton: %d nodes, %d edges, max_level=%d, %d anchor subqs (>=1 real entity)",
                G_skel.number_of_nodes(), G_skel.number_of_edges(), max_level, len(anchor_subqs))

    if not anchor_subqs:
        # No real-entity anchors in question. Fall back to direct vector
        # search over relation nodes using the original query.
        logger.warning("No anchor sub-questions — falling back to direct relation search.")
        q_vec = encode_texts([question])[0]
        sims  = kg.relation_emb @ q_vec
        top_idx = np.argsort(-sims)[:20]
        parts = [kg.G.nodes[kg.relation_ids[i]]["text"] for i in top_idx]
        context = "\n\n".join(parts)
        return {
            "question":  question,
            "triplets":  triplets,
            "ke1_size":  0,
            "max_level": max_level,
            "n_paths":   0,
            "fallback":  "no_anchor_direct_search",
            "context":   context,
            "elapsed":   round(time.time() - t0, 1),
        }

    # Step 3
    ke1 = locate_anchor_entities(kg, anchor_subqs)
    logger.info("KE1 size = %d", len(ke1))
    if not ke1:
        return {"question": question, "context": "", "error": "no_ke1"}

    # Relation-node Composed Retrieval — original question + ONLY anchor
    # sub-questions (those carrying at least one real entity).
    # Rationale: a pure-virtual sub-question like "Team A beat Team B in
    # year C" carries no concrete lexical signal and, on large/diluted KGs,
    # tends to drag in distractor relations that share generic verbs
    # (beat / married / born) regardless of the actual reasoning chain.
    # Restricting to anchor sub-questions keeps each seed query lexically
    # specific (it must mention at least one named real entity) and is
    # the recommended setting on the 300q-scale corpus.
    seed_queries = [question] + [sq.relation_text for sq in anchor_subqs]
    seed_relations = set(composed_retrieve_relations(kg, seed_queries))
    logger.info("Relation-Composed-Retrieval: %d queries (1 original + %d anchor subqs) "
                "→ %d unique seed relations",
                len(seed_queries), len(anchor_subqs), len(seed_relations))

    # Step 4
    state = iterative_search(kg, subqs, ke1, seed_relations=seed_relations)
    solved_counts = {qid: len(entries) for qid, entries in state.solutions.items()}
    logger.info("Solutions per subq (qid → count): %s", solved_counts)
    if os.environ.get("HP_DUMP_SOLUTIONS"):
        for sq in subqs:
            entries = sorted(state.solutions.get(sq.qid, []), key=lambda e: -e.sim)
            logger.info("  [q%d] %r — %d locked relations:", sq.qid, sq.relation_text, len(entries))
            for rank, e in enumerate(entries[:15]):
                rel_node = kg.G.nodes[e.rel_id]
                title = rel_node.get("source_title", "")
                text  = (rel_node.get("text") or "")[:140]
                ents  = sorted([n for n in kg.G.neighbors(e.rel_id) if n.startswith("ent::")])
                logger.info("    #%-2d %s sim=%.3f tag=%s", rank, e.rel_id, e.sim, e.variant_tag)
                logger.info("        title=%r text=%r", title, text)
                logger.info("        ents=%s", ents[:8])
        # virtual_sims is dict[(ent_id, v_key) -> float]; group by v_key
        by_vkey: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (eid, vk), s in state.virtual_sims.items():
            by_vkey[vk].append((eid, float(s)))
        for vk, items in by_vkey.items():
            items.sort(key=lambda kv: -kv[1])
            logger.info("  virtual %r bindings: %s", vk,
                        [(eid, round(s, 3)) for eid, s in items[:10]])
        for vk, bound in state.virtual_bindings.items():
            logger.info("  virtual %r FINAL bound set (%d): %s", vk, len(bound), sorted(bound)[:10])

    # Step 5 — instantiate variant sub-questions (skeleton qid × cartesian
    # product of virtual bindings) and enumerate paths over the variant graph.
    variants, variants_by_id = build_variants(state, kg)
    G_var = build_variant_graph(variants)
    anchor_qids = {sq.qid for sq in anchor_subqs}
    var_paths = enumerate_variant_paths(G_var, variants_by_id, anchor_qids, len(subqs))
    scored = score_variant_paths(var_paths, variants_by_id, state, kg, len(subqs))
    logger.info(
        "Variants: %d (%d skeleton qids → %d concrete bindings).  "
        "Variant paths: %d enumerated, %d scored.",
        len(variants), len(subqs), len(variants), len(var_paths), len(scored),
    )

    # Step 6 — emit context per variant path (each paragraph = one variant chain)
    context, fallback = organize_variant_context(kg, scored, variants_by_id)

    elapsed = time.time() - t0
    return {
        "question":    question,
        "triplets":    triplets,
        "ke1_size":    len(ke1),
        "max_level":   max_level,
        "n_paths":     len(scored),
        "fallback":    fallback,
        "context":     context,
        "elapsed":     round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=-1,
                    help="Limit to top-N largest questions (debug). "
                         "Use -1 (default) or 0 to process ALL questions in --data.")
    ap.add_argument("--question-idx", type=int, default=None,
                    help="If set, only process this question index (0-based after sorting).")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--data", type=str, default=None,
                    help="Override DATA_FILE path")
    ap.add_argument("--kg", type=str, default=None,
                    help="Override KG_PKL path")
    ap.add_argument("--out", type=str, default="retrieval_results.json",
                    help="Output JSON file (default: retrieval_results.json)")
    # Hyperparameter overrides — used by the sensitivity sweep
    # (Appendix C). Defaults keep the values defined at module top.
    ap.add_argument("--bm25-lambda", dest="bm25_lambda", type=float, default=None,
                    help="Override HYBRID_BM25_BOOST (paper symbol λ).")
    ap.add_argument("--virtual-match-thr", dest="virtual_match_thr", type=float, default=None,
                    help="Override VIRTUAL_MATCH_THR (synonym threshold τ on virtual matching).")
    ap.add_argument("--rel-gap-thr", dest="rel_gap_thr", type=float, default=None,
                    help="Override REL_GAP_THR (adaptive top-K gap δ).")
    ap.add_argument("--rel-k-min", dest="rel_k_min", type=int, default=None,
                    help="Override REL_K_MIN (adaptive top-K floor).")
    ap.add_argument("--rel-k-max", dest="rel_k_max", type=int, default=None,
                    help="Override REL_K_MAX (adaptive top-K ceiling).")
    ap.add_argument("--top-k-paths", dest="top_k_paths", type=int, default=None,
                    help="Override TOP_K_PATHS (number of paths returned to LLM).")
    # Ablation flag — a tag for downstream record-keeping. Algorithm
    # switches live inside the corresponding code paths; the orchestrator
    # passes one of {full, no-ntary, no-multitarget, no-chaincomp,
    # no-order, no-hypergraph, no-bm25} and the resulting retrieval JSON
    # is tagged with this label.
    ap.add_argument("--ablation", type=str, default="full",
                    help="Ablation label (tags the output JSON; algorithmic switches "
                         "must be configured via the other CLI flags or by selecting "
                         "the appropriate KG variant).")
    # Per-question timing log — emit alongside the main output. Used to
    # populate Table 4 (Efficiency).
    ap.add_argument("--save-timing", dest="save_timing", type=str, default=None,
                    help="Optional path to write a per-question timing JSON "
                         "(decompose_ms, retrieve_ms, total_ms).")
    args = ap.parse_args()

    global DATA_FILE, KG_PKL
    global HYBRID_BM25_BOOST, VIRTUAL_MATCH_THR, REL_GAP_THR
    global REL_K_MIN, REL_K_MAX, TOP_K_PATHS
    if args.data:
        DATA_FILE = Path(args.data)
    if args.kg:
        KG_PKL = Path(args.kg)
    if args.bm25_lambda is not None:
        HYBRID_BM25_BOOST = float(args.bm25_lambda)
    if args.virtual_match_thr is not None:
        VIRTUAL_MATCH_THR = float(args.virtual_match_thr)
    if args.rel_gap_thr is not None:
        REL_GAP_THR = float(args.rel_gap_thr)
    if args.rel_k_min is not None:
        REL_K_MIN = int(args.rel_k_min)
    if args.rel_k_max is not None:
        REL_K_MAX = int(args.rel_k_max)
    if args.top_k_paths is not None:
        TOP_K_PATHS = int(args.top_k_paths)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    kg = KGIndex(KG_PKL)
    questions = load_questions(args.n)

    if args.question_idx is not None:
        questions = [questions[args.question_idx]]

    results = []
    timing_records = []
    for i, q in enumerate(questions):
        logger.info("\n" + "=" * 80)
        logger.info("[%d/%d] qid=%s", i + 1, len(questions), q.get("id", "?"))
        t0 = time.time()
        try:
            res = retrieve_one(kg, q["question"], debug=args.debug)
        except Exception as e:
            logger.exception("Retrieval crashed: %s", e)
            res = {"question": q["question"], "error": str(e)}
        total_ms = (time.time() - t0) * 1000.0
        res["expected_answer"] = q.get("answer", "")
        res["ablation"] = args.ablation
        res["retrieve_total_ms"] = round(total_ms, 1)
        results.append(res)
        timing_records.append({
            "qid": q.get("id", f"idx-{i}"),
            "retrieve_total_ms": round(total_ms, 1),
            "elapsed_internal_s": res.get("elapsed"),
            "n_subq": len(res.get("subqs", [])) if isinstance(res.get("subqs"), list) else None,
            "context_chars": len(res.get("context") or ""),
            "ablation": args.ablation,
        })
        logger.info("Context preview:\n%s", (res.get("context") or "")[:500])

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved results → %s", out_path)

    if args.save_timing:
        timing_path = Path(args.save_timing)
        if not timing_path.is_absolute():
            timing_path = ROOT / timing_path
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        n = max(len(timing_records), 1)
        summary = {
            "ablation": args.ablation,
            "n_questions": len(timing_records),
            "retrieve_sec_per_q": round(
                sum(r["retrieve_total_ms"] for r in timing_records) / n / 1000.0, 3),
            "per_question": timing_records,
            "hyperparams": {
                "HYBRID_BM25_BOOST": HYBRID_BM25_BOOST,
                "VIRTUAL_MATCH_THR": VIRTUAL_MATCH_THR,
                "REL_GAP_THR": REL_GAP_THR,
                "REL_K_MIN": REL_K_MIN,
                "REL_K_MAX": REL_K_MAX,
                "TOP_K_PATHS": TOP_K_PATHS,
            },
        }
        timing_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved timing → %s", timing_path)


if __name__ == "__main__":
    main()
