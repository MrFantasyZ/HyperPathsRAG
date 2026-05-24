"""
HyperPathsRAG — KG Construction
Loads events from llm_inspection.json, embeds entities + relations,
builds a bipartite hypergraph, merges near-duplicate entities, saves kg.pkl.

Node types
  entity   : named entity (canonical key = name.lower())
  relation : one extracted event (hyperedge connecting its entities)

Edge types
  hyperedge : connects a relation node to each of its entity nodes
  synonym   : connects two entity nodes whose embeddings exceed MERGE_THRESHOLD
               but have different display names (same-name duplicates are merged)
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
import torch

# ── config ────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent
INSPECTION_JSON = ROOT / "llm_inspection.json"
OUT_DIR         = ROOT / "kg_output"
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MERGE_THRESHOLD = 0.85
EPS             = 1e-8

# ── logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUT_DIR / "build.log", mode="w", encoding="utf-8"),
        ],
        force=True,
    )

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & clean events
# ══════════════════════════════════════════════════════════════════════════════

# When the txt-inspection report was parsed back to JSON, nested parentheses
# in entity strings caused definitions to be stored as e.g.:
#   "facility) (administrative center and congress hall in Moscow"
# instead of the correct "administrative center and congress hall in Moscow".
# The pattern: corrupted = "<label>) (<real definition>"
_CORRUPT_DEF_RE = re.compile(r'^[\w][\w /\-]*\)\s*\((.+?)(?:\))?$')

# Re-run chunks kept the NER label suffix in entity names (e.g.
# "Vladimir Abramovich Rapoport (person)"). Strip such suffixes for consistency
# with the other 202 chunks (where the parser had already stripped them).
_NER_LABEL_RE = re.compile(
    r"\s*\((?:person|organization|location|date|time|work of art|event|"
    r"product|law|language|nationality|religion|title|facility|number|"
    r"country|date range)\)\s*$",
    re.IGNORECASE,
)


def _fix_definition(defn: str) -> str:
    defn = defn.strip().rstrip(')')
    m = _CORRUPT_DEF_RE.match(defn)
    if m:
        candidate = m.group(1).strip()
        if len(candidate) > 3:
            return candidate
    return defn


def _clean_entity_name(name: str) -> str:
    """Strip NER-label suffix like ' (person)' from entity display names."""
    return _NER_LABEL_RE.sub("", name).strip()


def load_events() -> list[dict]:
    """Load inspection JSON and return a flat, deduplicated list of events."""
    data = json.loads(INSPECTION_JSON.read_text(encoding="utf-8"))
    events: list[dict] = []
    seen_relations: set[str] = set()

    for chunk in data:
        title = chunk.get("title", "")
        text  = chunk.get("text", "")
        for ev in chunk.get("events", []):
            rel = ev.get("relation", "").strip()
            if not rel or rel in seen_relations:
                continue
            seen_relations.add(rel)

            clean_ents: list[dict] = []
            for e in ev.get("entities", []):
                name = _clean_entity_name(e.get("name", "").strip())
                defn = _fix_definition(e.get("definition", ""))
                if name:
                    clean_ents.append({"name": name, "definition": defn})

            if clean_ents:
                events.append({
                    "entities":     clean_ents,
                    "relation":     rel,
                    "source_title": title,
                    "source_text":  text,
                })

    logger.info("Loaded %d unique events from %d chunks", len(events), len(data))
    return events


# ══════════════════════════════════════════════════════════════════════════════
# 2. Embedding
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
    tok, model = get_embedder()
    all_vecs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        v = out.last_hidden_state[:, 0, :].cpu().numpy()   # CLS pooling
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        all_vecs.append(v / np.maximum(norms, EPS))
    return np.vstack(all_vecs)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Build hypergraph
# ══════════════════════════════════════════════════════════════════════════════

def build_hypergraph(events: list[dict]) -> nx.MultiGraph:

    # ── 3a. collect unique entities (accumulate all unique definitions) ──────
    ent_display: dict[str, str] = {}            # canonical_lower → first-seen display name
    ent_def_set: dict[str, list[str]] = defaultdict(list)  # canonical_lower → list of unique defs

    for ev in events:
        for e in ev["entities"]:
            key = e["name"].lower()
            if key not in ent_display:
                ent_display[key] = e["name"]
            d = e["definition"].strip()
            if d and d not in ent_def_set[key]:
                ent_def_set[key].append(d)

    # Merge multiple definitions with "; " so the merged entity isn't narrowly defined
    # by a single chunk's perspective (e.g., "Russian: Nationality of Alek Rapoport"
    # would be enriched with definitions from other chunks where "Russian" appears).
    ent_def: dict[str, str] = {k: "; ".join(defs) for k, defs in ent_def_set.items()}

    ent_keys = list(ent_display.keys())
    logger.info("Unique entities: %d (avg %.1f definitions/entity)",
                len(ent_keys),
                sum(len(d) for d in ent_def_set.values()) / max(1, len(ent_keys)))

    # ── 3b. dual entity embeddings (name-only + name:def) ────────────────────
    name_inputs    = [ent_display[k] for k in ent_keys]
    nameDef_inputs = [f"{ent_display[k]}: {ent_def[k]}" for k in ent_keys]
    logger.info("Embedding %d entities (name-only + name:def) …", len(ent_keys))
    ent_name_vecs    = encode_texts(name_inputs)         # (n_ent, 1024)
    ent_nameDef_vecs = encode_texts(nameDef_inputs)      # (n_ent, 1024)

    # ── 3c. embed relation texts (prefixed with source_title) ──────────────
    # Title carries chunk-level context (e.g. "1894–95 FA Cup") that the
    # extracted sentence may omit. Format: "(<title>: <relation text>)".
    rel_texts = [
        f"({ev.get('source_title','').strip()}: {ev['relation']})"
        if ev.get("source_title") else ev["relation"]
        for ev in events
    ]
    logger.info("Embedding %d relation texts (with title context) …", len(rel_texts))
    rel_vecs = encode_texts(rel_texts)                   # (n_rel, 1024)

    # ── 3d. build graph ──────────────────────────────────────────────────────
    G = nx.MultiGraph()

    # entity nodes (store both embeddings)
    for i, key in enumerate(ent_keys):
        G.add_node(
            f"ent::{key}",
            type           = "entity",
            canonical      = key,
            display        = ent_display[key],
            definition     = ent_def[key],
            embedding      = ent_nameDef_vecs[i].astype(np.float32),  # name+def (legacy key)
            name_embedding = ent_name_vecs[i].astype(np.float32),     # name-only
        )

    # ── 3d. title-as-entity nodes ────────────────────────────────────────────
    # The chunk title (Wikipedia article name) is the natural "summary" entity
    # for every fact extracted from that chunk. We materialise it as an entity
    # node and connect every relation from that chunk to it via a regular
    # hyperedge.  This recovers back-references like "the region" or "the
    # construction" that the chunk-to-fact LLM call occasionally drops:
    # even when "(Southeast Asia)" is lost from a fact's entity list, the
    # title→fact hyperedge keeps the fact reachable from the Southeast Asia
    # entity during iterative_search.
    unique_titles = sorted({ev.get("source_title", "").strip()
                            for ev in events if ev.get("source_title", "").strip()})
    # Embed titles that do not yet exist as entities
    new_titles = [t for t in unique_titles
                  if not G.has_node(f"ent::{t.lower()}")]
    if new_titles:
        logger.info("Embedding %d unseen chunk titles as entity nodes", len(new_titles))
        title_name_vecs = encode_texts(new_titles)
        # name+def for a title uses the title text itself as the definition
        # (it is its own most concise summary).
        title_def_vecs  = encode_texts([f"{t}: {t}" for t in new_titles])
        for t, nv, dv in zip(new_titles, title_name_vecs, title_def_vecs):
            G.add_node(
                f"ent::{t.lower()}",
                type           = "entity",
                canonical      = t.lower(),
                display        = t,
                definition     = t,                                # title acts as its own definition
                embedding      = dv.astype(np.float32),            # name+def embedding
                name_embedding = nv.astype(np.float32),            # name-only embedding
                is_title_entity= True,
            )

    # relation nodes + hyperedge connections
    # Store title-prefixed text so BM25 + context output also see the chunk title
    # (e.g. "1894-95 FA Cup" never appears in the body text but is the title).
    for i, ev in enumerate(events):
        rel_id = f"rel::{i}"
        title  = ev.get("source_title", "").strip()
        text   = f"({title}) {ev['relation']}" if title else ev["relation"]
        G.add_node(
            rel_id,
            type         = "relation",
            text         = text,
            embedding    = rel_vecs[i].astype(np.float32),
            source_title = ev["source_title"],
            source_text  = ev["source_text"],
        )
        for e in ev["entities"]:
            ent_id = f"ent::{e['name'].lower()}"
            if G.has_node(ent_id):
                G.add_edge(rel_id, ent_id, edge_type="hyperedge")
        # Always connect the relation to its chunk-title entity (recovers
        # back-references the LLM extractor may have dropped).
        if title:
            title_id = f"ent::{title.lower()}"
            if G.has_node(title_id) and not G.has_edge(rel_id, title_id):
                G.add_edge(rel_id, title_id, edge_type="hyperedge",
                           is_title_edge=True)

    n_ent = sum(1 for _, d in G.nodes(data=True) if d["type"] == "entity")
    n_rel = sum(1 for _, d in G.nodes(data=True) if d["type"] == "relation")
    logger.info("Pre-merge: %d entity nodes, %d relation nodes, %d edges",
                n_ent, n_rel, G.number_of_edges())

    # ── 3e. entity merging — average of name-only sim and name:def sim ───────
    # Dual-similarity provides "double insurance": names disambiguate when
    # definitions are too similar (multiple entities sharing a person mention),
    # and definitions disambiguate when names are too similar (Apple Inc. vs Apple fruit).
    n        = len(ent_keys)
    sim_name = ent_name_vecs    @ ent_name_vecs.T
    sim_def  = ent_nameDef_vecs @ ent_nameDef_vecs.T
    sim      = (sim_name + sim_def) / 2.0

    merge_map: dict[str, str]               = {}   # child_key → parent_key
    synonym_pairs: list[tuple[str, str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s < MERGE_THRESHOLD:
                continue
            ki, kj = ent_keys[i], ent_keys[j]
            if ent_display[ki].lower() == ent_display[kj].lower():
                # same surface form, different canonical keys → merge kj into ki
                merge_map[kj] = ki
            else:
                # different names but semantically close → synonym edge
                synonym_pairs.append((ki, kj, s))

    logger.info("Merging %d same-name duplicate entity nodes", len(merge_map))
    for child, parent in merge_map.items():
        cid, pid = f"ent::{child}", f"ent::{parent}"
        if not (G.has_node(cid) and G.has_node(pid)):
            continue
        for nbr, edgedict in list(G.adj[cid].items()):
            for _, edata in edgedict.items():
                G.add_edge(pid, nbr, **edata)
        G.remove_node(cid)

    logger.info("Adding %d synonym edges", len(synonym_pairs))
    for ki, kj, s in synonym_pairs:
        ni, nj = f"ent::{ki}", f"ent::{kj}"
        if G.has_node(ni) and G.has_node(nj):
            G.add_edge(ni, nj, edge_type="synonym", similarity=round(s, 4))

    n_ent = sum(1 for _, d in G.nodes(data=True) if d["type"] == "entity")
    n_rel = sum(1 for _, d in G.nodes(data=True) if d["type"] == "relation")
    logger.info("Post-merge: %d entity nodes, %d relation nodes, %d edges",
                n_ent, n_rel, G.number_of_edges())
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 4. Save
# ══════════════════════════════════════════════════════════════════════════════

def save_kg(G: nx.MultiGraph, events: list[dict], build_time_s: float | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "kg.pkl", "wb") as f:
        pickle.dump({"graph": G, "events": events}, f)

    n_ent = sum(1 for _, d in G.nodes(data=True) if d["type"] == "entity")
    n_rel = sum(1 for _, d in G.nodes(data=True) if d["type"] == "relation")
    n_hyp = sum(1 for *_, d in G.edges(data=True) if d.get("edge_type") == "hyperedge")
    n_syn = sum(1 for *_, d in G.edges(data=True) if d.get("edge_type") == "synonym")

    # Arity statistics: how many entities does each relation hyperedge
    # connect? Reported in Appendix F.
    arities = []
    for n, d in G.nodes(data=True):
        if d["type"] == "relation":
            deg = sum(1 for _, _, ed in G.edges(n, data=True)
                      if ed.get("edge_type") == "hyperedge")
            if deg:
                arities.append(deg)
    max_arity = max(arities) if arities else 0
    mean_arity = (sum(arities) / len(arities)) if arities else 0.0
    nary_relations = sum(1 for a in arities if a >= 3)

    stats = {
        "source_chunks":   208,
        "total_events":    len(events),
        "entity_nodes":    n_ent,
        "relation_nodes":  n_rel,
        "hyperedges":      n_hyp,
        "synonym_edges":   n_syn,
        "total_edges":     G.number_of_edges(),
        "max_arity":       max_arity,
        "mean_arity":      round(mean_arity, 3),
        "nary_relations":  nary_relations,
        "nary_fraction":   round(nary_relations / max(n_rel, 1), 4),
        "build_time_s":    round(build_time_s, 1) if build_time_s is not None else None,
        "build_time_min":  round(build_time_s / 60.0, 2) if build_time_s is not None else None,
    }
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also emit kg_stats.json — the orchestrator looks for this filename
    # to populate Table 4 (Efficiency) and Appendix F.
    (OUT_DIR / "kg_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("KG saved → %s\n%s", OUT_DIR, json.dumps(stats, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspection", type=str, default=None,
                    help="Path to llm_inspection.json (overrides default)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory for kg.pkl / stats.json")
    args = ap.parse_args()

    global INSPECTION_JSON, OUT_DIR
    if args.inspection:
        INSPECTION_JSON = Path(args.inspection)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    setup_logging()
    import time as _time
    _t0 = _time.time()
    events = load_events()
    if not events:
        logger.error("No events loaded — check %s", INSPECTION_JSON)
        return
    G = build_hypergraph(events)
    _elapsed = _time.time() - _t0
    save_kg(G, events, build_time_s=_elapsed)


if __name__ == "__main__":
    main()
