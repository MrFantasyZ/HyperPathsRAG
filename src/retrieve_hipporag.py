"""
HippoRAG-style retrieval baseline on the same KG.

Skips query decomposition entirely. The pipeline:

  1. Anchor entity selection — encode the query, rank all KG entities by
     averaged (name + definition) cosine sim, keep the top-N as anchors.
     (HippoRAG 2 uses NER; we substitute with BGE-only ranking since the
     same KG is already encoded.)

  2. Personalized PageRank over the hypergraph — entity & relation nodes
     are both vertices; hyperedges (entity↔relation) and synonym edges
     (entity↔entity) are both relayed. Personalization vector concentrates
     on the anchor entities; PPR propagates importance outward through
     graph topology.

  3. Top-K relation emission — sort relation nodes by PPR score, emit
     their .text values concatenated as context.

This is a pure-graph baseline used to test the hypothesis "is the KG
itself the bottleneck?" — if HippoRAG also fails, the KG quality is
suspect; if HippoRAG succeeds where anchor-driven fails, the
decomposition / anchor-driven machinery is the regression.
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np
import networkx as nx

# retrieve.py lives next to this file
sys.path.insert(0, str(Path(__file__).parent))
from retrieve import (
    KGIndex, encode_texts, normalize_for_embedding,
    load_questions, ROOT,
)
import retrieve as _retrieve_mod


logger = logging.getLogger(__name__)

# ── Hyperparameters ───────────────────────────────────────────────────────────
TOP_ANCHORS       = 5        # anchor entities seeded for PPR
PPR_ALPHA         = 0.5      # damping factor (HippoRAG default range 0.3-0.5)
TOP_K_RELATIONS   = 20       # relations emitted as context
ANCHOR_SIM_FLOOR  = 0.5      # ignore anchor candidates below this avg sim


def find_anchors(kg: KGIndex, query: str, top_k: int = TOP_ANCHORS,
                 sim_floor: float = ANCHOR_SIM_FLOOR) -> list[tuple[str, float]]:
    """Encode the whole question and pick the top-K most-similar KG entities
    by averaged (name + def) cosine sim. Returns [(ent_id, sim), ...]."""
    q_vec = encode_texts([normalize_for_embedding(query)])[0]
    sim_name = kg.entity_name_emb @ q_vec
    sim_def  = kg.entity_emb      @ q_vec
    sims = (sim_name + sim_def) / 2.0
    # Take top-K above the floor
    order = np.argsort(-sims)
    out: list[tuple[str, float]] = []
    for i in order[: top_k * 4]:    # over-fetch so floor doesn't shrink result
        s = float(sims[i])
        if s < sim_floor:
            break
        out.append((kg.entity_ids[int(i)], s))
        if len(out) >= top_k:
            break
    if not out:
        # If everything is below the floor, take the top-1 anyway —
        # PPR needs a personalization seed.
        out.append((kg.entity_ids[int(order[0])], float(sims[order[0]])))
    return out


def run_ppr(kg: KGIndex, anchors: list[tuple[str, float]],
            alpha: float = PPR_ALPHA) -> dict[str, float]:
    """Personalised PageRank over the KG.

    Personalization weights are proportional to each anchor's sim score —
    a strong anchor (high BGE sim with the query) seeds more probability
    mass than a weak one. Edge weights default to 1.0 for hyperedges and
    the stored similarity for synonym edges, so synonyms relay nearly
    full mass between equivalent entities.
    """
    G = kg.G
    personalization: dict[str, float] = {}
    for ent_id, sim in anchors:
        if G.has_node(ent_id):
            personalization[ent_id] = max(sim, 1e-3)
    if not personalization:
        return {}
    total = sum(personalization.values())
    personalization = {n: w / total for n, w in personalization.items()}

    # NetworkX PageRank wants edge attribute 'weight'. Build a weighted
    # simple graph collapsing the MultiGraph (sum weights of parallel edges).
    G_simple = nx.Graph()
    G_simple.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        w = float(data.get("similarity", 1.0)) if data.get("edge_type") == "synonym" else 1.0
        if G_simple.has_edge(u, v):
            G_simple[u][v]["weight"] = float(G_simple[u][v]["weight"]) + w
        else:
            G_simple.add_edge(u, v, weight=w)

    try:
        ppr = nx.pagerank(G_simple, alpha=alpha,
                          personalization=personalization,
                          weight="weight",
                          max_iter=200, tol=1e-6)
    except Exception as e:
        logger.warning("PPR failed: %s", e)
        return {}
    return ppr


def retrieve_one(kg: KGIndex, question: str) -> dict:
    t0 = time.time()
    anchors = find_anchors(kg, question)
    anchor_disp = [(kg.G.nodes[a].get("display", "?"), round(s, 3))
                    for a, s in anchors]
    logger.info("Anchors (top-%d): %s", len(anchors), anchor_disp)

    ppr = run_ppr(kg, anchors)
    if not ppr:
        return {
            "question": question, "anchors": anchor_disp,
            "context": "", "n_paths": 0, "fallback": "ppr_empty",
            "elapsed": round(time.time() - t0, 1),
        }

    rel_scores = [(n, s) for n, s in ppr.items() if n.startswith("rel::")]
    rel_scores.sort(key=lambda x: -x[1])
    top_rels = rel_scores[: TOP_K_RELATIONS]
    parts: list[str] = []
    for rid, _score in top_rels:
        txt = kg.G.nodes[rid].get("text", "")
        if txt:
            parts.append(txt)
    context = "\n".join(parts)

    return {
        "question":   question,
        "anchors":    anchor_disp,
        "n_anchors":  len(anchors),
        "n_paths":    1,
        "fallback":   "no",
        "context":    context,
        "elapsed":    round(time.time() - t0, 1),
        "top_rel_scores": [(r, round(s, 4)) for r, s in top_rels[:5]],
    }


def main():
    global TOP_K_RELATIONS  # noqa: PLW0603
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--kg",   type=str, default=None)
    ap.add_argument("--n",    type=int, default=-1)
    ap.add_argument("--question-idx", type=int, default=None)
    ap.add_argument("--out",  type=str, default="retrieval_hipporag.json")
    ap.add_argument("--ablation", type=str, default="hipporag")
    ap.add_argument("--top-k-relations", type=int, default=TOP_K_RELATIONS)
    args = ap.parse_args()
    TOP_K_RELATIONS = args.top_k_relations
    if args.data:
        _retrieve_mod.DATA_FILE = Path(args.data)
    if args.kg:
        _retrieve_mod.KG_PKL = Path(args.kg)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)],
                        force=True)

    kg = KGIndex(_retrieve_mod.KG_PKL)
    questions = load_questions(args.n)
    if args.question_idx is not None:
        questions = [questions[args.question_idx]]

    results = []
    for i, q in enumerate(questions):
        logger.info("=" * 80)
        logger.info("[%d/%d] qid=%s", i + 1, len(questions), q.get("id", "?"))
        t0 = time.time()
        try:
            res = retrieve_one(kg, q["question"])
        except Exception as e:
            logger.exception("Retrieval crashed: %s", e)
            res = {"question": q["question"], "error": str(e)}
        res["expected_answer"] = q.get("answer", "")
        res["ablation"]        = args.ablation
        res["retrieve_total_ms"] = round((time.time() - t0) * 1000, 1)
        results.append(res)
        logger.info("Context preview:\n%s", (res.get("context") or "")[:300])

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
