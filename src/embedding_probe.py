"""
Entity Embedding Strategy Probe — Appendix B.

Compares three strategies for entity equality / synonym detection on a
manually-curated probe set:

  1. name-only       : cos(emb(n_1), emb(n_2))
  2. name+definition : cos(emb("n_1: d_1"), emb("n_2: d_2"))
  3. hybrid (ours)   : 0.5 · [emb(n) + emb("n:d")], merge requires the
                       conjunction (cos >= tau AND exact name match)

Reports per-strategy synonym recall, false-merge rate, and F1 on a
2-class probe (positive = same-entity-different-form,
negative = same-name-different-entity).

Probe pairs live in `data/embedding_probe.json` (auto-created with a
default seed list if missing).

Usage:
  python src/embedding_probe.py --kg results/kg_stats/hotpot/kg.pkl \
                                --out results/appendix_B_embedding/hotpot/embedding_probe.json
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAU_DEFAULT = 0.85

logger = logging.getLogger(__name__)


DEFAULT_PROBE = {
    "positive": [
        # (name_a, def_a, name_b, def_b)  -- same real entity, different forms
        ["Salvador Dali", "Spanish surrealist painter",
         "Dali", "Catalan surrealist artist"],
        ["USSR", "Cold-war communist superpower",
         "Soviet Union", "Eurasian socialist federation"],
        ["Mao Zedong", "founding chairman of the PRC",
         "Chairman Mao", "leader of communist China"],
        ["NYC", "largest US city",
         "New York City", "New York metropolitan area"],
        ["JFK", "35th US president",
         "John F. Kennedy", "American Democratic president"],
        ["FDR", "32nd US president",
         "Franklin D. Roosevelt", "WWII-era US president"],
        ["UK", "European island nation",
         "United Kingdom", "constitutional monarchy in Europe"],
        ["UN", "international organisation",
         "United Nations", "intergovernmental peace organisation"]
    ],
    "negative": [
        # same surface form, different entities
        ["Apple", "temperate-climate fruit",
         "Apple", "Californian electronics company"],
        ["Jordan", "Middle Eastern monarchy",
         "Jordan", "American basketball player"],
        ["Mercury", "innermost planet of the solar system",
         "Mercury", "metallic element"],
        ["Java", "Indonesian island",
         "Java", "Sun-Microsystems programming language"],
        ["Python", "tropical reptile",
         "Python", "popular programming language"],
        ["Amazon", "South American rainforest",
         "Amazon", "American e-commerce company"],
        ["Saturn", "ringed gas-giant planet",
         "Saturn", "Roman god of agriculture"],
        ["Madonna", "American pop singer",
         "Madonna", "Renaissance religious icon"]
    ],
}


def get_embedder():
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    mdl = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE).eval()
    return tok, mdl


@torch.no_grad()
def encode(tok, mdl, texts: Sequence[str]) -> np.ndarray:
    """Mean-pool encoding, ℓ2-normalised."""
    enc = tok(list(texts), padding=True, truncation=True, max_length=128,
              return_tensors="pt").to(DEVICE)
    out = mdl(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled.cpu().numpy()


def evaluate_strategy(strategy_scores: dict[str, float],
                      labels: dict[str, int],
                      tau: float) -> dict:
    """Given pair_id -> similarity and pair_id -> label (1=synonym, 0=distinct),
    compute recall / false-merge / F1 at threshold tau."""
    pos = [pid for pid, l in labels.items() if l == 1]
    neg = [pid for pid, l in labels.items() if l == 0]
    tp = sum(1 for pid in pos if strategy_scores[pid] >= tau)
    fn = len(pos) - tp
    fp = sum(1 for pid in neg if strategy_scores[pid] >= tau)
    tn = len(neg) - fp
    syn_recall = tp / max(len(pos), 1)
    false_merge = fp / max(len(neg), 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * syn_recall / max(prec + syn_recall, 1e-9)
    return {
        "tau":          tau,
        "syn_recall":   round(syn_recall, 4),
        "false_merge":  round(false_merge, 4),
        "precision":    round(prec, 4),
        "f1":           round(f1, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=str, default=None,
                    help="Path to probe JSON (default: data/embedding_probe.json)")
    ap.add_argument("--out", type=str, required=True,
                    help="Output JSON path")
    ap.add_argument("--tau", type=float, default=TAU_DEFAULT)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)],
                        force=True)

    probe_path = Path(args.probe) if args.probe else (ROOT / "data" / "embedding_probe.json")
    if not probe_path.exists():
        logger.info("Probe file missing — writing default seed to %s", probe_path)
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path.write_text(json.dumps(DEFAULT_PROBE, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))

    pairs: list[tuple[str, str, str, str, int]] = []
    for a, da, b, db in probe["positive"]:
        pairs.append((a, da, b, db, 1))
    for a, da, b, db in probe["negative"]:
        pairs.append((a, da, b, db, 0))

    logger.info("Loaded %d pairs (%d positive, %d negative)",
                len(pairs), len(probe["positive"]), len(probe["negative"]))

    tok, mdl = get_embedder()

    # Build text variants in one pass
    name_texts = []
    namedef_texts = []
    for a, da, b, db, _ in pairs:
        name_texts.extend([a, b])
        namedef_texts.extend([f"{a}: {da}", f"{b}: {db}"])

    emb_name = encode(tok, mdl, name_texts)
    emb_namedef = encode(tok, mdl, namedef_texts)

    # Strategy 3 (hybrid): average of name-only and name+def embeddings
    emb_hybrid = (emb_name + emb_namedef) / 2.0
    # Re-normalise after averaging
    emb_hybrid = emb_hybrid / np.linalg.norm(emb_hybrid, axis=-1, keepdims=True).clip(min=1e-9)

    sim_name, sim_namedef, sim_hybrid = {}, {}, {}
    labels: dict[str, int] = {}
    pair_records = []
    for i, (a, da, b, db, lab) in enumerate(pairs):
        pid = f"P{i:03d}"
        labels[pid] = lab
        ia, ib = 2 * i, 2 * i + 1
        sim_name[pid]    = float(emb_name[ia]    @ emb_name[ib])
        sim_namedef[pid] = float(emb_namedef[ia] @ emb_namedef[ib])
        # Hybrid strategy: similarity is over averaged embedding, MERGE
        # also requires lowercase name equality (the paper's rule). For
        # synonym-edge detection here we report only the embedding sim,
        # since the paper rules same-name pairs as merge-eligible and
        # different-name pairs as synonym-edge-eligible — both share the
        # τ-gated embedding test.
        sim_hybrid[pid]  = float(emb_hybrid[ia]  @ emb_hybrid[ib])
        pair_records.append({
            "pid": pid, "name_a": a, "name_b": b, "label": lab,
            "sim_name": round(sim_name[pid], 4),
            "sim_namedef": round(sim_namedef[pid], 4),
            "sim_hybrid": round(sim_hybrid[pid], 4),
            "exact_name_match": a.lower() == b.lower(),
        })

    results = {
        "tau": args.tau,
        "n_positive": len(probe["positive"]),
        "n_negative": len(probe["negative"]),
        "by_strategy": {
            "name_only":     evaluate_strategy(sim_name,    labels, args.tau),
            "name_plus_def": evaluate_strategy(sim_namedef, labels, args.tau),
            "hybrid_ours":   evaluate_strategy(sim_hybrid,  labels, args.tau),
        },
        "pairs": pair_records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved embedding probe → %s", out_path)
    for k, v in results["by_strategy"].items():
        logger.info("  %s: syn_rec=%.3f false_merge=%.3f f1=%.3f",
                    k, v["syn_recall"], v["false_merge"], v["f1"])


if __name__ == "__main__":
    main()
