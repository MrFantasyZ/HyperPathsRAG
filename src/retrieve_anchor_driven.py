"""
HyperPathsRAG Retrieval Pipeline — Anchor-Driven Variant.

Differs from retrieve.py in Step 4 only:

  Old: each virtual placeholder is matched globally against KG entities
       using its abstract attribute-name embedding (e.g. "record label A").
       This pulls in any high-density "record label / records" entity
       regardless of whether it is actually connected to the anchor
       sub-question's locked relation. Concrete answer entities like
       "Sony Music Entertainment" lose to generic "Warner Records" /
       "Brunswick Records" because their abstract-placeholder cosine is
       lower.

  New: anchor-conditioned local matching, event-driven on sub-question
       readiness.
       1. Initialise one root variant per anchor sub-question (its real
          entities are anchored to KE1 ent-ids + their 1-hop synonyms).
       2. While the queue is non-empty:
            - Group queued variants by their starting anchor set (variants
              from the SAME starting point contend over the same 1-hop
              relation candidates).
            - For each group, the candidate relations = the union of
              kg.relations_of(anchor_ent) for anchor_ent in the group's
              starting set, minus globally locked relations.
            - Each candidate relation is scored against each variant's
              parent sub-question. It is assigned to the variant with the
              highest sim (contention). Per-variant adaptive top-K
              selects the kept relations (REL_K_MIN..REL_K_MAX with
              gap-based cutoff).
            - For each variant, every still-unbound virtual placeholder
              in its sub-question is matched only against the non-anchor
              entities of its newly-locked relations (NOT the whole KG).
              Adaptive gap-based top-N selects N concrete entities.
            - For each downstream skeleton sub-question that contains
              one of the newly-bound virtual keys, spawn one child
              variant per cartesian combination of the bindings. The
              child variant's anchor set = synonym-closure of its
              real entities + the just-bound virtuals' KG entities.
            - Child variants with no anchor set OR no locked relations
              die immediately — failed reasoning chains do not propagate.
       3. Lineage tree variants → root-to-leaf paths → score → context.

Other steps (decomposition, skeleton, anchor lookup, path scoring,
context organisation) reuse retrieve.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Re-use everything that has not changed (same module, same constants).
# retrieve.py lives next to this file so absolute import on PYTHONPATH=src works.
from retrieve import (
    # config / constants
    DATA_FILE, KG_PKL, EMBED_MODEL,
    HYBRID_BM25_BOOST, VIRTUAL_MATCH_THR, REL_GAP_THR,
    REL_K_MIN, REL_K_MAX, TOP_K_PATHS,
    REL_K_MIN_CONTEXT, REL_K_MAX_CONTEXT, REL_GAP_THR_CONTEXT,
    MAX_BINDINGS_PER_VIRTUAL, ATTR_FAMILIES,
    MIN_THR_STEP4,
    BM25_TOPN, EPS,
    # helpers
    encode_texts, normalize_for_embedding, bm25_tokenize,
    decompose_query, build_skeleton_dag, compute_levels,
    composed_retrieve, composed_retrieve_relations,
    compute_bm25_relation_scores,
    select_top_relations_with,
    compute_qid_chains, select_top_paths_per_chain,
    load_questions,
    # classes
    KGIndex, SubQuestion, EntityInfo, SolutionEntry,
)


logger = logging.getLogger(__name__)


# ── Anchor-driven specific knobs ─────────────────────────────────────────────
# How many KG ents may be selected per virtual when expanding a variant.
# Gap-based adaptive cutoff inside [1, MAX_VBIND_PER_VARIANT].
MAX_VBIND_PER_VARIANT = MAX_BINDINGS_PER_VIRTUAL   # 5
VBIND_GAP_THR         = REL_GAP_THR                # 0.05 — same shape as relation cutoff
# Variant explosion safety net (4-hop with N=3 → 81 leaves; we leave headroom).
MAX_TOTAL_VARIANTS    = 600
# Contention tied-winner band. A candidate relation can be locked by EVERY
# contestant within this ε of the highest-sim contestant — designed for
# multi-fact relations whose text expresses several semantic facts at once
# (e.g. rel::6322 "EMI is owned by UMG and based out of Santa Monica" reads
# as both "X is part of Y" for q0 and "Y has HQ at Z" for q1 — sims 0.592
# vs 0.551, gap 0.041 < ε so both ancestors win).
TIED_WINNER_EPS       = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# AnchorVariant — the unit of work in the event-driven beam
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnchorVariant:
    """One concrete instantiation of a sub-question with explicit anchor pool.

    A root variant comes from an anchor sub-question; its anchor_ents are the
    KG ent-ids of the sub-question's real entities + 1-hop synonyms. A child
    variant is spawned when its parent binds the virtual key(s) that appear
    in this variant's sub-question; the child's anchor_ents are the
    synonym-closure of the just-bound entities (plus any real entities of
    this variant's sub-question, if any).

    binding_chain carries every ancestor binding so a leaf variant knows the
    full reasoning chain that led to it (used for context lineage).

    bound_so_far tracks which entities have ALREADY been used to spawn
    children for each virtual key. New rounds that re-bind the same key
    only spawn children for NEW entities, preventing duplicate variants
    while the ancestor-contention loop keeps reactivating the variant.
    """
    var_id:         int
    qid:            int                           # parent skeleton qid
    parent_var_id:  int | None
    binding_chain:  dict[str, str]                # virtual_key → ent_id (cumulative)
    anchor_ents:    frozenset[str]                # where this variant searches from
    locked_rels:    list[SolutionEntry] = field(default_factory=list)
    new_bindings:   dict[str, list[str]] = field(default_factory=dict)  # virtual_key → ents bound by THIS variant
    bound_so_far:   dict[str, set[str]] = field(default_factory=dict)   # virtual_key → all ents ever spawned
    frozen:         bool = False                  # once True, no longer competes in any contention

    @property
    def bound_entities(self) -> frozenset[str]:
        ents = set(self.anchor_ents)
        ents.update(self.binding_chain.values())
        return frozenset(ents)


# ══════════════════════════════════════════════════════════════════════════════
# Anchor entity lookup — returns a per-key dict instead of a flat set
# ══════════════════════════════════════════════════════════════════════════════

def locate_anchor_entities_by_key(
    kg: KGIndex, anchor_subqs: list[SubQuestion],
) -> dict[str, set[str]]:
    """For each anchor sub-question's real entity, locate it in the KG
    via Composed Retriever and expand by 1-hop synonyms. Returns a dict
    keyed by the entity's canonical key (the question-side name)."""
    bindings: dict[str, set[str]] = defaultdict(set)
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
            bindings[ent.key].add(best)
            for syn in kg.synonyms_of(best):
                bindings[ent.key].add(syn)
            logger.info("  Anchor '%s' → %s (+%d synonyms)",
                        query, kg.G.nodes[best]["display"], len(kg.synonyms_of(best)))
    return dict(bindings)


def collect_virtuals(subqs: list[SubQuestion]) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """Same as retrieve.collect_virtuals + also returns virtual_key -> attribute."""
    seen: dict[str, int] = {}
    keys: list[str] = []
    attr_of: dict[str, str] = {}
    for sq in subqs:
        for ent in sq.entities:
            if ent.virtual and ent.key and ent.key not in seen:
                seen[ent.key] = len(keys)
                keys.append(ent.key)
                attr_of[ent.key] = ent.attribute or ""
    return keys, seen, attr_of


# ══════════════════════════════════════════════════════════════════════════════
# Variant expansion helpers
# ══════════════════════════════════════════════════════════════════════════════

def _adaptive_topn_by_sim(sims: np.ndarray, max_n: int, gap_thr: float, abs_min: float) -> list[int]:
    """Return indices of the top entries by sim that satisfy:
      - sim >= abs_min
      - within max_n
      - top1-anchored gap: every kept entry must be within `gap_thr` of #1.
    This is stricter (and more correct) than the adjacent-gap variant —
    a relation #5 that survives a chain of small adjacent gaps but ends
    up >gap_thr below the leader is rejected, since it likely reflects
    a noisy long tail rather than a genuine plateau.
    """
    if sims.size == 0:
        return []
    order = np.argsort(-sims)
    top1 = float(sims[order[0]])
    if top1 < abs_min:
        return []
    kept = [int(order[0])]
    for k in range(1, min(max_n, len(order))):
        cur = float(sims[order[k]])
        if cur < abs_min:
            break
        if top1 - cur > gap_thr:
            break
        kept.append(int(order[k]))
    return kept


def select_top_relations_top1_anchored(
    entries: list[SolutionEntry],
    k_min: int,
    k_max: int,
    gap_thr: float,
) -> list[SolutionEntry]:
    """Top1-anchored gap-based adaptive top-K cutoff (replaces the adjacent-
    gap variant in retrieve.select_top_relations_with).

    Always keeps top-K_MIN (floor). For positions K_MIN..K_MAX-1, keep
    entry iff (top1.sim - entry.sim) <= gap_thr. This is more robust to
    dense noise tails than the adjacent-gap algorithm: a dense plateau
    of low-sim relations no longer pulls the algorithm out to K_MAX
    just because the gaps between consecutive entries are tiny.

    Example (q2 in G1.3): top1=0.746, #2=0.542 (gap 0.204), #3-#10 dense
      adjacent-gap would keep top-5 (4 noise relations beyond gold #1)
      top1-anchored keeps only top-1 (gold), rejecting the noise tail
    """
    if not entries:
        return []
    es = sorted(entries, key=lambda e: -e.sim)
    n = len(es)
    if n <= k_min:
        return es
    top1 = es[0].sim
    kept = list(es[:k_min])
    for i in range(k_min, min(k_max, n)):
        if top1 - es[i].sim > gap_thr:
            break
        kept.append(es[i])
    return kept


def init_root_variants(
    kg: KGIndex, subqs: list[SubQuestion], anchor_bindings: dict[str, set[str]],
) -> list[AnchorVariant]:
    """Create one root variant per anchor sub-question.

    If a sub-question carries multiple real entities (n-ary anchor), all
    of them join the same anchor pool — the variant has a single "starting
    set" comprising the synonym-closure of every real entity. This matches
    round_relation_search's existing behaviour in retrieve.py.
    """
    roots: list[AnchorVariant] = []
    next_id = 0
    for sq in subqs:
        real_keys = [e.key for e in sq.real_entities]
        if not real_keys:
            continue
        anchor_ents: set[str] = set()
        chain: dict[str, str] = {}
        for k in real_keys:
            opts = anchor_bindings.get(k, set())
            anchor_ents |= opts
            for aid in opts:
                anchor_ents |= set(kg.synonyms_of(aid))
            # Record an arbitrary representative for the binding_chain; the
            # variant uses this for lineage tracking but every synonym is in
            # anchor_ents already.
            if opts:
                chain[k] = next(iter(sorted(opts)))
        if not anchor_ents:
            continue
        roots.append(AnchorVariant(
            var_id=next_id, qid=sq.qid, parent_var_id=None,
            binding_chain=chain, anchor_ents=frozenset(anchor_ents),
        ))
        next_id += 1
    return roots


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Anchor-driven iterative search
# ══════════════════════════════════════════════════════════════════════════════

def _ancestry_of(v: AnchorVariant, by_id: dict[int, AnchorVariant],
                 skip_frozen: bool = True) -> list[AnchorVariant]:
    """Return root → ... → v (inclusive). Used so a descendant's relation
    candidate pool is open for contention by every ancestor sub-question
    on the path — relations expressing 'X is part of Y' (q0 semantics) can
    then be claimed by the ancestor even when v's anchor sits one hop
    deeper than the ancestor's first relation chain.

    skip_frozen=True (default) filters out ancestors that have been
    permanently disqualified by the freeze mechanism (an ancestor that
    fails to win any relation in a full round is considered to have
    exhausted its retrieval reach and is locked out of further
    contention)."""
    chain: list[AnchorVariant] = []
    cur: AnchorVariant | None = v
    while cur is not None:
        if (not skip_frozen) or (not cur.frozen):
            chain.append(cur)
        cur = by_id.get(cur.parent_var_id) if cur.parent_var_id is not None else None
    chain.reverse()
    return chain


def _expand_virtuals_and_spawn(
    v: AnchorVariant,
    subqs: list[SubQuestion],
    kg: KGIndex,
    anchor_bindings: dict[str, set[str]],
    virtual_keys: list[str],
    virtual_idx_of: dict[str, int],
    virtual_attr_of: dict[str, str],
    virtual_vecs: np.ndarray,
    seen_variants: set[tuple[int, frozenset]],
    next_var_id_ref: list[int],     # mutable single-element holder
    new_children_out: list[AnchorVariant],
) -> None:
    """Run the virtual-binding expansion + child-spawning step for ONE
    variant. Idempotent across rounds: bound_so_far tracks which ent_ids
    each virtual key has already spawned children for; only freshly-seen
    ent_ids generate new children.
    """
    if not v.locked_rels:
        return
    sq = subqs[v.qid]
    unbound_virtuals = [e for e in sq.virtual_entities
                        if e.key not in v.binding_chain]
    if not unbound_virtuals:
        return

    # Non-anchor neighbours of v.locked_rels = candidate pool for ALL of v's
    # unbound virtuals. Recomputed every round because v.locked_rels grows.
    cand_ents_global: set[str] = set()
    for entry in v.locked_rels:
        for nbr in kg.entities_of_relation(entry.rel_id):
            if nbr not in v.anchor_ents:
                cand_ents_global.add(nbr)
    if not cand_ents_global:
        return

    fresh_bindings: dict[str, list[str]] = {}
    for ve in unbound_virtuals:
        v_attr = virtual_attr_of.get(ve.key, "")
        compatible = ATTR_FAMILIES.get(v_attr.strip().lower())
        if compatible:
            cand_ents = [
                eid for eid in cand_ents_global
                if (kg.G.nodes[eid].get("attribute") in compatible)
                or (not kg.G.nodes[eid].get("attribute"))
            ]
        else:
            cand_ents = sorted(cand_ents_global)
        if not cand_ents:
            continue
        v_idx_for_key = virtual_idx_of[ve.key]
        v_vec = virtual_vecs[v_idx_for_key]
        cand_ent_list = sorted(cand_ents)
        cand_idx_arr  = np.array([kg.ent_idx_of[e] for e in cand_ent_list])
        sim_def  = kg.entity_emb[cand_idx_arr]      @ v_vec
        sim_name = kg.entity_name_emb[cand_idx_arr] @ v_vec
        sims     = (sim_def + sim_name) / 2.0
        kept_idx = _adaptive_topn_by_sim(
            sims, MAX_VBIND_PER_VARIANT, VBIND_GAP_THR, VIRTUAL_MATCH_THR)
        if not kept_idx:
            continue
        solved = [cand_ent_list[i] for i in kept_idx]
        # Subtract ent_ids that already spawned children for this key
        already = v.bound_so_far.setdefault(ve.key, set())
        new_for_this_round = [e for e in solved if e not in already]
        if new_for_this_round:
            fresh_bindings[ve.key] = new_for_this_round
            already.update(new_for_this_round)
            logger.info(
                "  var %d: bound virtual %r → +%d new (sims=%s)",
                v.var_id, ve.key, len(new_for_this_round),
                [round(float(sims[i]), 3) for i in kept_idx if cand_ent_list[i] in new_for_this_round])
        # Track current-round binding (may include re-confirmed entities)
        v.new_bindings[ve.key] = solved

    if not fresh_bindings:
        return

    # ── spawn children for downstream sub-questions ─────────────────────────
    # For every downstream subq that shares at least one freshly-bound
    # virtual key with v, expand the cartesian product of bindings.
    # When a downstream subq shares multiple keys but only some are fresh
    # this round, the cartesian uses fresh × already-bound to cover the
    # newly-introduced combinations without re-emitting the all-old combo.
    fresh_keys = sorted(fresh_bindings.keys())
    for downstream_sq in subqs:
        if downstream_sq.qid == v.qid:
            continue
        ds_virtual_keys = {e.key for e in downstream_sq.virtual_entities}
        ds_real_keys    = [e.key for e in downstream_sq.real_entities]
        shared = sorted(set(v.new_bindings.keys()) & ds_virtual_keys)
        shared_fresh = [k for k in shared if k in fresh_keys]
        if not shared_fresh:
            continue
        # Build the per-key option lists. Each key contributes either its
        # full current solved set OR only the fresh-this-round subset; we
        # use the union to keep things simple and rely on seen_variants
        # to deduplicate identical (qid, binding_chain) variants.
        option_lists = [v.new_bindings[k] for k in shared]
        for combo in itertools.product(*option_lists):
            # Require that at least one component is fresh this round
            # (otherwise we would re-spawn an already-emitted child).
            if not any(combo[i] in v.bound_so_far.get(shared[i], set())
                       and combo[i] in fresh_bindings.get(shared[i], [])
                       for i in range(len(shared))):
                # Re-check using fresh_bindings only
                if not any(combo[i] in fresh_bindings.get(shared[i], [])
                           for i in range(len(shared))):
                    continue
            new_chain = dict(v.binding_chain)
            for k, eid in zip(shared, combo):
                new_chain[k] = eid
            child_anchors: set[str] = set()
            for k in ds_real_keys:
                opts = anchor_bindings.get(k, set())
                child_anchors |= opts
                for aid in opts:
                    child_anchors |= set(kg.synonyms_of(aid))
            for e in downstream_sq.virtual_entities:
                if e.key in new_chain:
                    eid = new_chain[e.key]
                    if eid:
                        child_anchors.add(eid)
                        child_anchors |= set(kg.synonyms_of(eid))
            if not child_anchors:
                continue
            # Dedup: same (qid, binding_chain) already created?
            key = (downstream_sq.qid, frozenset(new_chain.items()))
            if key in seen_variants:
                continue
            seen_variants.add(key)
            child = AnchorVariant(
                var_id=next_var_id_ref[0], qid=downstream_sq.qid,
                parent_var_id=v.var_id,
                binding_chain=new_chain,
                anchor_ents=frozenset(child_anchors),
            )
            next_var_id_ref[0] += 1
            new_children_out.append(child)


def iterative_search_anchor_driven(
    kg: KGIndex,
    subqs: list[SubQuestion],
    anchor_bindings: dict[str, set[str]],
    seed_relations: set[str] | None = None,
) -> list[AnchorVariant]:
    """Event-driven, anchor-conditioned retrieval with ancestor contention.

    Each round processes the current queue of variants. For every variant
    in the queue, the candidate relations (1-hop from its anchor_ents) are
    NOT scored against only its own sub-question — they are scored against
    every sub-question on the ancestor chain from root to this variant,
    and each relation is awarded to the ancestor whose sub-question has
    the highest sim with it. The ancestor's locked_rels grows, the
    ancestor re-runs virtual binding (with seen-ent tracking to avoid
    duplicate child spawning), and any newly-bound ent_ids spawn fresh
    child variants from the ancestor's qid — enabling chains like
    Con-Test → MCA → UMG → Santa Monica even though Con-Test's anchor
    is two hops short of UMG.

    Returns the full list of variants (root + every spawned child).
    """
    seed_relations = seed_relations or set()

    virtual_keys, virtual_idx_of, virtual_attr_of = collect_virtuals(subqs)
    virtual_vecs = encode_texts(
        [normalize_for_embedding(k) for k in virtual_keys]
    ) if virtual_keys else np.zeros((0, 1024), dtype=np.float32)

    subq_vecs = encode_texts([normalize_for_embedding(sq.relation_text) for sq in subqs])
    bm25_per_subq = compute_bm25_relation_scores(kg, subqs)   # (n_rel, n_subq)

    all_variants: list[AnchorVariant] = init_root_variants(kg, subqs, anchor_bindings)
    if not all_variants:
        logger.warning("No anchor-driven root variants — anchor sub-questions had no KE1.")
        return []
    # variant lookup + dedup set keyed by (qid, frozenset(binding_chain.items()))
    by_id: dict[int, AnchorVariant] = {v.var_id: v for v in all_variants}
    seen_variants: set[tuple[int, frozenset]] = set()
    for v in all_variants:
        seen_variants.add((v.qid, frozenset(v.binding_chain.items())))

    queue: list[AnchorVariant] = list(all_variants)
    locked_rels_global: set[str] = set()
    next_var_id_ref: list[int] = [max(v.var_id for v in all_variants) + 1]
    round_idx = 0

    while queue:
        round_idx += 1
        logger.info("Anchor-driven round %d: queue=%d", round_idx, len(queue))

        # ── Group queue by anchor_ents — same-anchor variants compete
        # together in the same contention pool, exactly like the legacy
        # round_relation_search did when multiple subqs shared the same
        # starting entity set.
        groups: dict[frozenset[str], list[AnchorVariant]] = defaultdict(list)
        for v in queue:
            groups[v.anchor_ents].append(v)

        # Tracked per round
        won_this_round:  set[int] = set()
        touched:         dict[int, AnchorVariant] = {}
        queue_ids:       set[int] = {v.var_id for v in queue}
        ancestors_seen:  set[int] = set()        # ancestors (non-queue) that took part this round

        for anchor_ents, group_vars in groups.items():
            # Contestants = filtered ancestries of every group member, deduped.
            # This realises: (a) same-anchor sibling variants contend over
            # the same relations, AND (b) each variant's whole ancestor
            # chain joins the same contention so a relation expressing
            # ancestor semantics can be claimed by the ancestor instead of
            # the descendant.
            contestants: list[AnchorVariant] = []
            seen_ids: set[int] = set()
            for v in group_vars:
                for a in _ancestry_of(v, by_id, skip_frozen=True):
                    if a.var_id in seen_ids:
                        continue
                    seen_ids.add(a.var_id)
                    contestants.append(a)
                    if a.var_id not in queue_ids:
                        ancestors_seen.add(a.var_id)
            if not contestants:
                continue

            # Candidate relations = union of 1-hop from anchor_ents, minus
            # globally-locked. Single-winner contention means a relation
            # already locked elsewhere does not re-enter the pool.
            cand_rels: set[str] = set()
            for ent in anchor_ents:
                cand_rels.update(kg.relations_of(ent))
            cand_rels -= locked_rels_global
            for rid in (seed_relations - locked_rels_global):
                if any(e in anchor_ents for e in kg.entities_of_relation(rid)):
                    cand_rels.add(rid)
            if not cand_rels:
                logger.info("  group anchor=%d ents: 0 candidate relations",
                            len(anchor_ents))
                continue

            cand_list = sorted(cand_rels)
            rel_idx_arr = np.array([kg.rel_idx_of[r] for r in cand_list])
            rel_emb     = kg.relation_emb[rel_idx_arr]

            # Fixed absolute floor only. The previously-tried dynamic
            # threshold max(0.4, mean(locked sims)) was too brittle: with
            # only 1 locked relation it locks in that relation's own sim
            # as the future floor (e.g. var 0's first lock at sim 0.795
            # raises its threshold to 0.795, blocking every legitimate
            # bridging relation in the 0.55-0.65 range from ever entering).
            # The top1-anchored adaptive top-K below is what enforces
            # discrimination instead.
            threshold = VIRTUAL_MATCH_THR  # 0.5

            # Similarity matrix (n_rel, n_contestant)
            cont_subq_vecs = np.vstack([subq_vecs[c.qid] for c in contestants])
            dense_sims = rel_emb @ cont_subq_vecs.T
            bm25_mat   = np.stack([bm25_per_subq[rel_idx_arr, c.qid]
                                   for c in contestants], axis=1)
            sims_mat   = dense_sims + HYBRID_BM25_BOOST * bm25_mat

            # Tied-winner contention: each relation R picks the highest-sim
            # eligible contestant AND every other contestant whose sim sits
            # within TIED_WINNER_EPS of that maximum. The tied band catches
            # multi-fact relations like rel::6322 ("X is part of Y" + "Y has
            # HQ at Z"), where a strict single-winner would award the relation
            # to one ancestor and starve the other.
            assigned: dict[int, list[tuple[str, float]]] = defaultdict(list)
            for r_idx in range(sims_mat.shape[0]):
                row = sims_mat[r_idx]
                eligible_mask = row >= threshold
                if not np.any(eligible_mask):
                    continue
                masked_row = np.where(eligible_mask, row, -np.inf)
                max_sim   = float(masked_row.max())
                tied_mask = masked_row >= (max_sim - TIED_WINNER_EPS)
                for c_idx in np.where(tied_mask)[0]:
                    assigned[int(c_idx)].append(
                        (cand_list[r_idx], float(masked_row[int(c_idx)])))

            # Per-winner top1-anchored adaptive top-K
            for c_idx, rel_list in assigned.items():
                rel_list.sort(key=lambda x: -x[1])
                c_obj = contestants[c_idx]
                entries = [SolutionEntry(rel_id=r, sim=s,
                                         variant_tag=f"v{c_obj.var_id}")
                           for r, s in rel_list]
                kept = select_top_relations_top1_anchored(
                    entries, REL_K_MIN, REL_K_MAX, REL_GAP_THR)
                if not kept:
                    continue
                c_obj.locked_rels.extend(kept)
                locked_rels_global.update(e.rel_id for e in kept)
                won_this_round.add(c_obj.var_id)
                touched[c_obj.var_id] = c_obj
                logger.info(
                    "  group anchor=%d ents: var %d (qid=%d) +%d rels  top-sim=%.3f  floor=%.3f",
                    len(anchor_ents), c_obj.var_id, c_obj.qid,
                    len(kept), kept[0].sim, threshold)

        # NO FREEZE in v6 — diagnostic on 5 questions showed that long
        # multi-hop chains (G1.1 Sony Music → UMG, Katharina Catholic Church
        # → Luther) need ancestors to keep harvesting bridging relations
        # across many rounds. Freezing them after one losing round kills the
        # very mechanism that makes ancestor-contention useful. Noise
        # control is already handled by:
        #   - single-winner contention (one relation → one ancestor)
        #   - top1-anchored adaptive top-K (no dense-tail expansion)
        #   - fixed 0.5 floor (rejects sub-threshold matches)
        #   - tied-winner ε for multi-fact relations

        # ── Binding expansion + spawn children for whoever got new relations
        new_children: list[AnchorVariant] = []
        for v in touched.values():
            _expand_virtuals_and_spawn(
                v, subqs, kg, anchor_bindings,
                virtual_keys, virtual_idx_of, virtual_attr_of, virtual_vecs,
                seen_variants, next_var_id_ref, new_children,
            )

        for c in new_children:
            by_id[c.var_id] = c
        all_variants.extend(new_children)
        if len(all_variants) > MAX_TOTAL_VARIANTS:
            logger.warning("Variant cap reached (%d > %d). Stopping expansion.",
                           len(all_variants), MAX_TOTAL_VARIANTS)
            break

        # Next round = newly-spawned children. Their ancestry-contention
        # will implicitly reactivate any unfrozen ancestors above them.
        queue = new_children

    return all_variants


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Lineage path scoring
# ══════════════════════════════════════════════════════════════════════════════

def enumerate_lineage_paths(all_variants: list[AnchorVariant]) -> list[list[int]]:
    """Each reasoning path = a root-to-leaf chain in the parent lineage tree."""
    by_id = {v.var_id: v for v in all_variants}
    has_child = set()
    for v in all_variants:
        if v.parent_var_id is not None:
            has_child.add(v.parent_var_id)
    leaves = [v.var_id for v in all_variants if v.var_id not in has_child]

    paths: list[list[int]] = []
    for leaf_id in leaves:
        path: list[int] = []
        cur: int | None = leaf_id
        while cur is not None:
            path.append(cur)
            cur = by_id[cur].parent_var_id
        path.reverse()
        # Drop a leaf path if every variant in it has no locked relations
        if any(by_id[vid].locked_rels for vid in path):
            paths.append(path)
    # Sort: longer paths first
    paths.sort(key=lambda p: (-len(p), p))
    return paths


def score_lineage_paths(
    paths: list[list[int]],
    all_variants: list[AnchorVariant],
    n_subqs: int,
) -> list[tuple[list[int], float, dict[int, list[SolutionEntry]]]]:
    """Score each lineage path by chain-completeness × per-segment mean sim.

    score(p) = (Σ over v in p of mean_sim(scoring_top(v))) / n_subqs

    Normalising by n_subqs (the total number of skeleton sub-questions, NOT
    by len(path)) penalises lineage paths that died mid-chain — incomplete
    reasoning chains are shorter and get a smaller score.
    """
    n_subqs = max(n_subqs, 1)
    by_id = {v.var_id: v for v in all_variants}
    scored: list[tuple[list[int], float, dict[int, list[SolutionEntry]]]] = []
    for path in paths:
        seg_top: dict[int, list[SolutionEntry]] = {}
        seg_sim_total = 0.0
        covered = 0
        for vid in path:
            v = by_id[vid]
            if not v.locked_rels:
                continue
            scoring = select_top_relations_top1_anchored(
                v.locked_rels, REL_K_MIN, REL_K_MAX, REL_GAP_THR)
            context = select_top_relations_top1_anchored(
                v.locked_rels, REL_K_MIN_CONTEXT, REL_K_MAX_CONTEXT, REL_GAP_THR_CONTEXT)
            seg_top[vid] = context
            seg_sim_total += float(np.mean([e.sim for e in scoring]))
            covered += 1
        if covered == 0:
            continue
        path_score = seg_sim_total / n_subqs
        scored.append((path, path_score, seg_top))
    scored.sort(key=lambda x: -x[1])
    return scored


def organize_lineage_context(
    kg: KGIndex,
    scored: list[tuple[list[int], float, dict[int, list[SolutionEntry]]]],
    top_k_paths: int = TOP_K_PATHS,
) -> tuple[str, str]:
    """Per top-K-path concatenation, one paragraph per reasoning chain.
    Mirrors organize_variant_context but indexes by variant_id (lineage)."""
    if not scored:
        return "", "yes_empty"
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


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════

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
    anchor_subqs = [sq for sq in subqs if any(not e.virtual for e in sq.entities)]
    logger.info("Skeleton: %d nodes, %d edges, max_level=%d, %d anchor subqs",
                G_skel.number_of_nodes(), G_skel.number_of_edges(),
                max_level, len(anchor_subqs))

    if not anchor_subqs:
        logger.warning("No anchor sub-questions — falling back to direct relation search.")
        q_vec = encode_texts([question])[0]
        sims  = kg.relation_emb @ q_vec
        top_idx = np.argsort(-sims)[:20]
        parts = [kg.G.nodes[kg.relation_ids[i]]["text"] for i in top_idx]
        return {
            "question": question, "triplets": triplets, "ke1_size": 0,
            "max_level": max_level, "n_paths": 0,
            "fallback": "no_anchor_direct_search",
            "context": "\n\n".join(parts), "elapsed": round(time.time() - t0, 1),
        }

    # Step 3 — anchor lookup, keyed per real entity
    anchor_bindings = locate_anchor_entities_by_key(kg, anchor_subqs)
    total_ke1 = sum(len(v) for v in anchor_bindings.values())
    logger.info("Anchor bindings: %s (total %d ents)",
                {k: len(v) for k, v in anchor_bindings.items()}, total_ke1)
    if not anchor_bindings:
        return {"question": question, "context": "", "error": "no_anchor_bindings"}

    # Step 3.5 — Relation-Composed-Retrieval seeds (same as legacy)
    seed_queries = [question] + [sq.relation_text for sq in anchor_subqs]
    seed_relations = set(composed_retrieve_relations(kg, seed_queries))
    logger.info("Relation-Composed-Retrieval: %d queries → %d unique seed relations",
                len(seed_queries), len(seed_relations))

    # Step 4 — anchor-driven iterative search
    all_variants = iterative_search_anchor_driven(
        kg, subqs, anchor_bindings, seed_relations=seed_relations)
    logger.info("Anchor-driven search produced %d variants", len(all_variants))
    if os.environ.get("HP_DUMP_SOLUTIONS"):
        for v in all_variants:
            logger.info("  var %d qid=%d parent=%s anchor_ents=%d chain=%s rels=%d",
                        v.var_id, v.qid, v.parent_var_id,
                        len(v.anchor_ents),
                        {k: v.binding_chain[k] for k in sorted(v.binding_chain)},
                        len(v.locked_rels))
            for entry in sorted(v.locked_rels, key=lambda e: -e.sim)[:5]:
                title = kg.G.nodes[entry.rel_id].get("source_title", "")
                text  = (kg.G.nodes[entry.rel_id].get("text") or "")[:120]
                logger.info("    rel %s sim=%.3f title=%r text=%r",
                            entry.rel_id, entry.sim, title, text)

    if not all_variants:
        return {"question": question, "triplets": triplets, "ke1_size": total_ke1,
                "max_level": max_level, "n_paths": 0, "fallback": "no_variants",
                "context": "", "elapsed": round(time.time() - t0, 1)}

    # Step 5 — lineage paths + scoring
    paths      = enumerate_lineage_paths(all_variants)
    scored_all = score_lineage_paths(paths, all_variants, len(subqs))

    # Per-chain quota: every parallel reasoning chain keeps its best path
    # before the remaining slots go to the global top scorers. A path's chain
    # is that of its root variant.
    by_id         = {v.var_id: v for v in all_variants}
    chain_of_qid  = compute_qid_chains(subqs)
    chain_of_path = [chain_of_qid.get(by_id[p[0]].qid, -1)
                     for p, _s, _t in scored_all]
    scored = select_top_paths_per_chain(scored_all, chain_of_path, TOP_K_PATHS)
    logger.info("Lineage paths: %d enumerated, %d scored, %d selected across %d chain(s)",
                len(paths), len(scored_all), len(scored),
                len(set(chain_of_qid.values())))

    # Step 6 — context
    context, fallback = organize_lineage_context(kg, scored)

    elapsed = time.time() - t0
    return {
        "question":  question,
        "triplets":  triplets,
        "ke1_size":  total_ke1,
        "max_level": max_level,
        "n_paths":   len(scored_all),
        "fallback":  fallback,
        "context":   context,
        "elapsed":   round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=-1,
                    help="Limit to top-N largest questions (debug). "
                         "Use -1 (default) or 0 to process ALL.")
    ap.add_argument("--question-idx", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--kg", type=str, default=None)
    ap.add_argument("--out", type=str, default="retrieval_anchor_driven.json")
    ap.add_argument("--ablation", type=str, default="anchor_driven")
    ap.add_argument("--save-timing", dest="save_timing", type=str, default=None)
    args = ap.parse_args()

    import retrieve as _retrieve_mod
    if args.data:
        _retrieve_mod.DATA_FILE = Path(args.data)
    if args.kg:
        _retrieve_mod.KG_PKL = Path(args.kg)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    kg = KGIndex(_retrieve_mod.KG_PKL)
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
        res["ablation"]        = args.ablation
        res["retrieve_total_ms"] = round(total_ms, 1)
        results.append(res)
        timing_records.append({
            "qid": q.get("id", f"idx-{i}"),
            "retrieve_total_ms": round(total_ms, 1),
            "elapsed_internal_s": res.get("elapsed"),
            "context_chars": len(res.get("context") or ""),
            "ablation": args.ablation,
        })
        logger.info("Context preview:\n%s", (res.get("context") or "")[:500])

    from retrieve import ROOT
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
        }
        timing_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved timing → %s", timing_path)


if __name__ == "__main__":
    main()
