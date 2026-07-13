"""
Aggregate every output JSON under `results/` into a single
`results/final_tables.md` that maps directly to the paper tables.

Walks the directory structure documented in RESULTS_LAYOUT.md and emits
one Markdown table per paper table, with `--` placeholders for any
configuration whose JSON has not yet been produced. This lets the user
re-run the aggregator incrementally as more configurations complete.

Usage:
  python src/aggregate_results.py --results results --out results/final_tables.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

DATASETS = ["hotpot", "2wiki", "musique"]
LLMS     = ["qwen3-14b", "llama3.3-70b", "gpt-4o-mini"]
METHODS_MAIN = [
    "NaiveGeneration", "StandardRAG", "GraphRAG", "HippoRAG2",
    "HyperGraphRAG", "Hyper-RAG", "Cog-RAG", "HGRAG",
    "BeyondChunksGraphs", "LogicRAG", "HyperPathsRAG",
]
ABLATIONS = ["full", "no-ntary", "no-multitarget", "no-chaincomp",
             "no-order", "no-hypergraph", "no-bm25"]
PRIMING_VARIANTS = ["reasoning-first", "answer-first", "answer-only"]
NER_MODELS = ["nuner-zero", "gliner", "spacy"]
EFFICIENCY_METHODS = ["GraphRAG", "HippoRAG2", "HyperGraphRAG", "HyperPathsRAG"]
SENSITIVITY_PARAMS = {
    "lambda": [0.0, 0.1, 0.3, 0.5, 1.0],
    "tau":    [0.70, 0.80, 0.85, 0.90, 0.95],
    "delta":  [0.01, 0.03, 0.05, 0.08, 0.12],
    "k-min":  [1, 2, 2, 3, 3],
    "k-max":  [3, 4, 5, 6, 8],
}


def load_json(p: Path) -> dict | None:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning("Failed to parse %s: %s", p, e)
    return None


def cell(v):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def render_main_table(results_dir: Path) -> str:
    out = ["## Table 2 — Main Experiments (EM / F1)\n"]
    header = "| Method | Answer LLM | " + " | ".join(
        f"{ds.upper()} EM | {ds.upper()} F1" for ds in DATASETS
    ) + " | Avg EM | Avg F1 |"
    sep = "|" + "|".join(["---"] * (2 + 2 * len(DATASETS) + 2)) + "|"
    out.append(header)
    out.append(sep)
    for method in METHODS_MAIN:
        for llm in LLMS:
            cells = [method, llm]
            em_sum = f1_sum = 0.0
            n_seen = 0
            for ds in DATASETS:
                p = results_dir / "main_table" / ds / method / llm / "score.json"
                d = load_json(p)
                if d:
                    em_sum += float(d.get("em", 0))
                    f1_sum += float(d.get("f1", 0))
                    n_seen += 1
                    cells.append(cell(d.get("em")))
                    cells.append(cell(d.get("f1")))
                else:
                    cells.append("--")
                    cells.append("--")
            if n_seen:
                cells.append(f"{em_sum / n_seen:.1f}")
                cells.append(f"{f1_sum / n_seen:.1f}")
            else:
                cells.extend(["--", "--"])
            out.append("| " + " | ".join(cells) + " |")
        out.append(sep)
    return "\n".join(out) + "\n"


def render_ablation_table(results_dir: Path, llm: str = "llama3.3-70b") -> str:
    out = [f"## Table 3 — Ablation Studies (answer LLM = {llm})\n"]
    out.append("| Variant | EM avg | F1 avg |")
    out.append("|---|---|---|")
    for abl in ABLATIONS:
        em_sum = f1_sum = 0.0
        n = 0
        for ds in DATASETS:
            p = results_dir / "ablation" / ds / abl / llm / "score.json"
            d = load_json(p)
            if d:
                em_sum += float(d.get("em", 0))
                f1_sum += float(d.get("f1", 0))
                n += 1
        em_v = cell(em_sum / n) if n else "--"
        f1_v = cell(f1_sum / n) if n else "--"
        out.append(f"| {abl} | {em_v} | {f1_v} |")
    return "\n".join(out) + "\n"


def render_efficiency_table(results_dir: Path) -> str:
    out = ["## Table 4 — Efficiency Comparison\n"]
    out.append("| Method | Retrieval (s/q) | Tokens/q | KG build (min) |")
    out.append("|---|---|---|---|")
    for method in EFFICIENCY_METHODS:
        ret_per_q = []
        tok_per_q = []
        build_min = []
        for ds in DATASETS:
            t = load_json(results_dir / "efficiency" / ds / method / "timing.json")
            kg = load_json(results_dir / "kg_stats" / ds / "kg_stats.json")
            if t and isinstance(t.get("retrieve_sec_per_q"), (int, float)):
                ret_per_q.append(t["retrieve_sec_per_q"])
            if t and isinstance(t.get("tokens_per_q"), dict):
                tot = t["tokens_per_q"].get("total_tokens")
                if isinstance(tot, (int, float)):
                    tok_per_q.append(tot)
            if kg and isinstance(kg.get("build_time_min"), (int, float)) and method == "HyperPathsRAG":
                build_min.append(kg["build_time_min"])
        ret = f"{sum(ret_per_q)/len(ret_per_q):.2f}" if ret_per_q else "--"
        tok = f"{sum(tok_per_q)/len(tok_per_q):.0f}" if tok_per_q else "--"
        bld = f"{sum(build_min)/len(build_min):.1f}" if build_min else "--"
        out.append(f"| {method} | {ret} | {tok} | {bld} |")
    return "\n".join(out) + "\n"


def render_appendix_B(results_dir: Path) -> str:
    out = ["## Appendix B — Entity Embedding Probe\n"]
    out.append("| Dataset | Strategy | Syn-Recall | False-Merge | F1 |")
    out.append("|---|---|---|---|---|")
    for ds in DATASETS:
        d = load_json(results_dir / "appendix_B_embedding" / ds / "embedding_probe.json")
        for strat in ["name_only", "name_plus_def", "hybrid_ours"]:
            if d and "by_strategy" in d:
                v = d["by_strategy"].get(strat, {})
                sr = cell(v.get("syn_recall", "--"))
                fm = cell(v.get("false_merge", "--"))
                f1 = cell(v.get("f1", "--"))
            else:
                sr = fm = f1 = "--"
            out.append(f"| {ds} | {strat} | {sr} | {fm} | {f1} |")
    return "\n".join(out) + "\n"


def render_appendix_C(results_dir: Path) -> str:
    out = ["## Appendix C — Hyperparameter Sensitivity (EM / F1 avg)\n"]
    for param, values in SENSITIVITY_PARAMS.items():
        out.append(f"### Param: {param}\n")
        header_vals = " | ".join(f"{v}" for v in values)
        out.append(f"| Dataset | {header_vals} |")
        out.append("|" + "|".join(["---"] * (len(values) + 1)) + "|")
        for ds in DATASETS:
            cells = [ds]
            for v in values:
                p = results_dir / "appendix_C_sensitivity" / ds / param / str(v) / "score.json"
                d = load_json(p)
                cells.append(f"{cell(d.get('em'))}/{cell(d.get('f1'))}" if d else "--")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def render_appendix_E(results_dir: Path) -> str:
    out = ["## Appendix E — Failure Mode Distribution\n"]
    categories = [
        "kg_missing_fact", "decomposition_under_coverage",
        "entity_disambiguation", "long_tail_relation",
        "llm_answering_error", "annotation_noise",
    ]
    header_cells = ["Failure mode"] + [ds.upper() for ds in DATASETS] + ["Mean"]
    out.append("| " + " | ".join(header_cells) + " |")
    out.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for cat in categories:
        row = [cat]
        vs = []
        for ds in DATASETS:
            d = load_json(results_dir / "appendix_E_failures" / ds / "failures.json")
            if d and "distribution_fraction" in d:
                v = d["distribution_fraction"].get(cat)
                row.append(cell(v))
                if isinstance(v, (int, float)):
                    vs.append(v)
            else:
                row.append("--")
        row.append(cell(sum(vs) / len(vs)) if vs else "--")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def render_appendix_G(results_dir: Path) -> str:
    out = ["## Appendix G — CoT Priming Order (EM avg / F1 avg)\n"]
    out.append("| Priming | EM avg | F1 avg |")
    out.append("|---|---|---|")
    for pr in PRIMING_VARIANTS:
        em_sum = f1_sum = 0.0
        n = 0
        for ds in DATASETS:
            d = load_json(results_dir / "appendix_G_cot" / ds / pr / "score.json")
            if d:
                em_sum += float(d.get("em", 0))
                f1_sum += float(d.get("f1", 0))
                n += 1
        em_v = cell(em_sum / n) if n else "--"
        f1_v = cell(f1_sum / n) if n else "--"
        out.append(f"| {pr} | {em_v} | {f1_v} |")
    return "\n".join(out) + "\n"


def render_appendix_H(results_dir: Path) -> str:
    out = ["## Appendix H — NER Model Comparison\n"]
    out.append("| Dataset | NER | Recall | Over-extract | End EM |")
    out.append("|---|---|---|---|---|")
    for ds in DATASETS:
        for ner in NER_MODELS:
            stat = load_json(results_dir / "appendix_H_ner" / ds / ner / "ner_stats.json")
            ee = load_json(results_dir / "appendix_H_ner" / ds / ner / "end_em.json")
            rec = cell(stat.get("recall")) if stat else "--"
            ov  = cell(stat.get("over_extract")) if stat else "--"
            em  = cell(ee.get("em")) if ee else "--"
            out.append(f"| {ds} | {ner} | {rec} | {ov} | {em} |")
    return "\n".join(out) + "\n"


def render_appendix_I(results_dir: Path) -> str:
    out = ["## Appendix I — RAGAS Faithfulness / Context-Relevance\n"]
    out.append("| Dataset | Method | Faithfulness | Context-Relevance |")
    out.append("|---|---|---|---|")
    for ds in DATASETS:
        for method in EFFICIENCY_METHODS:
            d = load_json(results_dir / "appendix_I_ragas" / ds / method / "ragas.json")
            f = cell(d.get("faithfulness_mean")) if d else "--"
            r = cell(d.get("context_relevance_mean")) if d else "--"
            out.append(f"| {ds} | {method} | {f} | {r} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="results")
    ap.add_argument("--out",     type=str, default="results/final_tables.md")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)],
                        force=True)

    results_dir = Path(args.results)
    blocks = [
        f"# HyperPathsRAG — Aggregated Results\n",
        f"Run aggregator: walked `{results_dir.resolve()}`.\n",
        render_main_table(results_dir),
        render_ablation_table(results_dir),
        render_efficiency_table(results_dir),
        render_appendix_B(results_dir),
        render_appendix_C(results_dir),
        render_appendix_E(results_dir),
        render_appendix_G(results_dir),
        render_appendix_H(results_dir),
        render_appendix_I(results_dir),
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(blocks), encoding="utf-8")
    logging.info("Wrote aggregated tables → %s", out_path)


if __name__ == "__main__":
    main()
