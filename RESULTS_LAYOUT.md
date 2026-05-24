# Experiment Output Layout

All scripts write into a single top-level `results/` directory that is
laid out so that each subdirectory corresponds to one paper table.
The aggregator (`src/aggregate_results.py`) walks this tree and emits
the final filled-in tables as Markdown.

```
results/
├── kg_stats/                       # KG construction stats per dataset
│   ├── hotpot/kg_stats.json        # entity_count, fact_count, build_time_s, ...
│   ├── 2wiki/kg_stats.json
│   └── musique/kg_stats.json
│
├── main_table/                     # Paper Table 2 (Main Experiments, 11 methods × 3 LLMs × 3 datasets)
│   └── <dataset>/<method>/<llm>/
│       ├── retrieval.json          # per-question retrieved context + retrieve_ms
│       ├── answers.json            # per-question predicted_answer + tokens
│       └── score.json              # {em, f1, n}
│
├── ablation/                       # Paper Table 3 (HyperPathsRAG variants)
│   └── <dataset>/<ablation>/<llm>/
│       └── score.json              # {em, f1, n}
│       # ablation ∈ {full, no-ntary, no-multitarget, no-chaincomp,
│       #            no-order, no-hypergraph, no-bm25}
│
├── efficiency/                     # Paper Table 4 (Efficiency)
│   └── <dataset>/<method>/
│       └── timing.json             # {retrieve_sec_per_q, tokens_per_q, kg_build_min}
│
├── appendix_B_embedding/           # Appendix B (entity embedding ablation)
│   └── <dataset>/embedding_probe.json
│       # {syn_recall, false_merge, f1} per strategy ∈ {name-only, name+def, hybrid}
│
├── appendix_C_sensitivity/         # Appendix C (hyperparameter sensitivity)
│   └── <dataset>/<param>/<value>/score.json
│       # param ∈ {lambda, tau, delta, k-min, k-max}
│
├── appendix_D_cases/               # Appendix D (case studies)
│   └── case_<id>/                  # frozen retrieval traces for selected queries
│       └── trace.json              # decomposition, variant graph, paths, scores
│
├── appendix_E_failures/            # Appendix E (failure mode analysis)
│   └── <dataset>/failures.json     # 50 random errors with categorisation
│
├── appendix_G_cot/                 # Appendix G (CoT priming order)
│   └── <dataset>/<priming>/score.json
│       # priming ∈ {reasoning-first, answer-first, answer-only}
│
├── appendix_H_ner/                 # Appendix H (NER model comparison)
│   └── <dataset>/<ner>/{kg_stats.json, end_em.json}
│       # ner ∈ {nuner-zero, gliner, spacy}
│
├── appendix_I_ragas/               # Appendix I (RAGAS scoring)
│   └── <dataset>/<method>/ragas.json
│       # {faithfulness, context_relevance}
│
└── final_tables.md                 # produced by aggregate_results.py
```

## Which paper table reads which JSON

| Paper table | Source directory | Aggregator output column |
|---|---|---|
| Table 2 (main) | `main_table/<ds>/<method>/<llm>/score.json` | `em`, `f1` |
| Table 3 (ablation) | `ablation/<ds>/<abl>/<llm>/score.json` | `em`, `f1` |
| Table 4 (efficiency) | `efficiency/<ds>/<method>/timing.json` + `kg_stats/<ds>/kg_stats.json` | `retrieve_sec_per_q`, `tokens_per_q`, `kg_build_min` |
| Appendix B | `appendix_B_embedding/<ds>/embedding_probe.json` | `syn_recall`, `false_merge`, `f1` |
| Appendix C | `appendix_C_sensitivity/<ds>/<param>/<v>/score.json` | per (param, value) cell |
| Appendix E | `appendix_E_failures/<ds>/failures.json` | per-category fraction |
| Appendix G | `appendix_G_cot/<ds>/<priming>/score.json` | `em`, `f1` |
| Appendix H | `appendix_H_ner/<ds>/<ner>/end_em.json` | `recall`, `over_extract`, `end_em` |
| Appendix I | `appendix_I_ragas/<ds>/<method>/ragas.json` | `faithfulness`, `context_relevance` |

## Single-run philosophy

Each (dataset, method, ablation, llm, priming) configuration is
computed at most once and its JSON is cached.  The orchestrator
`scripts/run_all.sh` skips any (config, output) pair whose JSON
already exists, so re-running the script after a partial failure is
safe and incremental.

Per-question intermediate state (decomposition, variant graph, path
list, retrieval timing) is saved inside `retrieval.json` for the main
run only, enabling Appendix D case studies and Appendix E error
categorisation to operate on the same data without re-running
retrieval.
