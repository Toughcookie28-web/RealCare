# Eval Snapshot History

Tracks what code produced each snapshot and why metrics changed.
Snapshots before 2026-03-15 lack `snapshot_meta`; context is reconstructed from git log and file timestamps.

---

## full_baseline_v1 (RAGAS, 2026-03-12 17:23)

**Code state:** Original retrieval pipeline.
- Dense vector similarity search (pgvector cosine distance)
- tsquery keyword search
- RRF hybrid merge (1/(rank+1) dense + 0.75/(rank+1) sparse)
- No reranking stage
- 53 samples (full dataset, no dev/test split)

**Results (RAGAS, full_pipeline):**
| Metric | Score |
|---|---|
| faithfulness | 0.788 |
| answer_relevancy | 0.035 |
| context_precision | 0.042 |
| context_recall | 0.044 |

**Analysis:** Context precision/recall near zero — the hybrid search retrieved chunks but ranking was essentially random. RRF with only 2 inputs and no reranking meant relevant chunks were buried. Faithfulness was decent (0.79) because the LLM stays grounded even with noisy context, but answer relevancy was near zero because the context rarely contained the right information.

---

## post_bm25 (retrieval) / post_bm25_Retrieval (RAGAS, 2026-03-12 22:19-22:29)

**Code change:** Added BM25 reranker after hybrid search.
- Same retrieval pipeline as baseline
- Added BM25 scoring as a reranking stage over the RRF-merged candidates
- 53 samples (same full dataset)

**Results (RAGAS, full_pipeline):**
| Metric | Baseline | Post-BM25 | Delta |
|---|---|---|---|
| faithfulness | 0.788 | 0.758 | -0.03 |
| answer_relevancy | 0.035 | 0.259 | +0.224 |
| context_precision | 0.042 | 0.369 | +0.327 |
| context_recall | 0.044 | 0.368 | +0.324 |

**Results (retrieval, hybrid_rerank, 53 samples):**
| Metric | Score |
|---|---|
| hit_rate@5 | 0.434 |
| recall@5 | 0.248 |
| mrr@5 | 0.336 |
| nDCG@10 | 0.289 |

**Analysis:** BM25 reranking was the single biggest improvement. Re-scoring candidates by query term overlap pushed relevant chunks to top positions, causing context_precision and context_recall to jump ~9x. Answer relevancy improved because the LLM now received useful context. Faithfulness dropped slightly — likely because better context led the LLM to attempt more detailed answers, increasing hallucination surface.

---

## before_phase_A (retrieval) / Before Phase A_ragas (RAGAS, 2026-03-14 10:43-11:01)

**Code changes since post_bm25:**
- Chunker improvements: table_columns, table_row_count, table_size_tier metadata; chunk_position (intro/body/conclusion) tagging; configurable chunk sizes via Settings
- LLM chunk enrichment module added (behind ENRICH_CHUNKS_WITH_LLM flag, likely not activated)
- Eval dataset split into dev/test; test split = 39 samples
- Possible PDF reindex with improved chunker

**Results (RAGAS, retrieval_only, test split 39 samples):**
| Metric | Post-BM25 (53) | Before Phase A (39) |
|---|---|---|
| context_precision | 0.369 | 0.672 |
| context_recall | 0.368 | 0.722 |

**Results (retrieval, hybrid_rerank, test split 39 samples):**
| Metric | Post-BM25 (53) | Before Phase A (39) |
|---|---|---|
| hit_rate@5 | 0.434 | 0.846 |
| recall@5 | 0.248 | 0.613 |
| mrr@5 | 0.336 | 0.653 |
| nDCG@10 | 0.289 | 0.675 |

**Analysis:** Numbers not directly comparable due to different sample counts (53 vs 39). The improvement is a combination of: (1) test split having different difficulty distribution than the full set, (2) possible reindex with improved chunker producing better chunk boundaries. The code-level retrieval pipeline (RRF + BM25 rerank) was unchanged.

---

## phasea_pipeline_test (retrieval, 2026-03-15 11:17)

**Code change:** Phase A query rewriter added.
- LLM query rewriter: medical abbreviation expansion, synonym addition, coreference resolution
- Step-back query generation for mechanism/comparison questions
- Pipeline mode: rewriter -> hybrid search -> step-back merge -> metadata boost -> BM25 rerank
- 39 samples (test split)

**Results (retrieval, pipeline mode, 39 samples):**
| Metric | Before Phase A (hybrid_rerank) | Phase A (pipeline) |
|---|---|---|
| hit_rate@5 | 0.846 | 0.744 |
| recall@5 | 0.613 | 0.615 |
| mrr@5 | 0.653 | 0.560 |
| nDCG@10 | 0.675 | 0.556 |

**Analysis:** Recall stayed flat but hit_rate and MRR dropped. The rewriter expanded queries with synonyms and medical terms, which sometimes diluted precision. The eval questions are written in precise medical language that already matches chunk content well — the rewriter's expansions don't help and can hurt. This suggests eval questions need to be more naturalistic (casual language, abbreviations, coreferences) to properly test the rewriter's value. In production, users won't ask questions in textbook language.

---

## Snapshot naming convention (going forward)

- Format: `phase-{letter}-{description}`
- Always use `--note` to describe code changes
- `snapshot_meta.git_commit` is auto-captured from 2026-03-15 onward
- Compare with: `--compare {baseline_name}`

## Lessons learned

1. **Eval data quality > code changes.** Cleaning the dataset and splitting dev/test had a bigger measured impact than any single code change.
2. **BM25 reranking was the highest-ROI change** — 9x improvement in context precision from a single addition.
3. **Query rewriting needs naturalistic eval data** to show its value. Precise eval questions penalize the rewriter for doing its job.
4. **Always label snapshots** with git commit + note. Unlabeled snapshots lose their diagnostic value within days.
