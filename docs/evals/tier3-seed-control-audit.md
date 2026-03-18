# Tier 3 Benchmark Note

This note records the current Tier 3 benchmark boundary after retiring the
synthetic RAGAS generation lane.

## Current state

- [tier3_rag.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.jsonl) is the single curated Tier 3 benchmark.
- The benchmark contains `53` approved `human_seed` rows with explicit lane tags:
  - `11` `benchmark_lane:baseline`
  - `16` `benchmark_lane:medium_hard`
  - `26` `benchmark_lane:retrieval_hard`
- The benchmark is now split deterministically into `14` `dev` rows and `39` `test` rows.
- Multi-context evidence is annotated as `required` versus `supporting`, and every row now carries stable evidence anchors plus explicit hard-negative candidates for later retrieval analysis.
- [tier3_chunk_corpus.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_chunk_corpus.jsonl) remains the frozen retrieval corpus used by retrieval evals and manual seed review.
- [tier3_manual_seed_pdf_authoring.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_manual_seed_pdf_authoring.py), [build_tier3_manual_seed_authoring_index.py](/home/tough/medical_chatbot/MediGenius/scripts/build_tier3_manual_seed_authoring_index.py), and [tier3_manual_seed_authoring_index.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_manual_seed_authoring_index.jsonl) are the supported Tier 3 authoring path.

## Retired synthetic path

- The old Tier 3 RAGAS generator, smoke runner, raw checkpoint, and generator-specific tests are now frozen under [bin/legacy_tier3_ragas](/home/tough/medical_chatbot/MediGenius/bin/legacy_tier3_ragas).
- Those archived files are retained for reference only.
- The active repo must not depend on them for dataset generation, CI, or retrieval evaluation.

## Why

The curated manual benchmark is the only Tier 3 source currently trusted for
later RAG pipeline evaluation. The synthetic RAGAS lane consumed substantial
effort without producing a benchmark-quality set, so it has been removed from
the active eval surface instead of kept as a half-supported path.
