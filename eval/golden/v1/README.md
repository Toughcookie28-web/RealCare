# Golden Dataset v1

This directory contains the current frozen eval datasets plus the active Tier 3
manual-authoring support artifacts.

Key dataset files:
- `tier1_guardrail.jsonl`
- `tier2_routing.jsonl`
- `tier3_rag.jsonl`
- `tier4_conversation.jsonl`
- `tier5_robustness.jsonl`

Tier 3 benchmark notes:
- `tier3_rag.jsonl` is split into deterministic `dev` and `test` slices.
- Each Tier 3 row carries `evidence.required`, `evidence.supporting`, and `evidence.hard_negative_candidates`.
- Evidence anchors are stable beyond `chunk_id` and include `doc_id`, `page`, `section`, `content_type`, and `anchor_text`.

Tier 3 support artifacts:
- `tier3_manual_seed_authoring_index.jsonl`
- `tier3_chunk_corpus.jsonl`

Retired synthetic Tier 3 artifacts now live under `bin/legacy_tier3_ragas/`.

Generated QA artifacts:
- `golden_v1_manifest.json`
- `qa_report.json`
