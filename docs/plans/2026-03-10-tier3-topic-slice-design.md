# Tier 3 Topic-Slice Design

Date: 2026-03-10

## Goal

Add an eval-only way to build a topic-bounded Tier 3 chunk subset from the frozen chunk corpus, so RAGAS can be tested on a narrower semantic pool without changing production chunking or the main RAG path.

## Why

Recent Tier 3 runs suggest RAGAS degrades on broad medical corpora because it has too many opportunities to create weak cross-topic joins. A deterministic topic slice gives us a cheaper, more controlled experiment:
- same production chunk corpus
- same Tier 3 generator
- smaller and more coherent topic space

## Scope

In scope:
- extend the existing subset builder
- allow repeatable topic keywords
- keep existing junk/front-matter/caption filters
- bias selection toward rows matching the requested topic keywords
- keep deterministic output and reporting

Out of scope:
- any production retrieval or chunking changes
- any changes to RAGAS internals
- any new Tier 3 generator mode beyond consuming a different chunk corpus file

## Design

### Input contract

Extend [scripts/build_tier3_chunk_subset.py](/home/tough/medical_chatbot/MediGenius/scripts/build_tier3_chunk_subset.py) with:
- repeatable `--topic-keyword`

If no topic keywords are provided, behavior remains unchanged.

If topic keywords are provided:
- rows must still pass the existing drop filters
- rows must match at least one topic keyword in section/content
- matching is deterministic and string-based
- the output path defaults to a topic-specific file when the caller does not override `--output`

### Matching strategy

Use deterministic text matching only:
- exact normalized phrase match in section/content
- or all normalized terms from a keyword phrase must be present in the combined row text

No embedding similarity, no LLM labeling, no fuzzy heuristics.

### Selection strategy

After filtering to topic-matching rows:
- keep the existing section-priority ordering
- keep existing page-band distribution
- keep per-page-section and per-page-window caps

This keeps the topic slice coherent while still preventing local repetition.

### Reporting

CLI summary should include:
- input rows
- topic keywords
- topic-matching rows
- kept rows
- page span
- dropped rows by reason

## Verification

Add tests for:
- topic-keyword filtering
- multi-keyword matching
- deterministic topic-specific default output path
- no behavior change when no topic keywords are supplied

## Expected usage

Example:

```bash
python3 scripts/build_tier3_chunk_subset.py \
  --input eval/golden/v1/tier3_chunk_corpus.jsonl \
  --topic-keyword "cervical cancer" \
  --topic-keyword "pap test" \
  --limit 120
```

Expected output file:

```text
eval/golden/v1/tier3_chunk_corpus.topic-cervical-cancer-pap-test.jsonl
```

This remains an eval-only input selection policy. It does not modify production chunking or production retrieval behavior.
