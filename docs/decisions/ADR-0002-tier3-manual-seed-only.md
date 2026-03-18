# ADR-0002: Retire Active Tier 3 Synthetic Generation And Keep Manual Seeds Only

## Status
Accepted

## Context
Tier 3 synthetic generation consumed substantial implementation and review
effort without producing a trusted benchmark-quality set. The final curated
Tier 3 benchmark is already manual-seed based and is the only Tier 3 source
currently trusted for later RAG pipeline evaluation.

Keeping the RAGAS generator in the active eval package was increasing
architectural confusion:

- active docs had to describe an eval path the team no longer trusted
- tests and scripts had to keep supporting a generator that was not serving the
  benchmark
- the manual authoring path depended on code that still carried synthetic
  generation responsibilities

## Decision
Retire the active Tier 3 synthetic RAGAS generation lane from the main eval
package.

The supported active Tier 3 surface is now:
- the curated benchmark in
  [eval/golden/v1/tier3_rag.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_rag.jsonl)
- the frozen retrieval corpus in
  [eval/golden/v1/tier3_chunk_corpus.jsonl](/home/tough/medical_chatbot/MediGenius/eval/golden/v1/tier3_chunk_corpus.jsonl)
- the PDF-first manual-seed authoring helpers in
  [eval/tier3_manual_seed_pdf_authoring.py](/home/tough/medical_chatbot/MediGenius/eval/tier3_manual_seed_pdf_authoring.py)
  and
  [scripts/build_tier3_manual_seed_authoring_index.py](/home/tough/medical_chatbot/MediGenius/scripts/build_tier3_manual_seed_authoring_index.py)

The retired synthetic generator, smoke runner, raw checkpoint, and
generator-specific tests are frozen under
[bin/legacy_tier3_ragas](/home/tough/medical_chatbot/MediGenius/bin/legacy_tier3_ragas)
for reference only.

## Consequences
What becomes easier:
- Tier 3 evaluation has one active benchmark contract again
- manual-seed authoring can be reviewed without synthetic-generation baggage
- active docs and tests describe only the trusted benchmark path

What becomes harder:
- there is no supported synthetic Tier 3 expansion path in the active package
- future synthetic experiments must be reintroduced explicitly instead of
  reusing the archived code path by accident

Tradeoff accepted:
- we prefer one smaller trusted Tier 3 surface over keeping a larger,
  untrusted synthetic workflow half-alive in the main repo

## Alternatives considered
- Keep the synthetic generator active but marked experimental
  - rejected because it still pollutes the active eval surface and invites
    accidental reuse
- Delete the synthetic generator entirely
  - rejected because preserving the legacy code and artifacts in a frozen
    archive is useful for history and comparison
