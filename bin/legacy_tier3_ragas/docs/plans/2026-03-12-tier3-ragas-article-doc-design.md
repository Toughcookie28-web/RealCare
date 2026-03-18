# Tier 3 RAGAS Article-Document Design

## Problem

The current Tier 3 synthetic generator feeds retrieval chunks directly into the default RAGAS document-generation path. That is a substrate mismatch.

- The retrieval chunks were built for indexing and retrieval, not for question generation.
- The medical book corpus has many generic headings and non-self-identifying chunks.
- Downstream filters cannot recover generation quality when the upstream source units are already fragmented.

Official RAGAS guidance fits a different model:
- use source documents with `generate_with_langchain_docs`
- let RAGAS split/transform those source documents internally
- use the pre-chunked path only when intentionally preserving chunking

## PDF structure check

The medical book behaves like an encyclopedia of topic entries.

Observed from the current PDF-derived authoring index:
- 357 article windows across the book
- average article span about 2.1 pages
- median span 2 pages
- most entries span 1-3 pages
- longer disease entries span 4-7 pages
- recurring headings include Definition, Description, Causes and symptoms, Diagnosis, Treatment, Prognosis, Prevention

That means the natural RAGAS source unit is an article/topic entry, not a retrieval chunk and not the whole book.

## Options considered

### Option 1: Whole-book or large chapter documents

Reject.

Why:
- topic bleed is too high
- generic headings repeat across unrelated entries
- question generation would drift across diseases/tests too easily
- this is a poor match for encyclopedia structure

### Option 2: Section-level article documents

Possible fallback, not recommended as the primary path.

Why:
- keeps topic identity better than retrieval chunks
- but still fragments the source too early
- weakens cross-section reasoning and recreates shallow generation pressure
- likely to drift back toward the same failure mode that made chunk-fed generation weak

### Option 3: Article-entry documents

Recommended.

Why:
- matches the actual book structure
- preserves entity/topic identity inside the source document
- lets RAGAS do official document-level transforms/splitting
- avoids using retrieval chunks as the generation substrate
- still allows post-generation mapping back to retrieval chunks for schema fields

## Recommended architecture

### Source documents for RAGAS

Build article-entry source documents from the PDF.

Each article source document should contain:
- full article text for the entry
- article title
- page span
- local headings
- stable article document id
- aligned retrieval chunk ids/pages/sections for later schema mapping

### Shared extraction layer

Extract the current PDF article-window logic into a neutral module that is not named for manual seeds.

Reason:
- Tier 3 synthetic generation and manual-seed authoring both need the same article boundary recovery
- the generator should not depend on a helper whose name implies a different responsibility

### Generator boundary

The Tier 3 generator should:
1. load cached article documents if available
2. otherwise build article documents from the PDF
3. checkpoint the article-document corpus for repeatability
4. run RAGAS on those article documents
5. checkpoint raw RAGAS rows
6. convert rows to schema_v1 draft samples
7. during conversion, align contexts back to retrieval chunks for `ground_truth_chunk_ids`

### Chunk usage after the change

Retrieval chunks are still needed, but only for:
- alignment metadata on article documents
- schema output fields such as `ground_truth_chunk_ids`
- retrieval evaluation downstream

Retrieval chunks are no longer the direct generation substrate.

## Smoke-first rule

Before any scale-up run, the pipeline should support a bounded smoke mode.

Smoke run requirements:
- small article count
- small requested sample count
- explicit temp output paths so the curated benchmark and golden artifacts are not touched
- same article-document path as the full run, not a fake separate code path

## Downstream cleanup stance

Do not restore the deleted downstream acceptance/filtering logic first.

Reason:
- the main issue is upstream source-document shape
- first rerun should be on the corrected document-first path with minimal downstream handling
- only restore or redesign downstream review logic if quality is still poor after the substrate fix

## Success criteria

The redesigned Tier 3 synthetic path is correct when:
- RAGAS receives article-entry documents, not retrieval chunks
- the pipeline remains repeatable via cached article docs and raw-row checkpoints
- schema_v1 outputs still include aligned retrieval chunk ids
- a small smoke run is available before any large run
- the manual-seed authoring helper remains compatible with the shared article-boundary logic
