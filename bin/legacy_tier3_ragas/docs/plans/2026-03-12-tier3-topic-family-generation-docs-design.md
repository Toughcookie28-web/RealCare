# Tier 3 Topic-Family Generation Docs Design

**Goal:** Replace raw article-window generation docs with cleaner, same-family generation docs that are large enough for medically coherent multi-hop but small enough to avoid artificial cross-topic bridges.

**Decision:** For this medical encyclopedia, Tier 3 generation should not use retrieval chunks or ultra-broad chapter blobs. The measured article distribution is already dense: median article size is about 2 pages and 1.8k tokens, while section units are mostly too small. The right intermediate unit is a topic-family generation doc in the rough 3k–8k token range.

**Current temporary implementation:** Keep article extraction as the canonical upstream unit, then:
- clean article bodies by dropping pre-title carryover on start pages
- stop article body extraction at intra-article back-matter markers like `BOOKS`, `PERIODICALS`, `ORGANIZATIONS`, and `OTHER`
- build temporary same-family generation docs for the cancer experiment:
  - one `Cancer` overview bundle
  - one `Cancer therapy` care bundle

**Why:** The earlier cancer-family run showed the remaining quality problem was not just RAGAS query distribution. Dirty article boundaries and reference/back-matter leakage were still contaminating the generation substrate with organization names and bylines. Cleaning the article docs and promoting the temporary experiment to two larger same-family generation docs should give RAGAS a more coherent graph for medically meaningful multi-hop.

**Non-goals for this step:**
- no canonical markdown rewrite yet
- no generalized family bundling across all books yet
- no retrieval-time chunking changes yet

**Next validation:** Re-run the narrow cancer-family Tier 3 generation and review whether the row quality shifts from organization/byline noise toward medically grounded same-family reasoning.
