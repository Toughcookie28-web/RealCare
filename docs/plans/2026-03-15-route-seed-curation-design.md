# Route Seed Curation Design

**Goal:** Add a high-precision raw route seed set for the k-NN fallback router, covering the active routing surfaces with medically realistic but intentionally unambiguous examples.

**Scope:** Create `data/route_seeds_raw.json` from scratch for `vector`, `chitchat`, `web`, `memory`, `literature`, and `future_clinical_db`; generate the embedded `data/route_seeds.json`; add a small contract test to keep route coverage stable.

**Design choices:**
- Use 10 seeds per route to give the embedding space enough width without introducing noisy mixed-intent phrasing.
- Keep each seed single-intent and route-specific.
- Treat `future_clinical_db` as patient-record / chart / labs / medication-history requests, distinct from textbook retrieval and distinct from conversation memory.
- Prefer canonical and realistic paraphrases over broad ambiguous phrasing.

**Route boundaries:**
- `vector`: textbook-style medical knowledge answerable from the indexed corpus.
- `chitchat`: greetings, social language, meta-assistant questions.
- `web`: current or recent information requests.
- `memory`: asks about prior conversation or user-provided history in the chat.
- `literature`: explicitly requests papers, studies, reviews, trials, or PubMed-style evidence.
- `future_clinical_db`: requests about a patient chart, labs, medication list, encounters, or EHR-like facts.
