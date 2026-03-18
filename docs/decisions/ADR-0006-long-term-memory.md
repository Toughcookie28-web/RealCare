# ADR-0006: Long-Term User Memory Layer

## Status

Accepted

## Context

All memory was session-scoped. Users returning after closing the browser lose all context — conditions, medications, past conversations. The system always starts from zero. For a medical chatbot, this is a significant UX and clinical quality problem: a user who has already shared their age, conditions, and medications should not have to repeat this every session.

The existing Redis working-memory overlay (ADR-0004) improved within-session retention but is keyed by `session_id` and expires. It does not survive a new session.

## Decision

Add a three-layer cross-session memory system:

1. **Persistent anonymous `user_token`** — a UUID generated in the browser and stored in `localStorage`. Sent as the `X-User-Token` header on every request. Mapped to a `users` table row. No login required.

2. **`long_term_facts` table** (user-scoped) — structured facts (age, conditions, medications, allergies). Always loaded by `MemoryAgent` at turn start when a `user_id` is present. Write rule: upsert with higher confidence wins.

3. **`conversation_memories` table** (user-scoped, pgvector) — LLM-summarised episodic memories of past turns, vector-embedded. Semantically recalled at query time (top-K cosine similarity) and injected into the executor prompt as an episodic context block.

**Write path:** Runs inside the existing `BackgroundTask` after each response. No new background workers or services. Embedding latency is async from the user's perspective.

**Read path:** `MemoryAgent` loads facts (always, low latency) and searches episodic memories (cosine similarity, top-K) before the pipeline runs. Both paths are gated by `LONG_TERM_MEMORY_ENABLED=false` by default.

## Consequences

### Positive

- Users with an established identity get continuity across sessions without requiring login.
- Medical context (conditions, medications, past clinical concerns) carries forward automatically.
- Anonymous users (no `X-User-Token`) degrade gracefully — all paths skip silently when `user_id` is `None`.
- Gated by `LONG_TERM_MEMORY_ENABLED=false`; safe to ship disabled until cross-session testing is ready.
- Write path reuses the existing `BackgroundTask` pattern from ADR-0004.

### Negative

- Adds ~1 DB read (facts) + ~1 pgvector query per turn for known users with the feature enabled.
- Episodic memory quality depends on LLM summarisation quality at write time.
- Browser `localStorage` identity is not durable across device switches or incognito sessions.

## Guardrails

- `LONG_TERM_MEMORY_ENABLED=false` by default; must be explicitly opted in.
- `user_id=None` → all long-term memory paths skip silently. No exceptions.
- Confidence-gated upsert: higher confidence wins. Lower-confidence writes must not overwrite higher-confidence facts.
- Long-term memory write failures must never fail the user response (BackgroundTask isolation).
- `users`, `long_term_facts`, and `conversation_memories` tables are managed exclusively by Alembic (migration `0005_add_long_term_memory`).

## Alternatives Considered

- **Auth-based user identity:** ruled out as overkill for a practice/demo medical chatbot. Anonymous identity via `localStorage` UUID is simpler and sufficient.
- **Session-level episodic memory only:** already exists via Redis working memory; does not survive session close. Not sufficient for cross-session recall.
- **Additional background worker:** unnecessary — the existing `BackgroundTask` pattern handles the async write path without a new service.
