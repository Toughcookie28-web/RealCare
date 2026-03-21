# TODOS — MediGenius Deferred Work

Items that are intentionally deferred with context. Each item has a reason for deferral
and enough context to pick it up later.

---

## P2: Grafana Alerting Rules

**What:** Add Prometheus alerting rules for: `grounding_score < 0.5` (sustained), clarification rate > 20% over 1h, reflection retry rate > 10%.
**Why:** Alerting turns passive observability into proactive monitoring. Without it, you only notice degradation when a user complains.
**Context:** Grafana alert rule syntax is separate from dashboard panels. Needs `grafana/provisioning/alerting/` directory and alert YAML. PromQL for grounding: `histogram_quantile(0.5, medigenius_reflection_grounding_score_bucket) < 0.5`. Deferring until grounding metric baseline is established (need real data first).
**Depends on:** OTel + Tempo observability PR (adds `medigenius_reflection_grounding_score`).
**Effort:** S (human: ~half-day / CC: ~15min)

---

## P2: Frontend Trace ID Deeplink

**What:** Show a "View trace →" link in the chat UI that opens Grafana Explore pre-filtered to the current `trace_id`.
**Why:** `ChatResponse.trace_id` already exists in the API response. The frontend just needs to render it as a Grafana Explore deeplink.
**Context:** URL template: `http://localhost:3000/explore?orgId=1&left=["now-1h","now","Tempo",{"query":"<trace_id>"}]`. Chat response already includes `trace_id` field — just wire it to the UI. Small frontend change in `templates/index.html` or the React/JS frontend.
**Depends on:** OTel + Tempo observability PR (provides the trace_id → Tempo pipeline).
**Effort:** S (human: ~2h / CC: ~10min)

---

## P3: Log-Trace Correlation

**What:** Inject `trace_id` and `span_id` into structured log lines so Grafana Loki logs and Tempo traces can be correlated.
**Why:** Currently logs and traces are separate. With correlation, clicking a log line in Loki can open the corresponding Tempo trace.
**Context:** Add `trace_id` from `trace.get_current_span().get_span_context().trace_id` to the logging extra dict in structured log calls. Requires Loki in docker-compose (not currently deployed). Deferring until Loki is added.
**Depends on:** Loki container in docker-compose (separate task).
**Effort:** M (human: ~1 day / CC: ~30min)

---

## P3: Image Chunking

**What:** Support PDF pages with images — chunk images alongside text using vision models or OCR for medical diagrams, charts, and tables.
**Why:** Medical literature often includes visual information (drug mechanism diagrams, dosing charts) that text-only chunking misses.
**Context:** Plan at `docs/plans/2026-03-17-image-chunking.md`. Requires Docling or similar vision-capable PDF parser already partially in the stack (DOCLING_ARTIFACTS_DIR is in docker-compose env). High compute cost.
**Depends on:** Nothing blocking. Can be implemented independently.
**Effort:** XL (human: ~2 weeks / CC: ~2h)

---

## P3: IntentReasoner Fan-out (Compound Query Splitting)

**What:** For complex multi-part questions, split into sub-queries, run them in parallel via LangGraph `Send` API, and merge answers.
**Why:** "What is the standard treatment for hypertension, and what are the side effects of metformin?" currently runs as a single retrieval. Fan-out would retrieve better chunks for each sub-question independently.
**Context:** Plan at `docs/plans/2026-03-17-intent-reasoner-fanout.md`. Feature is off by default pending eval evidence that it improves answer quality. `is_compound` and `compound_confidence >= 0.7` both required to trigger. Risk: slower responses, harder to debug, retry logic becomes complex.
**Depends on:** Eval evidence showing compound query splitting improves quality on the eval set.
**Effort:** L (human: ~1 week / CC: ~1h)
