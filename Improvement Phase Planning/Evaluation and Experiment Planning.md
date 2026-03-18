# Evaluation and Experiment Planning

## Continuity Note
This is a merged revision that preserves your original 4-phase structure and tier detail,
while adding strict data contracts, QA gates, and phase exit criteria.

Legacy test/eval scripts are archived in `bin/legacy_tests/` by decision.
New evaluation artifacts start under `eval/golden/v1/`.

## Philosophy
The previous lexical overlap + source-string proxy is insufficient for agentic quality.
We move to a tiered golden dataset and judge framework that can evaluate behavior, routing,
retrieval quality, memory continuity, and robustness.

Framework direction: RAGAS for RAG-centric quality + custom deterministic/judge rubrics where
RAGAS does not fit (guardrails, routing, multi-turn memory behavior).

---

## Phase 1 — Golden Dataset Construction (Current Priority)

### Objective
Build a robust, auditable synthetic dataset that covers all core behaviors and is safe to use
as the baseline for all later eval, observability, and experiment loops.

### Target Size
~155-185 samples across 5 tiers.

### Storage Layout (New Baseline)
- `eval/golden/v1/tier1_guardrail.jsonl`
- `eval/golden/v1/tier2_routing.jsonl`
- `eval/golden/v1/tier3_rag.jsonl`
- `eval/golden/v1/tier4_conversation.jsonl`
- `eval/golden/v1/tier5_robustness.jsonl`
- `eval/golden/v1/golden_v1_manifest.json`
- `eval/golden/v1/qa_report.json`

### Universal Sample Schema (`schema_v1`)
```jsonc
{
  "id": "T2-0012",
  "version": "v1",
  "tier": "routing",                          // guardrail | routing | rag | conversational | robustness
  "category": "ambiguous_medical",
  "question": "Tell me about aspirin.",
  "multi_turn_context": null,                  // required list for conversational tier
  "ground_truth": "...",                      // nullable for pure block
  "ground_truth_contexts": [],                 // required for rag tier

  "expected_planned_route": "web",            // planner output expectation
  "expected_final_route": "vector",           // final route after retriever-first policy
  "expected_behavior": "answer_with_citations",
  "expected_guardrail_action": null,           // required when expected_behavior=block

  "difficulty": "adversarial",                // simple | reasoning | multi_context | adversarial
  "split": "test",                            // train | dev | test
  "tags": ["routing", "ambiguous"],

  "provenance": {
    "source_type": "llm_generated",          // human_seed | llm_generated | ragas_generated | mined_pattern
    "source_ref": "seed-routing-04",
    "generator_model": "moonshotai/kimi-k2-instruct-0905",
    "generator_prompt_hash": "sha256:...",
    "review_status": "approved",             // draft | approved | rejected
    "reviewer": "user",
    "reviewed_at": "2026-02-22T00:00:00Z"
  }
}
```

### Tier 1 — Boundary and Behavioral Tests (Guardrails)
Goal: verify true positives and false positives, not just blocking rate.

Method preserved from your plan:
1. Human seeds (15-20)
- harmful synthesis
- self-harm and crisis
- prompt injection
- diagnosis-seeking that should pivot (not block)
- legitimate sensitive medical education queries that should not be blocked
- jailbreak and roleplay attack framing
- indirect harm framing (legitimate safety wording with hidden risk)
- medical misinformation probing (must correct, not comply)
- dosage boundary ambiguity (accidental overdose vs intentional misuse)
- child safety and emergency ingestion scenarios (must escalate, not block)
- professional impersonation / authority-claim pressure

2. LLM expansion (30-40 generated + human review)
- slang, typos, urgency, indirect phrasing
- dedup + manual approval
- split into should_block and should_not_block

Target: ~50 for v1 (roughly 25/25), with option to increase to 60 if category coverage is too thin.

Expected behavior labels:
- `block` + `expected_guardrail_action=safety_refusal`
- `block` + `expected_guardrail_action=crisis_resources`
- `educational_pivot`
- `answer`

### Tier 2 — Routing Accuracy Tests
Goal: verify both planner intent and final runtime route behavior.

Routing policy update:
- obvious `chitchat`: no retriever precheck
- all other routes: retriever-first precheck
- final route may differ from planned route for ambiguous queries

Method preserved from your plan:
- hand-crafted examples per route (`chitchat`, `vector`, `web`, `literature`)
- include ambiguous edge cases explicitly

Target: ~35.

Evaluation targets:
- planned route accuracy (`expected_planned_route`)
- final route accuracy (`expected_final_route`)
- mismatch analysis (expected planner vs runtime path)

### Tier 3 — RAG Quality Tests (RAGAS Synthetic)
Goal: evaluate retrieval and grounded generation quality against real corpus.

Method preserved from your plan:
1. ingest medical PDFs
2. generate simple/reasoning/multi-context questions via RAGAS
3. human review for medical quality

Target: 40-60.

Primary metrics:
- faithfulness
- answer relevancy
- context precision
- context recall
- answer correctness

Prerequisite:
Use the same production chunking/indexing logic as runtime retrieval.

### Tier 4 — Conversational and Multi-turn Tests
Goal: evaluate coreference resolution and memory continuity.

Method preserved:
- hand-crafted conversation chains
- categories:
- coreference resolution
- fact recall
- summary continuity
- topic switching
- correction handling (user updates prior facts)
- progressive disclosure (symptoms/facts revealed over turns)
- contradictory statements
- clarification loop (assistant asks, user clarifies, system integrates)

Target: 15-20 chains.

Evaluation:
custom judge rubric (RAGAS alone is insufficient for multi-turn memory behavior).

### Tier 5 — Robustness Tests
Goal: evaluate noisy real-world input handling by rewriter + downstream pipeline.

Method preserved:
- mine realistic public patterns pre-production
- augment with LLM (typos, slang, ambiguity, panic tone)
- replace with anonymized real logs post-production when available

Target: 15-20.

---

### Phase 1 Data Pipeline (Concrete)
1. Seed authoring
- create canonical seeds per tier and category.

2. Synthetic expansion
- generate variants with prompt templates per tier.

3. Normalization + dedup
- lexical dedup + semantic near-duplicate filter (>=0.92 similarity).

4. Human review
- every non-seed sample must be approved/rejected with reason.

5. Split assignment
- initial policy: dev 20%, test 80%, train 0%.
- enforce no near-duplicate leakage across splits.

6. Freeze
- create `golden_v1_manifest.json` and dataset fingerprint.

### Phase 1 Quality Gates (Must Pass)
1. 100% schema compliance.
2. All tiers/categories non-empty.
3. Tier 1 block/non-block ratio between 40/60 and 60/40.
4. Near-duplicate rate <= 5% after dedup.
5. 100% review completion for generated samples.
6. Ambiguous routing samples explicitly tagged.
7. Manifest and QA report generated.
8. Category coverage minimums:
- Tier 1: at least 4 approved samples per category.
- Tier 4: at least 2 approved chains per category.

### Phase 1 Exit Criteria
1. Tier files + manifest + QA report exist under `eval/golden/v1/`.
2. Quality gates pass.
3. HITL approval is recorded in implementation change log.

---

## Phase 2 — LLM-as-a-Judge Evaluation Framework

### Objective
Build deterministic and judge evaluators over `golden_v1`.

### Scope
1. Deterministic checks
- Tier 1 block/allow and guardrail action checks
- Tier 2 planned/final route checks

2. RAGAS metrics
- Tier 3 only

3. Judge rubrics
- Tier 4 and Tier 5
4. CI wiring requirement
- Evaluator must read from `eval/golden/v1/` (not legacy `tests/eval/` paths).

### Exit Criteria
1. unified `eval_result.json` per run
2. judge calibration report against human-labeled subset
3. thresholds defined (initially non-blocking)

---

## Phase 3 — Observability and Latency Tracking (Grafana)

### Objective
Correlate quality outcomes with runtime behavior.

### Scope
1. eval run metadata tagging (`run_id`, model, config)
2. dashboard panels: quality trends, p95 latency, route distribution, guardrail trigger rate
3. retriever diagnostics: retrieved chunk count, confidence distribution

### Exit Criteria
1. dashboards operational
2. alert rules defined for regression and latency degradation

---

## Phase 4 — Experimentation Loop

### Objective
Run controlled experiments with reproducible decision records.

### Rules
1. one-variable-at-a-time changes
2. locked baseline per experiment batch
3. each run logs config diff and metric deltas

### Exit Criteria
1. experiment registry and comparison outputs exist
2. promotion/rollback policy documented and applied

---

## Immediate Next Actions
1. scaffold `eval/golden/v1/` directory and schema validator
2. implement Tier 1 generation + review workflow first
3. generate first QA report draft
4. then proceed to Tier 2 using retriever-first routing policy labels
