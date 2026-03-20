---
# ADR-0007: HITL Proactive Clarification

**Status:** Accepted

**Context:**
The old HITL system was a reviewer-facing approval queue (deleted in this sprint). For a user-facing medical chatbot, the more useful form of human-in-the-loop is asking the user to clarify ambiguous intent before generating a wrong answer. A query like "what should I take?" is medically dangerous to answer without knowing the context.

**Decision:**
When `QueryRewriterAgent` detects `intent_confidence < 0.4` and `needs_clarification=True`, `PlannerAgent` sets `route='clarify'`. The workflow short-circuits to `ExecutorAgent` which returns the clarification question as the response. No retrieval, no DB storage. The next user message re-enters the pipeline normally.

**Trigger conditions:**
- `HITL_CLARIFICATION_ENABLED=true` (default)
- `intent_confidence < HITL_CLARIFICATION_CONFIDENCE_THRESHOLD` (default 0.4)
- `needs_clarification=True` set by rewriter

**Consequences:**
- Ambiguous medical queries get clarified instead of answered incorrectly
- No retrieval or storage overhead for clarification turns
- Fully conversational — no UI changes needed, response looks like a normal message
- Feature-flaggable: set `HITL_CLARIFICATION_ENABLED=false` to disable

**Alternatives considered:**
- Always answer and hope: bad for medical context where wrong assumptions can be harmful
- Threshold-based keyword detection: brittle, hard to maintain
---
