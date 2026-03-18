# 📄 Product Requirements Document (PRD)

**Project:** MediGenius Enterprise Upgrade - Phase 2 (Intelligence, Reasoning & Guardrails)
**Target Audience:** AI Coding Agents / Developer Agents
**Context:** Upgrading the cognitive abilities of the MediGenius LangGraph system. Phase 2 introduces query rewriting, self-reflection loops, robust fallback mechanisms, and advanced domain-specific medical searches.

## 🎯 1. Objective

To enhance the agentic intelligence of the system. This includes making the LLM's responses highly personalized and medically safe (via the "Educational Pivot"), implementing advanced reasoning loops (Query Rewriting & Self-Reflection), and ensuring the system never breaks via robust self-diagnosis and fallback nodes.

## 🏗️ 2. Architectural Upgrades & Feature Specifications

### Feature 1: Query Rewriting Node (Coreference Resolution)

**Target Files:** `core/langgraph_workflow.py`, `agents/planner_agent.py`

**Requirements:**

* **Objective:** Users write poorly formatted, contextual queries (e.g., "What are its symptoms?"). The vector database cannot process "its".
* **Action:** Implement a lightweight LLM pre-processing step before routing or searching.
* **Logic:** The rewriting agent takes the user's raw query and the last 3 conversation turns, and outputs a standalone, highly optimized search query (e.g., "What are the common clinical symptoms of Type 2 Diabetes?").

### Feature 2: Enterprise Persona & Educational Pivot (Prompt Engineering)

**Target Files:** `agents/llm_agent.py`, `agents/executor_agent.py`

**Requirements:**

* **Objective:** Move away from naive prompts and cold "dead end" responses.
* **Action:** Rewrite the System Prompts using meticulous, few-shot prompting techniques.
* **The Educational Pivot:** Replace the cold "Please consult a healthcare professional" with an educational pivot.
* **Example Prompt Instruction:** "If a user asks for a diagnosis, gracefully decline but pivot to education. E.g., 'While I cannot diagnose you, clinical literature associates symptoms like X and Y with Z. It is crucial to consult a doctor. Would you like me to find standard clinical guidelines for Z?'"
* **Personalization:** Instruct the LLM to adopt an empathetic, expert clinical tone, utilizing any long-term memory traits retrieved in Phase 1.

### Feature 3: Domain-Specific Medical Search

**Target Files:** `tools/search_tools.py`, `agents/tavily_agent.py`, `agents/wikipedia_agent.py`

**Requirements:**

* **Objective:** Prevent general web searches from pulling in unverified medical blogs or dangerous misinformation.
* **Action:** Modify the Tavily search tool parameters to strictly filter queries using `include_domains` (e.g., `['nih.gov', 'mayoclinic.org', 'who.int', 'cdc.gov']`).
* **New Tool:** Create a new tool specifically for querying the PubMed API or Europe PMC API directly to fetch peer-reviewed medical paper abstracts when the user requests "research", "studies", or "literature".

### Feature 4: Self-Reflection & Critique Node (Self-RAG)

**Target Files:** `core/langgraph_workflow.py`, `agents/reflection_agent.py` (New File)

**Requirements:**

* **Objective:** Ensure the generated answer actually answers the user's query and doesn't hallucinate.
* **Action:** Add a `critique_node` immediately after the `executor_agent` generates its draft response.
* **Logic:** A smaller, fast LLM acts as a judge. It outputs a JSON: `{"is_relevant": true/false, "has_hallucinations": true/false, "feedback": "..."}`.
* **Graph Edge:** If the answer is rejected, the graph triggers a conditional edge looping back to the `retriever_agent` or `planner_agent` with the judge's feedback to try again (max 2 retries).

### Feature 5: Executor Self-Diagnosis & Fallback Edge

**Target Files:** `core/langgraph_workflow.py`, `agents/executor_agent.py`, `tools/llm_client.py`

**Requirements:**

* **Objective:** Ensure a "PRO" system never crashes or hangs if the primary LLM fails.
* **Action:** Implement rigorous try/except logic within the `executor_agent`. Catch JSON parsing errors, LLM timeouts, or API limits.
* **Graph Edge:** If the primary executor fails, trigger a `fallback_condition` edge. This edge routes the state to a `fallback_executor_agent` powered by a highly reliable, larger cloud model (e.g., GPT-4o or Claude 3.5 Sonnet) to salvage the transaction and return a graceful response to the user.

## 🛠️ 3. Implementation Steps for AI Agent

* **Step 1: Graph Refactoring:** Open `core/langgraph_workflow.py` and map out the new nodes: `Query_Rewriter_Node`, `Reflection_Node`.
* **Step 2: Implement Query Rewriting:** Build the function that injects the chat history and outputs the standalone semantic query. Update the graph state to store both `raw_query` and `optimized_query`.
* **Step 3: Prompt Overhaul:** Navigate through `llm_agent.py` and `executor_agent.py`. Replace all basic system prompts with detailed Markdown-formatted persona instructions incorporating the "Educational Pivot".
* **Step 4: Update Search Tools:** Open `tools/search_tools.py` and `agents/tavily_agent.py`. Hardcode the trusted domains array. Add a new `fetch_pubmed_abstracts` Python function.
* **Step 5: Implement Reflection Loop:** Create `agents/reflection_agent.py`. Write the LangChain/Pydantic structured output parser for the judge. In the workflow, wire the conditional edge from Reflection back to Planning/Retrieval.
* **Step 6: Fallback Logic:** Add standard Python exception handling in the executor. Create a new fallback node in the LangGraph setup that triggers specifically on Exception or `Max_Retries_Reached`.

## ✅ 4. Acceptance Criteria (DoR/DoD)

* [ ] User asks "What are its symptoms?" sequentially after asking about "Diabetes". The Query Rewriter successfully logs that it transformed the query to "What are the symptoms of Diabetes?" before searching.
* [ ] A direct request for a diagnosis (e.g., "I have a headache and fever, do I have meningitis?") results in the LLM utilizing the "Educational Pivot" instead of a flat refusal or hallucinating a diagnosis.
* [ ] Web searches triggered by the Tavily Agent log that they are strictly returning results from NIH, Mayo Clinic, CDC, or WHO.
* [ ] Draft answers that fail the Pydantic validation for the Reflection Node successfully trigger a loop back to the search/retrieval phase (verifiable in terminal logs).
* [ ] If the main LLM endpoint is manually disconnected or forced to throw a `TimeoutError`, the LangGraph successfully traverses the fallback edge and returns an error-handled or fallback-model response without crashing the application.
