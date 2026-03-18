# Phase 1

# 📄 Product Requirements Document (PRD)

**Project:** MediGenius Enterprise Upgrade - Phase 1 (Core Infrastructure & RAG)
**Target Audience:** AI Coding Agents / Developer Agents
**Context:** Upgrading a naive LangGraph/RAG medical chatbot into a sophisticated, production-grade agentic system.

## 🎯 1. Objective

To restructure the foundational architecture of the MediGenius repository. Phase 1 focuses on improving memory handling, implementing dynamic routing, migrating to an enterprise vector database, and upgrading to an Advanced RAG pipeline.

## 🏗️ 2. Architectural Upgrades & Feature Specifications

### Feature 1: Semantic Caching (Memory & Latency Optimization)

**Target Files:** `agents/memory_agent.py`, `tools/llm_client.py`

**Requirements:**

* Implement Semantic Caching using GPTCache or Redis.
* Before executing an LLM call or a Vector retrieval, embed the user's query and check the semantic cache for mathematical proximity (e.g., threshold > 0.95 cosine similarity).
* If a cache hit occurs, return the cached response to bypass the LangGraph workflow entirely, saving latency and token costs.

### Feature 2: Dynamic Semantic Routing (Planning Agent)

**Target Files:** `agents/planner_agent.py`, `core/langgraph_workflow.py`

**Requirements:**

* Replace static string-matching or simple IF/ELSE routing logic with an LLM-based Router or Semantic Router.
* Define distinct routes:
  * `vectorstore_route`: For medical context extracted from PDFs.
  * `web_search_route`: For current/recent medical events or literature.
  * `chitchat_route`: For standard greetings (bypasses RAG/Web).
  * Future Placeholder: `clinical_db_route` (for structured SQL data).
* The planner must output a structured JSON response (or use function calling) determining the precise route and extracting the reformulated query.

### Feature 3: Database Migration (SQLite -> PostgreSQL / pgvector)

**Target Files:** `tools/vector_store.py`, `app.py`, `chat_db/` setup.

**Requirements:**

* Deprecate local SQLite/ChromaDB.
* Migrate to PostgreSQL with the pgvector extension (e.g., via Supabase or local Docker Postgres instance).
* Create a unified database schema handling both:
  * **Relational Data:** User accounts, session metadata, chat history.
  * **Vector Data:** Document chunk embeddings.
* Implement connection pooling to handle concurrent DB requests safely.

### Feature 4: Advanced RAG Pipeline (End-to-End RAG Optimization)

**Target Files:** `tools/pdf_loader.py`, `tools/vector_store.py`, `agents/retriever_agent.py`

**Requirements:**

* **Hierarchical Chunking:** Update `pdf_loader.py` to use Parent-Child chunking (or Sentence-Window chunking). Store large chunks (parents) but embed/retrieve smaller, highly specific chunks (children).
* **Hybrid Search:** Update the retrieval logic to combine Dense Retrieval (vector embeddings) and Sparse Retrieval (BM25 for exact keyword matching, crucial for medical terminology).
* **Re-ranking:** Implement a Cross-Encoder reranker (e.g., BGE-Reranker or Cohere Rerank API). Retrieve the top 20 chunks via Hybrid Search, score them with the reranker, and pass only the top 5 to the LLM agent.

### Feature 5: Long-Term Memory Handling

**Target Files:** `agents/llm_agent.py`, `agents/memory_agent.py`, `core/state.py`

**Requirements:**

* **Summary Buffer Memory:** Stop naively fetching the "last 5 messages". Implement a system that keeps the last 3 turns exactly, but maintains an AI-generated running summary of the older conversation.
* **Fact Extraction:** Create an asynchronous task (or utilize a framework like Mem0) that reads the user's queries to extract static medical facts (e.g., "User is 45 years old", "User has a peanut allergy") and saves them to the user's relational DB profile or as long-term vector traits. Inject these traits into the system prompt for personalized answers.

## 🛠️ 3. Implementation Steps for AI Agent

* **Step 1: Dependency Updates:** Update `requirements.txt` to include `redis`, `gptcache`, `langchain-postgres`, `psycopg2-binary`, `sentence-transformers` (for local reranking) or `cohere`, and `rank_bm25`.
* **Step 2: DB Setup:** Refactor `tools/vector_store.py` to connect to PostgreSQL. Implement initialization scripts for the pgvector extension and required tables.
* **Step 3: Ingestion Pipeline Update:** Modify `tools/pdf_loader.py` to support Parent-Child Document retrieval. Ensure the vector store inserts these correctly.
* **Step 4: RAG Upgrades:** Write the Hybrid Search + Reranker pipeline in `agents/retriever_agent.py`.
* **Step 5: Planner Agent Upgrade:** Refactor `agents/planner_agent.py` to use semantic routing logic. Update the State class in `core/state.py` to support the new routing keys.
* **Step 6: Memory Upgrade:** Implement SummaryBufferMemory logic in `agents/memory_agent.py` and integrate GPTCache for semantic caching.

## ✅ 4. Acceptance Criteria (DoR/DoD)

* [ ] System successfully connects to a PostgreSQL database, stores embeddings using pgvector, and retrieves them.
* [ ] Exact keyword queries (e.g., "Acetaminophen") successfully trigger the BM25 sparse search and return relevant documents.
* [ ] The system logs show 20 chunks retrieved, which are then pruned down to 5 via the Reranker before being sent to the LLM.
* [ ] Planner Agent successfully routes a simple "Hello" to the chitchat route without triggering the VectorDB or Web Search.
* [ ] Repeated identical (or highly semantically similar) questions return a cached response in under 500ms without triggering an LLM generation call.
* [ ] Conversations exceeding 8 turns successfully maintain the core context via the automated Summary Buffer without exceeding context window limits.







# 




