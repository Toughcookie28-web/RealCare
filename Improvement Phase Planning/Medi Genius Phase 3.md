
# 📄 Product Requirements Document (PRD)

**Project:** MediGenius Enterprise Upgrade - Phase 3 (Production Readiness, XAI & Multi-Modal)
**Target Audience:** AI Coding Agents / Developer Agents
**Context:** Final architectural phase for the MediGenius system. This phase introduces explainability, vision capabilities, input safety guardrails, user experience (streaming), and an automated evaluation pipeline.

## 🎯 1. Objective

To finalize the MediGenius agentic system for production deployment. The system must become transparent (Explainable AI/Citations), capable of processing images (Multi-modal), legally and ethically safe (Guardrails & Human-in-the-loop), and strictly measurable (RAGAS evaluation).

## 🏗️ 2. Architectural Upgrades & Feature Specifications

### Feature 1: Explainable AI (XAI) Citation Agent

**Target Files:** `agents/explanation_agent.py` (Update), `agents/executor_agent.py`

**Requirements:**

* **Objective:** Address Bullet 11. The system must provide verifiable proof for its medical claims to build user trust.
* **Action:** Integrate the Explanation Agent into the LangGraph workflow directly after the Executor Agent (or have the Executor trigger it as a tool).
* **Logic:** The agent maps the generated LLM response back to the retrieved vector chunks. It appends strict, formatted inline citations to the final output (e.g., `[Source: Medical_Book.pdf, Page 42, Section: Diabetes Symptoms]`).

### Feature 2: Input Guardrails & HIPAA Compliance

**Target Files:** `core/guardrails.py` (New File), `core/langgraph_workflow.py`

**Requirements:**

* **Objective:** Prevent the system from processing illegal, highly dangerous, or non-medical malicious prompts.
* **Action:** Implement an Input Guardrail Node as the absolute first step in the LangGraph, prior to the Planner Agent.
* **Logic:** Use a lightweight classification model (like Llama Guard) or a strict zero-shot prompt to evaluate the raw user input. If the user asks for illicit drug synthesis, self-harm instructions, or definitive fatal diagnoses, the guardrail halts the graph immediately and returns a standardized safety refusal.

### Feature 3: Multi-Modal Vision Integration

**Target Files:** `agents/vision_agent.py` (New File), `app.py`, `static/js/main.js`

**Requirements:**

* **Objective:** Allow users to upload medical images (e.g., nutritional labels, non-sensitive skin rashes, general X-ray examples for education).
* **Action:** Create a Vision Agent utilizing a multi-modal LLM (e.g., GPT-4o Vision or Claude 3.5 Sonnet).
* **UI/Backend:** Update the frontend to accept image uploads (multipart/form-data). Pass the base64 encoded image to the LangGraph state.
* **Routing:** Update the Planner Agent to detect image inputs and route the state to the `vision_route` for visual analysis before normal text synthesis.

### Feature 4: Streaming & UX Transparency

**Target Files:** `app.py`, `core/langgraph_workflow.py`, `static/js/main.js`

**Requirements:**

* **Objective:** Agentic workflows take time. The user must see exactly what the AI is "thinking" to prevent UI abandonment.
* **Action:** Implement asynchronous streaming responses (`StreamingResponse` in FastAPI/Flask).
* **Logic:** As the LangGraph transitions through nodes, yield status updates to the frontend via Server-Sent Events (SSE) or WebSockets.
* **UI Output:** The chat interface should display transient statuses: ⏳ Rewriting query... -> 📚 Searching vector database... -> 🧠 Verifying citations... -> [Final streamed text].

### Feature 5: Human-in-the-Loop (HITL) Triggers

**Target Files:** `core/langgraph_workflow.py`

**Requirements:**

* **Objective:** For highly sensitive queries, pause the system for human review.
* **Action:** Utilize LangGraph's native `interrupt_before` or `interrupt_after` functionality.
* **Logic:** If the Planner or Executor detects a "High Risk" query (e.g., medication dosage adjustments), the graph pauses its state. It requires a manual API call (simulating a doctor/admin approval dashboard) to resume the graph and send the message to the user.

### Feature 6: Automated Evaluation Pipeline (RAGAS)

**Target Files:** `tests/eval_ragas.py` (New File), `requirements.txt`

**Requirements:**

* **Objective:** We need empirical data that the RAG pipeline is "perfect."
* **Action:** Integrate the `ragas` library.
* **Logic:** Create a test suite with 20 standard medical questions and ground-truth answers. The script runs these questions through the LangGraph and scores the outputs based on:
  * **Faithfulness:** Did it hallucinate?
  * **Answer Relevance:** Did it answer the specific question?
  * **Context Precision:** Did the retriever find the right PDF chunks?

## 🛠️ 3. Implementation Steps for AI Agent

* **Step 1: Graph Security:** Create `core/guardrails.py`. Write the validation logic. Inject this node at the very start of `langgraph_workflow.py`.
* **Step 2: Explanation & Citations:** Revive `agents/explanation_agent.py`. Write prompt logic to cross-reference the draft answer with `context_chunks` and append Markdown citations.
* **Step 3: Vision Capabilities:** Add UI elements in `index.html` and `main.js` for image uploads. Create `vision_agent.py` and integrate it as a conditional node in the workflow.
* **Step 4: Streaming Implementation:** Refactor the backend API endpoint (`/chat`) to yield generator responses. Update the frontend JS to handle a readable stream and update the DOM incrementally. Add UI badges for agent state transitions.
* **Step 5: Add HITL Checkpoints:** Configure the LangGraph compilation step (e.g., `checkpointer=memory`, `interrupt_before=["sensitive_executor"]`).
* **Step 6: Evaluation Setup:** Install `ragas`, `datasets`, and `langsmith`. Write the batch evaluation script in `tests/eval_ragas.py` to be run via CLI.

## ✅ 4. Acceptance Criteria (DoR/DoD)

* [ ] Attempting to ask the system "How do I synthesize [illicit substance]?" is immediately blocked by the Guardrail Node with a standard safety message, without calling the retrieval or executor agents.
* [ ] When asking a complex medical question, the final output includes specific document citations mapped directly to the user's uploaded knowledge base (e.g., `[Medical_Book.pdf, p. 14]`).
* [ ] Uploading an image of a nutritional label with the prompt "Is this safe for a diabetic?" successfully triggers the Vision Agent, which analyzes the sugar content and responds accurately.
* [ ] The UI displays real-time agent state transitions (e.g., "Searching...", "Reflecting...") while processing, and the final text streams in character-by-character.
* [ ] Running `pytest tests/eval_ragas.py` outputs a structured evaluation matrix showing scores (0.0 to 1.0) for Faithfulness, Answer Relevance, and Context Precision.
