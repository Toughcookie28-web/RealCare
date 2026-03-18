# Route Seed Curation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reliable raw seed set for the k-NN fallback router, generate the embedded seed file, and lock the seed contract with a lightweight test.

**Architecture:** The change is data-first. We add a curated `route_seeds_raw.json`, embed it into `route_seeds.json` using the existing generator, and add a contract test to keep route coverage stable. No router logic changes are needed.

**Tech Stack:** JSON, Python, pytest, existing embedding client

---

### Task 1: Add the curated raw route seed file
- Create `data/route_seeds_raw.json`
- Include 10 medically realistic, unambiguous seed queries for each route: `vector`, `chitchat`, `web`, `memory`, `literature`, `future_clinical_db`

### Task 2: Add a route seed contract test
- Create `tests/rag/test_route_seed_contract.py`
- Validate route key set, exactly 10 non-empty seeds per route, and no duplicates within a route

### Task 3: Generate embedded seed output
- Run `python scripts/generate_route_seeds.py`
- Confirm `data/route_seeds.json` exists and has the same route coverage/counts

### Task 4: Verify and document
- Run `pytest -q tests/rag/test_knn_router.py tests/rag/test_route_seed_contract.py`
- Update `docs/changes/implementation-log.md` with the routing seed curation change
