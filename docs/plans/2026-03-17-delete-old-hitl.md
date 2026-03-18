# Delete Old HITL Approval Queue Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the reviewer-approval-queue HITL system (dangerous query flagging) which is unused and will be replaced by the lightweight proactive clarification HITL.

**Architecture:** Delete 4 files, remove `HitlReviewModel` from db/models.py, remove `enable_hitl_interrupts` from settings, unregister the API router. No behavior change to the running workflow — this code path was never wired into `langgraph_workflow.py`.

**Tech Stack:** Python, FastAPI, SQLAlchemy, Alembic

---

### Task 1: Find all references to old HITL

**Files:**
- Read: `core/hitl.py`, `api/routes/hitl.py`, `db/models.py`, `core/settings.py`, `app.py`

**Step 1: Search for all HITL references**

Run: `grep -rn "hitl\|HitlReview\|enable_hitl" --include="*.py" /home/tough/medical_chatbot/MediGenius/ | grep -v ".pyc"`

Expected: hits in `core/hitl.py`, `api/routes/hitl.py`, `db/models.py`, `core/settings.py`, and wherever the router is registered in `app.py` or similar.

**Step 2: Check for tests**

Run: `find tests/ -name "*hitl*" -o -name "*HITL*"`

Delete any test files found.

---

### Task 2: Delete core HITL files

**Files:**
- Delete: `core/hitl.py`
- Delete: `api/routes/hitl.py`

**Step 1: Delete the files**

```bash
rm core/hitl.py
rm api/routes/hitl.py
```

**Step 2: Verify gone**

Run: `ls core/hitl.py api/routes/hitl.py 2>&1`
Expected: "No such file"

**Step 3: Commit**

```bash
git rm core/hitl.py api/routes/hitl.py
git commit -m "chore: delete old HITL approval queue files"
```

---

### Task 3: Remove HitlReviewModel from db/models.py

**Files:**
- Modify: `db/models.py`

**Step 1: Remove the model class**

Delete the entire `HitlReviewModel` class (lines ~59–69 in current file).

**Step 2: Verify no remaining references**

Run: `grep -n "HitlReview\|hitl_review" db/models.py db/repositories.py`
Expected: no matches

**Step 3: Commit**

```bash
git add db/models.py
git commit -m "chore: remove HitlReviewModel from db/models"
```

---

### Task 4: Remove settings field and router registration

**Files:**
- Modify: `core/settings.py` — remove `enable_hitl_interrupts`
- Modify: `app.py` — unregister `hitl` router

**Step 1: Remove from settings**

In `core/settings.py`, delete:
```python
enable_hitl_interrupts: bool = Field(default=True, alias='ENABLE_HITL_INTERRUPTS')
```

**Step 2: Unregister router in app.py**

Find where `hitl` router is included (likely `app.include_router(hitl_router)`) and remove it along with the import.

**Step 3: Verify app still starts**

Run: `python -c "from app import app; print('ok')"`
Expected: `ok`

**Step 4: Run tests to confirm nothing broke**

Run: `pytest tests/ -x -q 2>&1 | tail -20`
Expected: no new failures

**Step 5: Commit**

```bash
git add core/settings.py app.py
git commit -m "chore: remove enable_hitl_interrupts setting and router registration"
```

---

### Task 5: Check Alembic migrations (if DB migration needed)

**Files:**
- Check: `alembic/versions/`

If a migration exists that creates `hitl_reviews` table, add a new migration to drop it. If the table doesn't exist in the live DB, skip.

```bash
# Check if migration exists
grep -r "hitl_reviews" alembic/
```

If found, create drop migration:
```bash
alembic revision --autogenerate -m "drop hitl_reviews table"
```

Review the generated migration and run:
```bash
alembic upgrade head
```

**Commit:**
```bash
git add alembic/
git commit -m "chore: migration to drop hitl_reviews table"
```
