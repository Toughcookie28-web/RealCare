# Live Runtime Validation

This document defines the minimal live Docker/runtime contract for the post-stabilization phase.

## Blocking local validation path

Run these checks from the repo root:

1. Build the `app` image.
2. Build the `eval` image.
3. Start `db` and `app`.
4. Run Alembic upgrade to head inside `app` with the explicit config path `/app/alembic.ini`.
5. Verify `/health/live`.
6. Verify `/health/ready`.
7. Run `python3 scripts/preflight_rag.py` inside the `eval` container.
8. Verify writable cache and artifact paths inside the `eval` container.

The local helper for this path is:

```bash
python3 scripts/run_live_runtime_smoke.py --dry-run
```

If your machine only supports the legacy compose binary:

```bash
python3 scripts/run_live_runtime_smoke.py --compose-bin docker-compose --dry-run
```

The Alembic step is intentionally explicit so the container runtime does not depend on working-directory-based config discovery:

```bash
docker-compose exec -T app python3 -m alembic -c /app/alembic.ini upgrade head
```

The container cache/artifact contract is also explicit: compose services always use `/app/.cache/embeddings` inside the container, regardless of any host-local `EMBEDDING_CACHE_DIR` value in `.env`.
The first Docling-based ingest on a fresh cache may spend extra time downloading layout/table artifacts into `/app/.cache/embeddings/docling_artifacts`.

## Non-blocking extended shadow path

These checks are useful, but too runtime-heavy to treat as required CI gates yet:

1. Containerized `python3 scripts/reindex_pdf.py`
2. Containerized `python3 -m eval.retrieval_eval --save live-shadow`
3. Containerized `python3 -m eval.chunking_eval --save live-shadow`

The helper for this path is:

```bash
python3 scripts/run_live_rag_shadow.py --compose-bin docker-compose
```

## Artifacts to review

- `eval/retrieval_snapshots/live-shadow.json`
- `eval/chunking_snapshots/live-shadow.json`
- terminal output from `scripts/reindex_pdf.py`
- `docker compose logs app`
- `docker compose logs db`
- `docker compose logs eval`
