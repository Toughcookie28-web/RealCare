FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV EMBEDDING_CACHE_DIR=/app/.cache/embeddings

ARG INSTALL_EXTRAS=""
ARG APP_UID=1000
ARG APP_GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv ${VIRTUAL_ENV} \
    && groupadd -g ${APP_GID} appuser \
    && useradd -m -u ${APP_UID} -g ${APP_GID} appuser

COPY --chown=appuser:appuser pyproject.toml README.md /app/
COPY --chown=appuser:appuser agents /app/agents
COPY --chown=appuser:appuser api /app/api
COPY --chown=appuser:appuser alembic /app/alembic
COPY --chown=appuser:appuser alembic.ini /app/alembic.ini
COPY --chown=appuser:appuser core /app/core
COPY --chown=appuser:appuser db /app/db
COPY --chown=appuser:appuser observability /app/observability
COPY --chown=appuser:appuser tools /app/tools
COPY --chown=appuser:appuser app.py /app/app.py
RUN if [ -n "${INSTALL_EXTRAS}" ]; then \
        python3 -m pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"; \
    else \
        python3 -m pip install --no-cache-dir .; \
    fi
RUN python3 -c "import fastapi, langchain_core, langgraph; print('dependency_smoke_check_ok')"
RUN mkdir -p "${EMBEDDING_CACHE_DIR}/docling_artifacts" \
             "${EMBEDDING_CACHE_DIR}/rapidocr/models" \
             "${EMBEDDING_CACHE_DIR}/sentence_transformers" \
    && if python3 -c "import rapidocr" >/dev/null 2>&1; then \
         RAPIDOCR_MODELS_DIR=$(python3 -c "import pathlib, rapidocr; print((pathlib.Path(rapidocr.__file__).resolve().parent / 'models').as_posix())"); \
         rm -rf "${RAPIDOCR_MODELS_DIR}"; \
         ln -s "${EMBEDDING_CACHE_DIR}/rapidocr/models" "${RAPIDOCR_MODELS_DIR}"; \
       fi \
    && chown -R ${APP_UID}:${APP_GID} /app/.cache

COPY --chown=appuser:appuser .git_commit /app/.git_commit
COPY --chown=appuser:appuser eval /app/eval
COPY --chown=appuser:appuser scripts /app/scripts
COPY --chown=appuser:appuser static /app/static
COPY --chown=appuser:appuser templates /app/templates
COPY --chown=appuser:appuser data /app/data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"]

CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
