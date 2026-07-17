# syntax=docker/dockerfile:1

# ---- Builder: install deps into a venv so the final image doesn't carry
# build tooling (gcc, headers) that the RAG stack's compiled deps need. ----
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Runtime: slim image, non-root user, only what's needed to run. ----
FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
RUN pip install --no-cache-dir --no-deps -e .

# data/ and vector_db/ are runtime state, not baked into the image — mount
# them as a volume (docker-compose.yml does this) or point AGENTICOS_DB_PATH
# / AGENTICOS_VECTOR_DB_DIR at attached storage (e.g. an EFS mount on ECS).
RUN mkdir -p /app/data /app/vector_db && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

# Seed the DB on first boot if it doesn't exist yet, then serve the API.
# For the Streamlit UI instead, override the command with:
#   streamlit run src/agenticos/ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
CMD ["sh", "-c", "python scripts/seed_db.py --if-missing && uvicorn agenticos.api.app:app --host ${AGENTICOS_API_HOST:-0.0.0.0} --port ${AGENTICOS_API_PORT:-8000}"]
