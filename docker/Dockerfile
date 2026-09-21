# CPU image: API + retrieval + evaluation. ~700 MB.
# The GPU variant (models, CUDA) is docker/Dockerfile.gpu.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

# Dependencies are installed into a venv in their own layer so a code change
# does not invalidate the (slow) dependency layer.
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps .


FROM python:3.12-slim AS runtime

# ffmpeg is required to decode webm/ogg/m4a uploads; curl is for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Never run the service as root.
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY prompts ./prompts
COPY data ./data
COPY configs ./configs

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VOICERAG_SERVICE__HOST=0.0.0.0 \
    VOICERAG_SERVICE__PORT=8000 \
    VOICERAG_RAG__STORE_PATH=/app/storage/index

RUN mkdir -p /app/storage && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Build the index on first start if the mounted volume is empty, then serve.
CMD ["sh", "-c", "[ -f \"$VOICERAG_RAG__STORE_PATH/meta.json\" ] || voicerag ingest --corpus /app/data/corpus; exec voicerag serve"]
