# Multi-stage: the wheel build needs compilers, the runtime does not.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# Split the install so a code change doesn't rebuild the torch layer.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install -r requirements.txt


FROM python:3.12-slim AS runtime

# libGL and libglib are OpenCV's runtime deps; ffmpeg backs video decoding.
# opencv-python-headless still needs libglib, which trips people up.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    SVQA_BACKEND=stub

WORKDIR /app
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Non-root: the container never needs to write outside /app/data.
RUN useradd --create-home --uid 10001 svqa \
    && mkdir -p /app/data/artifacts /app/data/raw \
    && chown -R svqa:svqa /app
USER svqa

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# One worker: the model registry is per-process, so N workers means N copies
# of every resident checkpoint. Scale with replicas, not workers.
CMD ["uvicorn", "svqa.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
