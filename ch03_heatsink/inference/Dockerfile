# Heat-sink PINN surrogate — production serving image (ONNX Runtime, no PyTorch).
#
# Build from the chapter root so the .dockerignore applies:
#   cd ch03_heatsink
#   python inference/export.py --ckpt outputs/heat_sink.pt   # produce artifacts first
#   docker build -f inference/Dockerfile -t heat-sink-api:1.0.0 .
#   docker run -p 8000:8000 heat-sink-api:1.0.0
#
# Mount a newer artifact without rebuilding:
#   docker run -p 8000:8000 -v "$PWD/inference/artifacts/heat_sink:/app/artifacts/heat_sink:ro" heat-sink-api:1.0.0

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACT_DIR=/app/artifacts/heat_sink \
    ENGINE=onnx \
    PORT=8000

WORKDIR /app

# 1) Dependencies first for better layer caching.
COPY inference/requirements.txt /app/inference/requirements.txt
RUN pip install --no-cache-dir -r /app/inference/requirements.txt

# 2) Serving code + pre-exported artifact (export.py must be run before build).
COPY inference/ /app/inference/
COPY inference/artifacts/ /app/artifacts/

# 3) Non-root user, read-only artifact at runtime.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

CMD ["uvicorn", "inference.serve:app", "--host", "0.0.0.0", "--port", "8000"]
