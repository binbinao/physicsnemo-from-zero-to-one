#!/usr/bin/env python3
"""Production inference API for the ch03 heat-sink PINN surrogate.

Design goals (beyond the demo server in ch07/api/app.py):

  * **Self-contained artifact** — the API is configured by ``model_meta.json``
    + ``model.onnx`` produced by ``inference/export.py``; geometry, physics and
    input units travel with the model, never hard-coded in the server.
  * **Physical input validation** — coordinates are mm; queries outside the
    model bounding box or inside air gaps are rejected with HTTP 422 instead of
    returning a number the network was never trained to produce.
  * **No-PyTorch serving** — default engine is ONNX Runtime (small container).
    ``ENGINE=torch`` is available for golden-number checks against the original
    checkpoint.
  * **Ops readiness** — /healthz (liveness), /readyz (model loaded),
    /metrics (Prometheus), request-id propagation, structured request logging,
    bounded batch/field sizes.

Run (from ``ch03_heatsink/``):

    python inference/export.py --ckpt outputs/heat_sink.pt
    uvicorn inference.serve:app --host 0.0.0.0 --port 8000

Env vars:
    ARTIFACT_DIR   inference/artifacts/heat_sink   (dir with model.onnx + model_meta.json)
    ENGINE         onnx | torch                    (torch needs CKPT_PATH)
    CKPT_PATH      ../outputs/heat_sink.pt         (only for ENGINE=torch)
    MAX_FIELD_POINTS  400000                       (guard against huge grids)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from domain import HeatsinkDomain
from engines import load_engine

# ---------------------------------------------------------------------------
# Settings (env overrides)
# ---------------------------------------------------------------------------
ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR", os.path.join(_HERE, "artifacts", "heat_sink"))
META_PATH = os.path.join(ARTIFACT_DIR, "model_meta.json")
ENGINE = os.environ.get("ENGINE", "onnx").lower()
CKPT_PATH = os.environ.get(
    "CKPT_PATH", os.path.join(os.path.dirname(_HERE), "outputs", "heat_sink.pt"))
MAX_FIELD_POINTS = int(os.environ.get("MAX_FIELD_POINTS", "400000"))

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heat_sink.api")

# ---------------------------------------------------------------------------
# Prometheus metrics (degraded gracefully)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _REQUESTS = Counter("heat_sink_http_requests_total",
                        "HTTP requests", ["method", "path", "status"])
    _PREDICT_SECONDS = Histogram("heat_sink_predict_seconds",
                                 "Engine predict() latency", buckets=(0.0001, 0.0005,
                                 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0))
    _HAS_PROM = True
except Exception:  # pragma: no cover
    _HAS_PROM = False
    _REQUESTS = _PREDICT_SECONDS = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"


# ---------------------------------------------------------------------------
# App state (loaded once at startup)
# ---------------------------------------------------------------------------
class AppState:
    meta: dict = {}
    domain: HeatsinkDomain | None = None
    engine = None
    artifact_dir: str = ARTIFACT_DIR
    ckpt_path: str = CKPT_PATH


state = AppState()


def _reject_out_of_domain(x_mm: float, y_mm: float) -> str | None:
    if state.domain is None:
        return "domain not initialised"
    return state.domain.rejection_reason(x_mm, y_mm)


def _predict(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    t0 = time.perf_counter()
    try:
        out = state.engine.predict(x_mm, y_mm)
    finally:
        if _HAS_PROM:
            _PREDICT_SECONDS.observe(time.perf_counter() - t0)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("engine returned non-finite values — model defect or "
                           "input outside the trained regime slipped through")
    return out


@asynccontextmanager
async def lifespan(_app):
    if not os.path.exists(META_PATH):
        raise RuntimeError(
            f"model metadata not found: {META_PATH} — run "
            f"'python inference/export.py --ckpt outputs/heat_sink.pt' first")
    with open(META_PATH, encoding="utf-8") as f:
        state.meta = json.load(f)
    state.domain = HeatsinkDomain.from_meta(state.meta)
    state.engine = load_engine(ENGINE, state.artifact_dir,
                               ckpt_path=state.ckpt_path)

    q = (state.meta.get("quality") or {})
    if q.get("validator_pass_all") is not True:
        log.warning("quality gate NOT passed for this artifact — serving anyway. "
                    "Production deployments should gate at the registry/CI layer "
                    "(quality=%s)", q)

    log.info("engine=%s artifact=%s model=%s params=%s quality=%s",
             state.engine.name, state.artifact_dir,
             state.meta.get("model_name"), state.meta.get("net", {}).get("n_params"),
             q.get("validator_pass_all"))
    yield


from fastapi import FastAPI, HTTPException, Request      # noqa: E402
from fastapi.responses import Response                   # noqa: E402
from pydantic import BaseModel, Field                    # noqa: E402

app = FastAPI(
    title="Heat Sink Temperature Surrogate API",
    description="PINN surrogate for steady 2D heat conduction in a finned "
                "heat sink (input mm, output degC). Self-contained artifact.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class Point(BaseModel):
    x_mm: float
    y_mm: float


class PointRequest(BaseModel):
    x_mm: float = Field(description="x coordinate in mm")
    y_mm: float = Field(description="y coordinate in mm")


class BatchRequest(BaseModel):
    points: list[Point] = Field(min_length=1, max_length=8192)


class FieldRequest(BaseModel):
    nx: int = Field(300, ge=2, le=2000, description="grid points along x")
    ny: int = Field(150, ge=2, le=2000, description="grid points along y")


class PointResponse(BaseModel):
    x_mm: float
    y_mm: float
    T_c: float
    model_version: str | None = None


# ---------------------------------------------------------------------------
# Middleware: request-id + Prometheus counters
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        if _HAS_PROM:
            _REQUESTS.labels(request.method, request.url.path, "500").inc()
        raise
    response.headers["X-Request-Id"] = rid
    if _HAS_PROM:
        _REQUESTS.labels(request.method, request.url.path,
                         str(response.status_code)).inc()
    log.info("req=%s method=%s path=%s status=%d dur_ms=%.1f",
             rid, request.method, request.url.path, response.status_code,
             (time.perf_counter() - t0) * 1e3)
    return response


# ---------------------------------------------------------------------------
# Ops endpoints
# ---------------------------------------------------------------------------
@app.get("/healthz", tags=["ops"])
def healthz():
    """Liveness — the process is up."""
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readyz():
    """Readiness — model loaded and artifact reachable."""
    if state.engine is None or state.meta is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return {"status": "ready", "engine": state.engine.name}


@app.get("/metrics", tags=["ops"])
def metrics():
    if not _HAS_PROM:
        return Response(content="# prometheus-client not installed\n",
                        media_type=CONTENT_TYPE_LATEST)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/model/info", tags=["api"])
def model_info():
    """Return the immutable artifact metadata (no weights)."""
    if not state.meta:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    m = state.meta
    return {
        "model_name": m.get("model_name"),
        "task": m.get("task"),
        "engine": state.engine.name,
        "net": m.get("net"),
        "physics": m.get("physics"),
        "geometry": m.get("geometry"),
        "inputs_spec": m.get("inputs_spec"),
        "outputs_spec": m.get("outputs_spec"),
        "quality": m.get("quality"),
        "provenance": m.get("provenance"),
    }


# ---------------------------------------------------------------------------
# Inference endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/temperature", response_model=PointResponse, tags=["api"])
def predict_point(req: PointRequest):
    reason = _reject_out_of_domain(req.x_mm, req.y_mm)
    if reason:
        raise HTTPException(status_code=422, detail=reason)
    T = float(_predict(np.array([req.x_mm], dtype=np.float32),
                       np.array([req.y_mm], dtype=np.float32)).reshape(-1)[0])
    return PointResponse(x_mm=req.x_mm, y_mm=req.y_mm, T_c=round(T, 5),
                         model_version=str(state.meta.get("schema_version", "")))


@app.post("/v1/temperature/batch", tags=["api"])
def predict_batch(req: BatchRequest):
    """Predict for many solid points at once (single engine call)."""
    xs = np.array([p.x_mm for p in req.points], dtype=np.float32)
    ys = np.array([p.y_mm for p in req.points], dtype=np.float32)
    reasons = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        reason = _reject_out_of_domain(float(x), float(y))
        if reason:
            reasons.append({"index": i, "x_mm": float(x), "y_mm": float(y),
                            "reason": reason})
    if reasons:
        raise HTTPException(status_code=422,
                            detail=f"{len(reasons)} point(s) out of domain",
                            headers=None)
    T = _predict(xs, ys).reshape(-1)
    return {
        "count": len(req.points),
        "x_mm": [round(float(v), 4) for v in xs],
        "y_mm": [round(float(v), 4) for v in ys],
        "T_c": [round(float(v), 5) for v in T],
    }


@app.post("/v1/field", tags=["api"])
def predict_field(req: FieldRequest):
    """Full temperature field on a regular grid over the model bounding box.

    Returns row-major (ny, nx) arrays; cells outside the solid heat-sink are
    NaN in ``T_c`` and False in ``solid_mask``.
    """
    if req.nx * req.ny > MAX_FIELD_POINTS:
        raise HTTPException(status_code=413,
                            detail=f"{req.nx * req.ny} grid points exceed "
                                   f"MAX_FIELD_POINTS={MAX_FIELD_POINTS}")
    if state.domain is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    d = state.domain
    xg = np.linspace(d.x_min, d.x_max, req.nx)
    yg = np.linspace(d.y_min, d.y_max, req.ny)
    X, Y = np.meshgrid(xg, yg)
    xs = X.ravel().astype(np.float32)
    ys = Y.ravel().astype(np.float32)
    solid = state.domain.contains(xs, ys)
    T = np.full(xs.shape, np.nan, dtype=np.float32)
    T[solid] = _predict(xs[solid], ys[solid]).reshape(-1)
    return {
        "shape": [req.ny, req.nx],
        "bbox_mm": {"x_min": d.x_min, "x_max": d.x_max,
                    "y_min": d.y_min, "y_max": d.y_max},
        "x_mm": [round(float(v), 4) for v in xg],
        "y_mm": [round(float(v), 4) for v in yg],
        "T_c": [None if np.isnan(v) else round(float(v), 5) for v in T],
        "solid_mask": [bool(v) for v in solid],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
