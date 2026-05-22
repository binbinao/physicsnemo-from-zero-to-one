#!/usr/bin/env python3
"""
Chapter 7: FastAPI Prediction Server
======================================
  uvicorn api.app:app --host 0.0.0.0 --port 8000

  # Test:
  curl -X POST http://localhost:8000/predict_cd \
    -H "Content-Type: application/json" \
    -d '{"body_length":4.5,"body_width":1.8,"body_height":1.4,"front_angle":20,"rear_angle":15,"ground_clearance":0.15,"wheel_diameter":0.65}'

If FastAPI is not installed, prints instructions.
"""

import os
import sys

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    import torch
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

if HAS_FASTAPI:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.cd_mlp import load_cd_mlp_from_checkpoint

    # ── Load model ────────────────────────────────────────
    CKPT_PATH = os.environ.get("CKPT_PATH", "outputs/best.pt")
    _model = None

    def get_model():
        global _model
        if _model is None:
            if not os.path.exists(CKPT_PATH):
                raise RuntimeError(
                    f"Checkpoint not found: {CKPT_PATH}. "
                    "Run train.py first, or set CKPT_PATH env var."
                )
            _model, _ = load_cd_mlp_from_checkpoint(CKPT_PATH, map_location="cpu")
        return _model

    # ── API Schema ────────────────────────────────────────
    class DesignRequest(BaseModel):
        body_length: float = Field(..., ge=3.5, le=5.5, description="Body length (m)")
        body_width: float = Field(..., ge=1.5, le=2.2, description="Body width (m)")
        body_height: float = Field(..., ge=1.2, le=1.8, description="Body height (m)")
        front_angle: float = Field(..., ge=10, le=40, description="Front hood angle (deg)")
        rear_angle: float = Field(..., ge=5, le=35, description="Rear angle (deg)")
        ground_clearance: float = Field(..., ge=0.1, le=0.3, description="Ground clearance (m)")
        wheel_diameter: float = Field(..., ge=0.5, le=0.8, description="Wheel diameter (m)")

    class PredictionResponse(BaseModel):
        cd: float
        status: str = "ok"

    # ── App ───────────────────────────────────────────────
    app = FastAPI(
        title="PhysicsNeMo Aero Surrogate API",
        description="Predict drag coefficient from car design parameters",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/predict_cd", response_model=PredictionResponse)
    def predict_cd(req: DesignRequest):
        model = get_model()
        x = torch.tensor([[
            req.body_length, req.body_width, req.body_height,
            req.front_angle, req.rear_angle,
            req.ground_clearance, req.wheel_diameter,
        ]], dtype=torch.float32)

        with torch.no_grad():
            cd = model(x).item()

        return PredictionResponse(cd=round(cd, 5))

    @app.post("/predict_batch")
    def predict_batch(designs: list[DesignRequest]):
        model = get_model()
        xs = []
        for d in designs:
            xs.append([
                d.body_length, d.body_width, d.body_height,
                d.front_angle, d.rear_angle,
                d.ground_clearance, d.wheel_diameter,
            ])
        x = torch.tensor(xs, dtype=torch.float32)
        with torch.no_grad():
            cds = model(x).squeeze(-1).tolist()
        return {"predictions": [round(c, 5) for c in cds]}


if __name__ == "__main__":
    if not HAS_FASTAPI:
        print("=" * 55)
        print("FastAPI not installed.")
        print()
        print("Install:  pip install fastapi uvicorn")
        print()
        print("Then run: uvicorn api.app:app --port 8000")
        print("=" * 55)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
