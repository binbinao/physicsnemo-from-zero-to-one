"""Inference engines behind one shared ``predict(x_mm, y_mm) -> T`` contract.

Two interchangeable backends:

* ``OnnxEngine``  — default production engine. Runs on ONNX Runtime, so the
  serving container does **not** need PyTorch (small image, no CUDA required).
* ``TorchEngine`` — reference engine used for golden-number comparisons in CI
  and local validation. It reads the original ``heat_sink.pt`` checkpoint.

Heavy imports (torch / onnxruntime) happen lazily inside the factories so that
importing this module stays cheap and an ONNX-only deployment never imports
torch.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Engine(Protocol):
    """predict physical coordinates in mm and return temperatures (n, 1)."""

    name: str

    def predict(self, x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
        ...


def _to_inputs(x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_mm, dtype=np.float32).reshape(-1)
    y = np.asarray(y_mm, dtype=np.float32).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x_mm and y_mm must have the same length")
    return x, y


# ---------------------------------------------------------------------------
# ONNX Runtime engine (production default)
# ---------------------------------------------------------------------------
@dataclass
class OnnxEngine:
    artifact_dir: str
    session: object = None  # onnxruntime.InferenceSession

    name: str = "onnx"

    def __post_init__(self) -> None:
        if self.session is None:
            import onnxruntime as ort
            path = os.path.join(self.artifact_dir, "model.onnx")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"ONNX artifact not found: {path} — run inference/export.py first")
            # Threads: leave 0 -> ORT chooses; sessions are thread-safe.
            so = ort.SessionOptions()
            self.session = ort.InferenceSession(path, sess_options=so)

    def predict(self, x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
        x, y = _to_inputs(x_mm, y_mm)
        # The exported graph expects column vectors (batch, 1) per input.
        (out,) = self.session.run(
            ["T"], {"x_mm": x.reshape(-1, 1), "y_mm": y.reshape(-1, 1)})
        return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# PyTorch engine (validation / golden reference only)
# ---------------------------------------------------------------------------
@dataclass
class TorchEngine:
    ckpt_path: str
    model: object = None
    scale: float = 30.0

    name: str = "torch"

    def __post_init__(self) -> None:
        if self.model is None:
            import torch
            from net import build_from_checkpoint, scale_from_checkpoint
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.ckpt_path}")
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
            self.model = build_from_checkpoint(ckpt).eval()
            self.scale = scale_from_checkpoint(ckpt)

    def predict(self, x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
        import torch
        x, y = _to_inputs(x_mm, y_mm)
        xt = torch.from_numpy(x / self.scale).unsqueeze(1)
        yt = torch.from_numpy(y / self.scale).unsqueeze(1)
        with torch.no_grad():
            out = self.model(xt, yt).numpy()
        return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def load_engine(engine: str, artifact_dir: str, ckpt_path: str | None = None) -> Engine:
    engine = (engine or "onnx").lower()
    if engine == "onnx":
        return OnnxEngine(artifact_dir=artifact_dir)
    if engine == "torch":
        if not ckpt_path:
            raise ValueError("ENGINE=torch requires CKPT_PATH to point at heat_sink.pt")
        return TorchEngine(ckpt_path=ckpt_path)
    raise ValueError(f"Unknown engine: {engine!r} (choose 'onnx' or 'torch')")
