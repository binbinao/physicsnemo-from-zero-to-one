#!/usr/bin/env python3
"""Smoke + golden-number tests for the heat-sink inference pipeline.

    # From ch03_heatsink/
    python inference/export.py --ckpt outputs/heat_sink.pt          # once
    python inference/test_smoke.py --ckpt outputs/heat_sink.pt       # engine parity
    python inference/test_smoke.py --url http://127.0.0.1:8000       # live HTTP checks

Checks:
  1. ONNX engine loads and runs (finite outputs).
  2. ONNX vs PyTorch reference agree to 1e-4 (golden parity).
  3. If the quality gate passed, physical sanity holds (bottom ~ T_source,
     temperature decreases towards the fin tip).
  4. With ``--url``: health/readiness, in-domain 200, out-of-domain 422,
     batch and field endpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from engines import OnnxEngine, TorchEngine  # noqa: E402
from domain import HeatsinkDomain  # noqa: E402

DEFAULT_ARTIFACT = os.path.join(_HERE, "artifacts", "heat_sink")


def check_engine_parity(ckpt_path: str, artifact_dir: str, tol: float = 1e-4) -> None:
    onnx_path = os.path.join(artifact_dir, "model.onnx")
    if not os.path.exists(onnx_path):
        print(f"[1] ONNX artifact missing ({onnx_path}) — run "
              f"'python inference/export.py --ckpt {ckpt_path}' first")
        sys.exit(1)

    domain = _domain_from_meta(artifact_dir)
    rng = np.random.default_rng(42)
    # Solid cross-section is ~47% of the bounding box; draw once, keep the
    # first 512 points that really are inside the heat-sink domain.
    xs = rng.uniform(domain.x_min, domain.x_max, 4096)
    ys = rng.uniform(domain.y_min, domain.y_max, 4096)
    inside = domain.contains(xs, ys)
    x, y = xs[inside][:512], ys[inside][:512]
    assert len(x) == 512, "fewer than 512 in-domain points sampled"

    onnx = OnnxEngine(artifact_dir=artifact_dir)
    out_onnx = onnx.predict(x, y).reshape(-1)
    assert np.all(np.isfinite(out_onnx)), "ONNX output contains NaN/Inf"
    print(f"[1] OK  onnx engine, 512 in-domain points, "
          f"T∈[{out_onnx.min():.3f}, {out_onnx.max():.3f}]")

    try:
        torch_engine = TorchEngine(ckpt_path=ckpt_path)
    except FileNotFoundError:
        print("[2] SKIP PyTorch parity — checkpoint not found:", ckpt_path)
        return
    out_torch = torch_engine.predict(x, y).reshape(-1)
    diff = float(np.max(np.abs(out_onnx - out_torch)))
    print(f"[2] OK? max|ONNX - PyTorch| = {diff:.3e} (tol={tol})")
    assert diff <= tol, f"engine parity failed: {diff:.3e} > {tol}"


def check_physics_sanity(ckpt_path: str, artifact_dir: str) -> None:
    meta = _read_meta(artifact_dir)
    q = meta.get("quality") or {}
    if q.get("validator_pass_all") is not True or q.get("steps_trained", 0) < 1000:
        print("[3] SKIP physics sanity — artifact did not pass the quality gate. "
              "Train until validator.py reports pass_all=true (default 3000 steps "
              "is usually NOT enough — see validator thresholds).")
        return
    tor = TorchEngine(ckpt_path=ckpt_path)
    phys = meta["physics"]
    # Bottom centre should be near the heat-source temperature.
    T_bottom = float(tor.predict(np.array([0.0]), np.array([1.0]))[0, 0])
    # Tip of the centre fin must be cooler than the base (pure conduction upward).
    y_axis = np.linspace(1.0, float(meta["geometry"]["params"]["fin_height_mm"]) + 9.9, 20)
    T_axis = tor.predict(np.full_like(y_axis, 0.0), y_axis).reshape(-1)
    assert abs(T_bottom - float(phys["T_source"])) < 20.0, \
        f"bottom temperature {T_bottom:.1f} far from T_source={phys['T_source']}"
    assert T_axis[0] >= T_axis[-1] - 1e-3, "temperature must not increase along the fin"
    print(f"[3] OK  physics sanity: T_bottom={T_bottom:.2f} °C, "
          f"T_fin_tip≈{T_axis[-1]:.2f} °C (T_source={phys['T_source']} °C)")


# ---------------------------------------------------------------------------
# Live HTTP checks
# ---------------------------------------------------------------------------
def http_json(url: str, method: str = "GET", payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check_live(url: str) -> None:
    base = url.rstrip("/")
    st, _ = http_json(f"{base}/readyz")
    assert st == 200, f"readyz returned {st}"
    st, body = http_json(f"{base}/v1/temperature", "POST",
                         {"x_mm": 0.0, "y_mm": 5.0})
    assert st == 200 and "T_c" in body, f"point predict failed: {st} {body}"
    st, body = http_json(f"{base}/v1/temperature", "POST",
                         {"x_mm": 7.0, "y_mm": 15.0})   # air gap between fins
    assert st == 422, f"out-of-domain point should be 422, got {st}: {body}"
    st, body = http_json(f"{base}/v1/temperature/batch", "POST",
                         {"points": [{"x_mm": -20.0, "y_mm": 2.0},
                                     {"x_mm": 20.0, "y_mm": 2.0}]})
    assert st == 200 and body["count"] == 2, f"batch failed: {st}"
    st, body = http_json(f"{base}/v1/field", "POST", {"nx": 60, "ny": 30})
    assert st == 200 and body["shape"] == [30, 60], f"field failed: {st}"
    print("[4] OK  live HTTP: readyz / point / 422-ood / batch / field")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read_meta(artifact_dir: str) -> dict:
    with open(os.path.join(artifact_dir, "model_meta.json"), encoding="utf-8") as f:
        return json.load(f)


def _domain_from_meta(artifact_dir: str) -> HeatsinkDomain:
    return HeatsinkDomain.from_meta(_read_meta(artifact_dir))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="outputs/heat_sink.pt")
    p.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT)
    p.add_argument("--url", default=None, help="live server base URL for HTTP checks")
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()

    if args.url:
        check_live(args.url)
        return
    check_engine_parity(args.ckpt, args.artifact_dir, tol=args.tol)
    check_physics_sanity(args.ckpt, args.artifact_dir)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
