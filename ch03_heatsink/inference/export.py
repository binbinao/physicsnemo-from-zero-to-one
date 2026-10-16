#!/usr/bin/env python3
"""Package a trained ``heat_sink.pt`` into a deployable model artifact.

    python inference/export.py --ckpt outputs/heat_sink.pt          # from ch03_heatsink/
    python inference/export.py --ckpt ... --require-quality         # fail unless quality gate passes

Produces (default ``inference/artifacts/heat_sink/``):

    model.onnx       self-contained graph — inputs in *physical mm* (the internal
                     /scale normalisation is baked into the graph), dynamic batch.
    model_meta.json  machine-readable provenance: schema version, physics &
                     geometry parameters, input domain, training summary,
                     quality-gate result, sha256 of the source checkpoint.

Why export to ONNX?
  * The serving container no longer needs PyTorch (~100 MB image instead of ~2 GB).
  * One self-describing file is easy to version, hash, and move between
    environments (dev -> staging -> prod) or into Triton / other runners.
  * The ``model_meta.json`` lets the API validate inputs against the exact
    geometry & physics the model was trained on.

Quality gate (production rule of thumb)
  -------------------------------------
  Deployment must only see checkpoints that pass ``validator.py``
  (``pass_all: true`` in ``validation_report.json``). This tool *records* the
  gate result in the metadata and warns loudly on failure; pass
  ``--require-quality`` in CI to make a failing model an error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from net import TemperatureSurrogate, build_from_checkpoint, scale_from_checkpoint  # noqa: E402
from domain import HeatsinkDomain  # noqa: E402

META_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_DIR = os.path.join(_HERE, "artifacts", "heat_sink")

if sys.platform == "win32":
    # Windows 控制台默认 GBK，torch.onnx 导出日志中的 ✅ 等符号会触发 UnicodeEncodeError
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
def _quality_summary(ckpt_path: str, ckpt: dict) -> dict | None:
    """Summarise training quality from the adjacent validation_report.json (if any)."""
    rep_path = os.path.join(os.path.dirname(ckpt_path), "validation_report.json")
    report = None
    if os.path.exists(rep_path):
        try:
            with open(rep_path, encoding="utf-8") as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            report = None
    cfg = ckpt.get("config", {}) or {}
    history = ckpt.get("history", {}) or {}
    last_total = None
    for k in ("loss", "total"):
        if k in history and history[k]:
            last_total = float(history[k][-1])
            break
    summary = {
        "steps_trained": int(cfg.get("steps", 0)),
        "last_total_loss": last_total,
        "validator_report_present": report is not None,
    }
    if report:
        summary["validator_pass_all"] = bool(report.get("pass_all"))
        summary["validator"] = {k: report[k] for k in (
            "pde_residual_rms", "robin_residual_rms", "T_min", "T_max",
            "fin_tip_1d_bound", "pass_pde", "pass_robin",
            "pass_temperature_range") if k in report}
    return summary


def _check_quality_gate(ckpt_path: str, ckpt: dict, require: bool) -> dict | None:
    summary = _quality_summary(ckpt_path, ckpt)
    if not summary:
        print("[quality] no validation_report.json found next to the checkpoint — "
              "cannot run the quality gate.")
        if require:
            sys.exit(f"ERROR: --require-quality set but no validation_report.json "
                     f"next to {ckpt_path}")
        return None

    passed = (summary.get("validator_pass_all") is True and
              summary.get("steps_trained", 0) >= 1000)
    status = "PASS" if passed else "FAIL"
    print(f"[quality] gate = {status}  (steps={summary.get('steps_trained')}, "
          f"validator_pass_all={summary.get('validator_pass_all')}, "
          f"last_total_loss={summary.get('last_total_loss')})")
    if not passed:
        msg = ("model did NOT pass the quality gate — deploying it would serve "
               "physically meaningless temperatures. Train until validator.py "
               "reports pass_all=true (default 3000 steps is usually not enough).")
        if require:
            sys.exit(f"ERROR: {msg}")
        print(f"[quality] WARNING: {msg}")
    return summary


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def export_onnx(ckpt_path: str, artifact_dir: str, require_quality: bool = False) -> str:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" not in ckpt:
        sys.exit(f"ERROR: {ckpt_path} has no 'model' state_dict — not a ch03 heat_sink.pt")
    cfg = ckpt.get("config", {}) or {}
    scale = scale_from_checkpoint(ckpt)

    net = build_from_checkpoint(ckpt)
    n_params = sum(v.numel() for v in net.state_dict().values())

    # Geometry / physics recorded into metadata (single source of truth for the API).
    domain = HeatsinkDomain(
        fin_height=float(cfg.get("fin_height", 20.0)),
        n_fins=int(cfg.get("n_fins", 3)),
    )
    physics = {k: cfg.get(k) for k in ("k", "Q", "T_source", "h_conv", "T_inf")}

    os.makedirs(artifact_dir, exist_ok=True)
    onnx_path = os.path.join(artifact_dir, "model.onnx")

    # Self-contained graph: input physical mm, normalise inside.
    # Example inputs are column vectors (batch, 1) — the network expects each
    # feature on the last axis (forward concatenates x and y along dim=-1).
    surrogate = TemperatureSurrogate(net, scale=scale).eval()
    dummy = (torch.ones(1, 1, dtype=torch.float32),
             torch.ones(1, 1, dtype=torch.float32))
    torch.onnx.export(
        surrogate, dummy, onnx_path,
        input_names=["x_mm", "y_mm"],
        output_names=["T"],
        dynamic_axes={"x_mm": {0: "batch"}, "y_mm": {0: "batch"}, "T": {0: "batch"}},
        opset_version=18,
    )
    print(f"[export] saved {onnx_path}")

    quality = _check_quality_gate(ckpt_path, ckpt, require=require_quality)

    activation = str(cfg.get("activation", "tanh"))
    family = f"mlp_{activation}"
    if cfg.get("hard_bottom"):
        family += "_hard_bottom"
    if int(cfg.get("fourier_maps", 0) or 0) > 0:
        family += f"_fourier{int(cfg['fourier_maps'])}"

    meta = {
        "schema_version": META_SCHEMA_VERSION,
        "model_name": "heat_sink_2d_temperature",
        "task": "steady_2d_heat_conduction_solid",
        "engine": "onnx",
        "artifact_file": os.path.basename(onnx_path),
        "input_units": "mm",
        "net": {
            "family": family,
            "hidden": int(cfg.get("hidden", 128)),
            "depth": int(cfg.get("depth", 5)),
            "n_params": int(n_params),
            "input_normalisation_scale": float(scale),
        },
        "inputs_spec": [
            {"name": "x_mm", "unit": "mm", "description": "x in the heat-sink cross-section"},
            {"name": "y_mm", "unit": "mm", "description": "y in the heat-sink cross-section"},
        ],
        "outputs_spec": [
            {"name": "T", "unit": "degC", "description": "steady-state temperature"},
        ],
        "physics": physics,
        "geometry": domain.to_meta(),
        "quality": quality,
        "provenance": {
            "source_ckpt": os.path.basename(ckpt_path),
            "source_ckpt_sha256": sha256_file(ckpt_path),
            "export_tool": "ch03_heatsink/inference/export.py",
        },
    }
    meta_path = os.path.join(artifact_dir, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[export] saved {meta_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# Verify against the PyTorch reference (golden numbers)
# ---------------------------------------------------------------------------
def verify_onnx(ckpt_path: str, onnx_path: str, tol: float = 1e-4) -> None:
    try:
        import torch
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed — skip (pip install onnxruntime)")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    torch_model = build_from_checkpoint(ckpt)
    scale = scale_from_checkpoint(ckpt)

    rng = np.random.default_rng(0)
    x = rng.uniform(-30, 30, 512)
    y = rng.uniform(0, 30, 512)

    with torch.no_grad():
        ref = torch_model(
            torch.from_numpy(x / scale).float().unsqueeze(1),
            torch.from_numpy(y / scale).float().unsqueeze(1),
        ).numpy().reshape(-1)

    sess = ort.InferenceSession(onnx_path)
    (out,) = sess.run(
        ["T"],
        {"x_mm": x.astype(np.float32).reshape(-1, 1),
         "y_mm": y.astype(np.float32).reshape(-1, 1)})
    out = out.reshape(-1)

    diff = np.max(np.abs(out - ref))
    print(f"[verify] max |ONNX - PyTorch| over 512 points = {diff:.3e}")
    if diff > tol:
        print("[verify] WARNING: ONNX/PyTorch mismatch above tolerance "
              f"({tol:.0e}) — investigate before deploying.")
    else:
        print("[verify] OK — ONNX matches PyTorch reference.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="outputs/heat_sink.pt",
                        help="ch03 checkpoint (outputs/heat_sink.pt)")
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR,
                        help="output artifact directory")
    parser.add_argument("--require-quality", action="store_true",
                        help="exit(1) unless the quality gate passes (CI)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="do not compare ONNX against the PyTorch reference")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"ERROR: checkpoint not found: {args.ckpt}")
    onnx_path = export_onnx(args.ckpt, args.artifact_dir,
                            require_quality=args.require_quality)
    if not args.skip_verify:
        verify_onnx(args.ckpt, onnx_path)


if __name__ == "__main__":
    main()
