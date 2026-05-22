#!/usr/bin/env python3
"""
Chapter 7: Export Trained Model to ONNX
========================================
  python export_onnx.py --checkpoint outputs/best.pt
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models.cd_mlp import load_cd_mlp_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Export CdMLP to ONNX")
    parser.add_argument("--checkpoint", type=str, default="outputs/best.pt")
    parser.add_argument("--output", type=str, default="outputs/cd_mlp.onnx")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run train.py first.")
        sys.exit(1)

    model, ckpt = load_cd_mlp_from_checkpoint(args.checkpoint, map_location="cpu")
    export_dropout = ckpt.get("args", {}).get("dropout", 0.1)
    if export_dropout > 0:
        print(f"[INFO] Exporting with training dropout={export_dropout} (eval mode).")

    dummy = torch.randn(1, 7)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["car_params"],
        output_names=["cd"],
        dynamic_axes={
            "car_params": {0: "batch"},
            "cd": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX model saved: {args.output}")
    print(f"  Input:  car_params (batch, 7)")
    print(f"  Output: cd (batch, 1)")

    # Verify
    with torch.no_grad():
        cd_torch = model(dummy).item()
    print(f"  Torch prediction (dummy): {cd_torch:.4f}")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output)
        cd_onnx = sess.run(None, {"car_params": dummy.numpy()})[0].item()
        print(f"  ONNX prediction  (dummy): {cd_onnx:.4f}")
        diff = abs(cd_torch - cd_onnx)
        print(f"  Diff: {diff:.6e} {'OK' if diff < 1e-5 else 'WARNING'}")
    except ImportError:
        print("  (onnxruntime not installed, skip ONNX verification)")


if __name__ == "__main__":
    main()
