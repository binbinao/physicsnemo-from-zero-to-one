"""Standalone network definition for the ch03 heat-sink PINN.

Deliberately dependency-free: no Hydra / matplotlib / physicsnemo / tb_logger,
so the inference package can be shipped and deployed on its own.

The checkpoint written by ``heat_sink_train.py`` stores:

    {
      "model":  OrderedDict (plain state_dict),
      "config": {"hidden", "depth", "steps", physics + geometry ...},
      "scale":  float (input normalisation, geometry mm -> ~unit)
    }

``HeatSinkNet.forward`` expects *normalised* inputs (x/scale, y/scale).
``TemperatureSurrogate`` wraps it so that callers / the exported ONNX graph
work directly in physical millimetres.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HeatSinkNet(nn.Module):
    """Fully-connected surrogate: (x_norm, y_norm) -> T(x, y).

    ``hard_bottom=True`` (must match training) enforces the bottom Dirichlet
    BC T = T_source at y_norm = 0 exactly by construction:
        T = (1 - y_norm) * T_source + y_norm * U
    This is the same transform baked into ``heat_sink_train.HeatSinkNet``.
    """

    def __init__(self, hidden: int = 128, depth: int = 5,
                 hard_bottom: bool = False, T_source: float = 100.0,
                 activation: str = "tanh") -> None:
        super().__init__()
        act = nn.Tanh if activation == "tanh" else nn.SiLU
        layers: list[nn.Module] = [nn.Linear(2, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.hard_bottom = bool(hard_bottom)
        self.T_source = float(T_source)
        self.activation = activation

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.net(torch.cat([x, y], dim=-1))
        if self.hard_bottom:
            return (1.0 - y) * self.T_source + y * h
        return h


class TemperatureSurrogate(nn.Module):
    """Physical-units wrapper: inputs in mm, internal /scale normalisation.

    Exporting *this* module to ONNX makes the deployed graph self-contained —
    clients send mm and never need to know the internal ``scale``.
    """

    def __init__(self, net: HeatSinkNet, scale: float = 30.0) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x_mm: torch.Tensor, y_mm: torch.Tensor) -> torch.Tensor:
        return self.net(x_mm / self.scale, y_mm / self.scale)


def build_from_checkpoint(ckpt: dict) -> HeatSinkNet:
    """Rebuild the trained net from a raw ``heat_sink.pt`` checkpoint."""
    cfg = ckpt.get("config", {}) or {}
    model = HeatSinkNet(hidden=int(cfg.get("hidden", 128)),
                        depth=int(cfg.get("depth", 5)),
                        hard_bottom=bool(cfg.get("hard_bottom", False)),
                        T_source=float(cfg.get("T_source", 100.0)),
                        activation=str(cfg.get("activation", "tanh")))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def scale_from_checkpoint(ckpt: dict) -> float:
    return float(ckpt.get("scale", 30.0))
