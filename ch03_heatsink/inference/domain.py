"""Input-domain policy for the heat-sink temperature model.

The PINN is trained only inside the *solid* cross-section of the heat sink
(base plate + fins). A query point outside the solid (e.g. in an air gap or
outside the part) has no physical meaning for this model, so a production API
must reject it instead of silently returning a number the network was never
trained to produce.

Self-contained mirror of the CSG geometry in ``heat_sink_geometry.py`` so the
inference package can be deployed without importing the tutorial codebase.
All parameters are recorded into ``model_meta.json`` at export time and the
domain is rebuilt from that metadata (no hard-coded drift).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Rectangle:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    name: str = "rect"

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ((x >= self.x_min) & (x <= self.x_max) &
                (y >= self.y_min) & (y <= self.y_max))


@dataclass(frozen=True)
class HeatsinkDomain:
    """2D heat-sink solid domain: base plate + ``n_fins`` fins on top."""

    base_width: float = 60.0
    base_height: float = 10.0
    fin_width: float = 4.0
    fin_height: float = 20.0
    n_fins: int = 3

    # -- helpers ---------------------------------------------------------
    @classmethod
    def from_meta(cls, meta: dict) -> "HeatsinkDomain":
        g = (meta.get("geometry") or {}).get("params") or {}
        return cls(
            base_width=float(g.get("base_width_mm", 60.0)),
            base_height=float(g.get("base_height_mm", 10.0)),
            fin_width=float(g.get("fin_width_mm", 4.0)),
            fin_height=float(g.get("fin_height_mm", 20.0)),
            n_fins=int(g.get("n_fins", 3)),
        )

    def to_meta(self) -> dict:
        return {
            "family": "heatsink_2d",
            "params": {
                "base_width_mm": self.base_width,
                "base_height_mm": self.base_height,
                "fin_width_mm": self.fin_width,
                "fin_height_mm": self.fin_height,
                "n_fins": self.n_fins,
            },
            "bbox_mm": {
                "x_min": self.x_min, "x_max": self.x_max,
                "y_min": self.y_min, "y_max": self.y_max,
            },
        }

    @property
    def x_min(self) -> float: return -self.base_width / 2.0
    @property
    def x_max(self) -> float: return self.base_width / 2.0
    @property
    def y_min(self) -> float: return 0.0
    @property
    def y_max(self) -> float: return self.base_height + self.fin_height

    def _rects(self) -> Iterable[Rectangle]:
        yield Rectangle(self.x_min, 0.0, self.x_max, self.base_height, name="base")
        spacing = self.base_width / (self.n_fins + 1)
        for i in range(self.n_fins):
            cx = self.x_min + spacing * (i + 1)
            yield Rectangle(cx - self.fin_width / 2.0, self.base_height,
                            cx + self.fin_width / 2.0, self.y_max, name=f"fin_{i}")

    def contains(self, x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
        """Boolean mask: True for points inside the solid heat-sink cross-section."""
        x = np.asarray(x_mm, dtype=float)
        y = np.asarray(y_mm, dtype=float)
        mask = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=bool)
        for r in self._rects():
            mask |= r.contains(x, y)
        return mask

    def rejection_reason(self, x_mm: float, y_mm: float) -> str | None:
        """Return a human-readable reason if the point must be rejected."""
        if not (self.x_min <= x_mm <= self.x_max and self.y_min <= y_mm <= self.y_max):
            return (f"point ({x_mm:.2f}, {y_mm:.2f}) mm is outside the model bounding "
                    f"box x∈[{self.x_min},{self.x_max}], y∈[{self.y_min},{self.y_max}]")
        if not bool(self.contains(np.array([x_mm]), np.array([y_mm]))[0]):
            return (f"point ({x_mm:.2f}, {y_mm:.2f}) mm lies in an air gap or outside "
                    f"the solid heat-sink cross-section; temperature is undefined there")
        return None
