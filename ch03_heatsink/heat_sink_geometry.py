"""
Chapter 3: Heat Sink CSG Geometry
==================================
Constructive Solid Geometry for a 2D heat sink:
  - Base plate (rectangle)
  - 3 rectangular fins on top

Pure Python geometry sampling — no PhysicsNeMo dependency.
"""

import numpy as np


class Rectangle:
    """Axis-aligned rectangle defined by (x_min, y_min, x_max, y_max)."""

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float,
                 name: str = "rect"):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.name = name

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask: True if point is inside the rectangle."""
        return ((x >= self.x_min) & (x <= self.x_max) &
                (y >= self.y_min) & (y <= self.y_max))

    def sample_interior(self, n: int) -> tuple:
        """Uniform random samples inside the rectangle."""
        x = np.random.uniform(self.x_min, self.x_max, n)
        y = np.random.uniform(self.y_min, self.y_max, n)
        return x, y

    def sample_boundary(self, n: int) -> tuple:
        """Uniform random samples on the rectangle boundary."""
        perimeter = 2 * ((self.x_max - self.x_min) + (self.y_max - self.y_min))
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        # Distribute points proportional to edge length
        n_bottom = max(1, int(n * w / perimeter))
        n_top = max(1, int(n * w / perimeter))
        n_left = max(1, int(n * h / perimeter))
        n_right = n - n_bottom - n_top - n_left

        xs, ys = [], []
        # Bottom
        xs.append(np.random.uniform(self.x_min, self.x_max, n_bottom))
        ys.append(np.full(n_bottom, self.y_min))
        # Top
        xs.append(np.random.uniform(self.x_min, self.x_max, n_top))
        ys.append(np.full(n_top, self.y_max))
        # Left
        xs.append(np.full(n_left, self.x_min))
        ys.append(np.random.uniform(self.y_min, self.y_max, n_left))
        # Right
        xs.append(np.full(n_right, self.x_max))
        ys.append(np.random.uniform(self.y_min, self.y_max, n_right))

        return np.concatenate(xs), np.concatenate(ys)

    @property
    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class HeatSinkGeometry:
    """
    2D heat sink: rectangular base plate + 3 fins.

    Default dimensions (in mm, normalized to ~unit scale):
        Base:  60 x 10  (x: -30..30, y: 0..10)
        Fins:  4 x fin_height, equally spaced on top of base

    Parameters:
        base_width, base_height, fin_width, fin_height, n_fins
    """

    def __init__(self, base_width: float = 60.0, base_height: float = 10.0,
                 fin_width: float = 4.0, fin_height: float = 20.0,
                 n_fins: int = 3):
        self.base_width = base_width
        self.base_height = base_height
        self.fin_width = fin_width
        self.fin_height = fin_height
        self.n_fins = n_fins

        # Base plate
        self.base = Rectangle(-base_width / 2, 0, base_width / 2, base_height,
                              name="base")

        # Fins (equally spaced on top of base)
        self.fins = []
        spacing = base_width / (n_fins + 1)
        for i in range(n_fins):
            cx = -base_width / 2 + spacing * (i + 1)
            fin = Rectangle(cx - fin_width / 2, base_height,
                           cx + fin_width / 2, base_height + fin_height,
                           name=f"fin_{i}")
            self.fins.append(fin)

    def sample_interior(self, n: int) -> tuple:
        """Sample interior points across the entire heat sink (base + fins)."""
        # Distribute samples proportional to area
        total_area = self.base.area + sum(f.area for f in self.fins)
        n_base = max(1, int(n * self.base.area / total_area))
        n_per_fin = max(1, (n - n_base) // self.n_fins)

        xs, ys = [], []
        xb, yb = self.base.sample_interior(n_base)
        xs.append(xb)
        ys.append(yb)

        for fin in self.fins:
            xf, yf = fin.sample_interior(n_per_fin)
            xs.append(xf)
            ys.append(yf)

        return np.concatenate(xs), np.concatenate(ys)

    def sample_bottom_bc(self, n: int) -> tuple:
        """Bottom of base plate (heat source): y = 0."""
        x = np.random.uniform(-self.base_width / 2, self.base_width / 2, n)
        y = np.zeros(n)
        return x, y

    def sample_side_bc(self, n: int) -> tuple:
        """Side walls of base (insulated / Neumann)."""
        n_half = n // 2
        y_left = np.random.uniform(0, self.base_height, n_half)
        x_left = np.full(n_half, -self.base_width / 2)
        y_right = np.random.uniform(0, self.base_height, n - n_half)
        x_right = np.full(n - n_half, self.base_width / 2)
        return (np.concatenate([x_left, x_right]),
                np.concatenate([y_left, y_right]))

    def sample_robin_bc(self, n: int) -> tuple:
        """
        Top and side surfaces of fins (convective / Robin BC).
        Returns x, y, nx, ny (outward normals).
        """
        xs, ys, nxs, nys = [], [], [], []
        n_per_fin = max(1, n // self.n_fins)

        for fin in self.fins:
            # Top of fin
            n_top = n_per_fin // 3
            xt = np.random.uniform(fin.x_min, fin.x_max, n_top)
            yt = np.full(n_top, fin.y_max)
            xs.append(xt); ys.append(yt)
            nxs.append(np.zeros(n_top)); nys.append(np.ones(n_top))

            # Left side of fin
            n_side = n_per_fin // 3
            yl = np.random.uniform(fin.y_min, fin.y_max, n_side)
            xl = np.full(n_side, fin.x_min)
            xs.append(xl); ys.append(yl)
            nxs.append(-np.ones(n_side)); nys.append(np.zeros(n_side))

            # Right side of fin
            n_right = n_per_fin - n_top - n_side
            yr = np.random.uniform(fin.y_min, fin.y_max, n_right)
            xr = np.full(n_right, fin.x_max)
            xs.append(xr); ys.append(yr)
            nxs.append(np.ones(n_right)); nys.append(np.zeros(n_right))

        return (np.concatenate(xs), np.concatenate(ys),
                np.concatenate(nxs), np.concatenate(nys))

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Check if points are inside the heat sink domain."""
        mask = self.base.contains(x, y)
        for fin in self.fins:
            mask |= fin.contains(x, y)
        return mask


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    geo = HeatSinkGeometry()
    print(f"Base area: {geo.base.area:.1f}")
    print(f"Fin area (each): {geo.fins[0].area:.1f}")

    xi, yi = geo.sample_interior(5000)
    xb, yb = geo.sample_bottom_bc(200)
    xs, ys = geo.sample_side_bc(200)
    xr, yr, nxr, nyr = geo.sample_robin_bc(300)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xi, yi, s=0.5, alpha=0.3, label="Interior")
    ax.scatter(xb, yb, s=5, c="red", label="Bottom BC")
    ax.scatter(xs, ys, s=5, c="blue", label="Side BC")
    ax.scatter(xr, yr, s=5, c="green", label="Robin BC")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Heat Sink Geometry Sampling")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    import os
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/geometry_sampling.png", dpi=150)
    print("Saved outputs/geometry_sampling.png")
