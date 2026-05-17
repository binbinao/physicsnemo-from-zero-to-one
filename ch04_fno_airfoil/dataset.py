"""
Chapter 4: Synthetic Dataset Generators
=========================================
Generate toy 2D data for FNO training without external downloads.

1. Darcy flow: -∇·(a(x) ∇u) = f  on [0,1]² with Dirichlet BC
   - a(x): random piecewise-constant permeability field
   - u(x): solved via finite differences

2. Airfoil-like pressure: synthetic pressure field around a NACA-like shape
"""

import os
import numpy as np
import torch


def generate_darcy_data(n_samples: int = 200, resolution: int = 64,
                        seed: int = 42) -> dict:
    """
    Generate synthetic Darcy flow data.
    
    -∇·(a(x)∇u) = 1 on [0,1]², u = 0 on boundary.
    
    a(x) is a random piecewise-constant coefficient field.
    u(x) is solved via iterative Jacobi method.
    
    Returns:
        dict with 'a' (N, 1, H, W) and 'u' (N, 1, H, W) tensors.
    """
    rng = np.random.RandomState(seed)
    H = W = resolution
    dx = 1.0 / (H - 1)
    f_rhs = 1.0  # constant forcing

    all_a = []
    all_u = []

    for i in range(n_samples):
        # Generate random permeability field (piecewise constant on blocks)
        block_size = rng.choice([4, 8, 16])
        n_blocks = H // block_size
        a_blocks = rng.uniform(1.0, 12.0, (n_blocks, n_blocks))
        a_field = np.repeat(np.repeat(a_blocks, block_size, axis=0),
                           block_size, axis=1)[:H, :W]

        # Solve -∇·(a ∇u) = f via Jacobi iteration
        u = np.zeros((H, W))
        for _ in range(500):  # Jacobi iterations
            u_old = u.copy()
            # Interior points
            for ix in range(1, H - 1):
                for iy in range(1, W - 1):
                    # 5-point stencil with variable coefficient
                    a_c = a_field[ix, iy]
                    neighbors = (a_c * (u_old[ix+1, iy] + u_old[ix-1, iy] +
                                       u_old[ix, iy+1] + u_old[ix, iy-1]))
                    u[ix, iy] = (neighbors + f_rhs * dx**2) / (4.0 * a_c)

        all_a.append(a_field)
        all_u.append(u)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_samples} Darcy samples")

    return {
        'a': torch.tensor(np.array(all_a), dtype=torch.float32).unsqueeze(1),
        'u': torch.tensor(np.array(all_u), dtype=torch.float32).unsqueeze(1),
    }


def generate_darcy_data_fast(n_samples: int = 200, resolution: int = 64,
                              seed: int = 42) -> dict:
    """
    Fast vectorized Darcy data generation using numpy.
    Uses Gaussian random field for permeability + Jacobi solver.
    """
    rng = np.random.RandomState(seed)
    H = W = resolution
    dx = 1.0 / (H - 1)

    all_a = np.zeros((n_samples, H, W), dtype=np.float32)
    all_u = np.zeros((n_samples, H, W), dtype=np.float32)

    for i in range(n_samples):
        # Smooth random permeability via low-freq Fourier
        noise = rng.randn(H, W)
        # Low-pass filter
        freq_x = np.fft.fftfreq(H)
        freq_y = np.fft.fftfreq(W)
        FX, FY = np.meshgrid(freq_x, freq_y, indexing='ij')
        power = np.exp(-50 * (FX**2 + FY**2))
        a_field = np.real(np.fft.ifft2(np.fft.fft2(noise) * power))
        a_field = np.exp(a_field)  # ensure positive
        a_field = np.clip(a_field, 0.5, 20.0)

        # Jacobi solve
        u = np.zeros((H, W), dtype=np.float64)
        for _ in range(300):
            u_new = np.zeros_like(u)
            u_new[1:-1, 1:-1] = (
                a_field[1:-1, 1:-1] * (
                    u[2:, 1:-1] + u[:-2, 1:-1] +
                    u[1:-1, 2:] + u[1:-1, :-2]
                ) + dx**2
            ) / (4.0 * a_field[1:-1, 1:-1])
            u = u_new

        all_a[i] = a_field.astype(np.float32)
        all_u[i] = u.astype(np.float32)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_samples} Darcy samples (fast)")

    return {
        'a': torch.tensor(all_a).unsqueeze(1),
        'u': torch.tensor(all_u).unsqueeze(1),
    }


def generate_airfoil_pressure(n_samples: int = 200, resolution: int = 64,
                               seed: int = 42) -> dict:
    """
    Generate synthetic pressure-like fields around a parametric airfoil.
    
    This is a simplified synthetic model — NOT real CFD data.
    The "airfoil" is parameterized by thickness and camber.
    The "pressure" is a synthetic potential-flow-inspired field.
    
    Returns:
        dict with 'params' (N, 2), 'pressure' (N, 1, H, W)
    """
    rng = np.random.RandomState(seed)
    H = W = resolution

    x_grid = np.linspace(-1, 2, W)
    y_grid = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x_grid, y_grid)

    all_params = []
    all_pressure = []

    for i in range(n_samples):
        thickness = rng.uniform(0.06, 0.20)
        camber = rng.uniform(-0.05, 0.05)

        # NACA-like half-thickness distribution
        # yt = 5t [0.2969√x - 0.126x - 0.3516x² + 0.2843x³ - 0.1015x⁴]
        x_chord = np.clip(X, 0, 1)
        yt = 5 * thickness * (0.2969 * np.sqrt(x_chord + 1e-8) -
                               0.126 * x_chord -
                               0.3516 * x_chord**2 +
                               0.2843 * x_chord**3 -
                               0.1015 * x_chord**4)
        y_camber = camber * x_chord * (1 - x_chord) * 4

        # Distance from airfoil surface (approximate)
        dist_upper = Y - (y_camber + yt)
        dist_lower = -(Y - (y_camber - yt))
        dist = np.minimum(np.abs(dist_upper), np.abs(dist_lower))
        inside = (X >= 0) & (X <= 1) & (np.abs(Y - y_camber) < yt)

        # Synthetic pressure field (potential flow inspired)
        r = np.sqrt((X - 0.25)**2 + (Y - camber)**2) + 0.01
        pressure = 1.0 - (thickness / r)**2
        pressure = np.where(inside, 0.0, pressure)
        # Add some variation
        pressure += 0.1 * np.sin(2 * np.pi * X) * np.exp(-Y**2)

        all_params.append([thickness, camber])
        all_pressure.append(pressure)

    return {
        'params': torch.tensor(np.array(all_params), dtype=torch.float32),
        'pressure': torch.tensor(np.array(all_pressure),
                                  dtype=torch.float32).unsqueeze(1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["darcy", "airfoil", "both"],
                        default="both")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.type in ("darcy", "both"):
        print("Generating Darcy flow data...")
        data = generate_darcy_data_fast(args.n_samples, args.resolution)
        path = os.path.join(args.out_dir, "darcy_data.pt")
        torch.save(data, path)
        print(f"Saved {path}: a={list(data['a'].shape)}, u={list(data['u'].shape)}")

    if args.type in ("airfoil", "both"):
        print("Generating airfoil pressure data...")
        data = generate_airfoil_pressure(args.n_samples, args.resolution)
        path = os.path.join(args.out_dir, "airfoil_data.pt")
        torch.save(data, path)
        print(f"Saved {path}: params={list(data['params'].shape)}, "
              f"pressure={list(data['pressure'].shape)}")

    print("Done!")
