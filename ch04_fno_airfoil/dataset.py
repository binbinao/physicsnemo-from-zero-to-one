"""
Chapter 4: Synthetic Dataset Generators
=========================================
Generate toy 2D data for FNO training without external downloads.

1. Darcy flow: -∇·(a(x) ∇u) = f  on [0,1]² with Dirichlet BC
   - a(x): random smooth permeability field (log-normal via Fourier)
   - u(x): solved via direct sparse solver with conservative flux discretization

2. Airfoil-like pressure: synthetic pressure field around a NACA-like shape
"""

import os
import numpy as np
import torch

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
except ImportError:
    raise ImportError(
        "scipy is required for Darcy data generation. "
        "Install it with: pip install scipy"
    )


def generate_darcy_data_fast(n_samples: int = 200, resolution: int = 64,
                              seed: int = 42) -> dict:
    """
    Generate synthetic Darcy flow data using direct sparse solve.

    PDE: -∇·(a(x) ∇u) = 1 on [0,1]², u = 0 on boundary.

    Uses conservative flux discretization with half-point averaged
    coefficients (same as ch05_darcy_hybrid/darcy_residual.py) and
    scipy.sparse.linalg.spsolve for exact direct solution.

    Discretization:
        a_{i+1/2,j} = (a[i+1,j] + a[i,j]) / 2
        flux_x = a_{i+1/2,j}*(u[i+1,j]-u[i,j]) - a_{i-1/2,j}*(u[i,j]-u[i-1,j])
        (same for y direction)

    Args:
        n_samples: number of data pairs to generate
        resolution: grid resolution (H = W = resolution)
        seed: random seed for reproducibility

    Returns:
        dict with 'a' (N, 1, H, W) and 'u' (N, 1, H, W) tensors.
    """
    rng = np.random.RandomState(seed)
    H = W = resolution
    dx = 1.0 / (H - 1)
    f_rhs = 1.0  # constant forcing

    all_a = np.zeros((n_samples, H, W), dtype=np.float32)
    all_u = np.zeros((n_samples, H, W), dtype=np.float32)

    # Precompute grid topology (same for all samples)
    N = H * W
    idx = np.arange(N).reshape(H, W)

    # Boundary mask
    is_boundary = np.zeros((H, W), dtype=bool)
    is_boundary[0, :] = True
    is_boundary[-1, :] = True
    is_boundary[:, 0] = True
    is_boundary[:, -1] = True

    # Interior indices
    interior = ~is_boundary
    int_i, int_j = np.where(interior)
    int_k = idx[int_i, int_j]

    # Neighbor linear indices for interior nodes
    k_ip = idx[int_i + 1, int_j]
    k_im = idx[int_i - 1, int_j]
    k_jp = idx[int_i, int_j + 1]
    k_jm = idx[int_i, int_j - 1]

    # Boundary linear indices
    bnd_k = idx[is_boundary]
    n_bnd = len(bnd_k)
    n_int = len(int_k)

    # RHS vector (constant across samples since f=1, boundary=0)
    rhs = np.zeros(N, dtype=np.float64)
    rhs[int_k] = f_rhs

    for i in range(n_samples):
        # Smooth random permeability via low-freq Fourier filtering
        noise = rng.randn(H, W)
        freq_x = np.fft.fftfreq(H)
        freq_y = np.fft.fftfreq(W)
        FX, FY = np.meshgrid(freq_x, freq_y, indexing='ij')
        power = np.exp(-50.0 * (FX**2 + FY**2))
        a_field = np.real(np.fft.ifft2(np.fft.fft2(noise) * power))
        a_field = np.exp(a_field)  # ensure positive
        a_field = np.clip(a_field, 0.5, 20.0)

        # Half-point permeability averages (conservative discretization)
        a_ip = (a_field[int_i + 1, int_j] + a_field[int_i, int_j]) / 2.0
        a_im = (a_field[int_i - 1, int_j] + a_field[int_i, int_j]) / 2.0
        a_jp = (a_field[int_i, int_j + 1] + a_field[int_i, int_j]) / 2.0
        a_jm = (a_field[int_i, int_j - 1] + a_field[int_i, int_j]) / 2.0

        # Diagonal: sum of all half-point coefficients
        diag_int = (a_ip + a_im + a_jp + a_jm) / dx**2

        # Assemble sparse matrix in COO format
        rows = np.concatenate([bnd_k, int_k, int_k, int_k, int_k, int_k])
        cols = np.concatenate([bnd_k, int_k, k_ip, k_im, k_jp, k_jm])
        vals = np.concatenate([
            np.ones(n_bnd, dtype=np.float64),
            diag_int,
            -a_ip / dx**2,
            -a_im / dx**2,
            -a_jp / dx**2,
            -a_jm / dx**2,
        ])

        A_mat = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        u_flat = spsolve(A_mat, rhs)
        u = u_flat.reshape(H, W)

        all_a[i] = a_field.astype(np.float32)
        all_u[i] = u.astype(np.float32)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_samples} Darcy samples")

    return {
        'a': torch.tensor(all_a).unsqueeze(1),
        'u': torch.tensor(all_u).unsqueeze(1),
    }


def generate_darcy_data(n_samples: int = 200, resolution: int = 64,
                        seed: int = 42) -> dict:
    """
    Generate synthetic Darcy flow data.

    -∇·(a(x)∇u) = 1 on [0,1]², u = 0 on boundary.

    This is an alias for generate_darcy_data_fast() which uses a direct
    sparse solver with conservative flux discretization.

    Args:
        n_samples: number of data pairs to generate
        resolution: grid resolution (H = W = resolution)
        seed: random seed for reproducibility

    Returns:
        dict with 'a' (N, 1, H, W) and 'u' (N, 1, H, W) tensors.
    """
    return generate_darcy_data_fast(n_samples, resolution, seed)


def generate_airfoil_pressure(n_samples: int = 200, resolution: int = 64,
                               seed: int = 42) -> dict:
    """
    Generate synthetic pressure-like fields around a parametric airfoil.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║ WARNING (Issue #140):                                              ║
    ║ The pressure field computed here uses a CYLINDER potential-flow    ║
    ║ formula (1 - (R/r)^2), NOT real airfoil CFD data. The airfoil     ║
    ║ geometry only defines a mask — the pressure has no physical        ║
    ║ relationship to actual airfoil aerodynamics.                       ║
    ║                                                                    ║
    ║ This function is suitable ONLY for demonstrating the FNO data     ║
    ║ pipeline and training loop. Do NOT use these outputs for any       ║
    ║ aerodynamic analysis or validation.                                ║
    ╚══════════════════════════════════════════════════════════════════════╝

    The "airfoil" is parameterized by thickness and camber.
    The "pressure" is a synthetic potential-flow-inspired field around
    a point source at (0.25, camber), masked inside the airfoil shape.

    Args:
        n_samples: number of samples to generate
        resolution: grid resolution (H = W = resolution)
        seed: random seed for reproducibility

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
