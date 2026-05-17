"""
Chapter 6: Generate Synthetic Weather Data
============================================
Create toy weather-like data for FourCastNet training.
No ERA5 download needed — purely synthetic.

Generates fields that mimic weather variables on a lat-lon grid:
  - Temperature (T): smooth spatial field + temporal advection
  - Geopotential (Z): large-scale wave patterns
  - Wind (U, V): derived from geopotential via geostrophic balance

Usage:
    python scripts/generate_toy_weather.py
    python scripts/generate_toy_weather.py --n_time 200 --resolution 32
"""

import os
import argparse

import numpy as np
import torch


def generate_toy_weather(n_time: int = 200, resolution: int = 32,
                          n_vars: int = 4, seed: int = 42) -> dict:
    """
    Generate synthetic weather-like time series on a 2D grid.

    Variables (channels):
      0: Temperature-like field (T)
      1: Geopotential-like field (Z)
      2: Zonal wind-like field (U)
      3: Meridional wind-like field (V)

    Args:
        n_time: number of time steps
        resolution: spatial grid size (H = W = resolution)
        n_vars: number of weather variables
        seed: random seed

    Returns:
        dict with 'data' tensor of shape (n_time, n_vars, H, W)
    """
    rng = np.random.RandomState(seed)
    H = W = resolution
    dt = 0.01  # time step (small for numerical stability)

    # Spatial grids (latitude, longitude)
    lat = np.linspace(-90, 90, H)
    lon = np.linspace(0, 360, W, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)
    LON_rad = np.deg2rad(LON)
    LAT_rad = np.deg2rad(LAT)

    data = np.zeros((n_time, n_vars, H, W), dtype=np.float32)

    # Initial conditions
    # Temperature: warm equator, cold poles + random perturbation
    T0 = 30 * np.cos(LAT_rad) + rng.randn(H, W) * 2
    # Geopotential: large-scale Rossby wave
    k_wave = rng.choice([2, 3, 4])  # zonal wavenumber
    Z0 = 500 * np.cos(LAT_rad) * np.cos(k_wave * LON_rad) + \
         rng.randn(H, W) * 10
    # Wind from geostrophic balance (simplified)
    U0 = -np.gradient(Z0, axis=0) * 0.1
    V0 = np.gradient(Z0, axis=1) * 0.1

    data[0, 0] = T0
    data[0, 1] = Z0
    data[0, 2] = U0
    data[0, 3] = V0

    # Time evolution: simple advection + diffusion + random forcing
    for t in range(1, n_time):
        T_prev = data[t - 1, 0]
        Z_prev = data[t - 1, 1]
        U_prev = data[t - 1, 2]
        V_prev = data[t - 1, 3]

        # Advect temperature by wind (clamped for stability)
        dT_dx = np.gradient(T_prev, axis=1)
        dT_dy = np.gradient(T_prev, axis=0)
        advection = U_prev * dT_dx + V_prev * dT_dy
        advection = np.clip(advection, -50, 50)  # prevent blowup
        T_new = T_prev - dt * advection
        # Diffusion
        lap_T = np.gradient(np.gradient(T_prev, axis=0), axis=0) + \
                np.gradient(np.gradient(T_prev, axis=1), axis=1)
        T_new += 0.5 * dt * lap_T
        # Random forcing
        T_new += rng.randn(H, W) * 0.1

        # Evolve geopotential (Rossby wave propagation)
        phase_speed = -5.0 * dt  # westward propagation
        Z_shifted = np.roll(Z_prev, int(phase_speed), axis=1)
        Z_new = Z_shifted + rng.randn(H, W) * 0.5

        # Update wind
        U_new = -np.gradient(Z_new, axis=0) * 0.1 + rng.randn(H, W) * 0.05
        V_new = np.gradient(Z_new, axis=1) * 0.1 + rng.randn(H, W) * 0.05

        data[t, 0] = T_new
        data[t, 1] = Z_new
        data[t, 2] = U_new
        data[t, 3] = V_new

    # Check for NaN/Inf and replace
    data = np.nan_to_num(data, nan=0.0, posinf=100.0, neginf=-100.0)

    # Normalize each variable to zero mean, unit std
    for v in range(n_vars):
        mean = data[:, v].mean()
        std = data[:, v].std() + 1e-8
        data[:, v] = (data[:, v] - mean) / std

    return {
        'data': torch.tensor(data),
        'n_vars': n_vars,
        'resolution': resolution,
        'n_time': n_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_time", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    print(f"Generating toy weather data: {args.n_time} steps, "
          f"{args.resolution}x{args.resolution} grid...")

    result = generate_toy_weather(args.n_time, args.resolution)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "toy_weather.pt")
    torch.save(result, path)
    print(f"Saved {path}: shape={list(result['data'].shape)}")
    print(f"Variables: T, Z, U, V  |  Normalized to zero-mean, unit-std")
