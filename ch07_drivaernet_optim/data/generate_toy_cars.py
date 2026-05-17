"""
Chapter 7: Generate Synthetic Car Aerodynamics Data
=====================================================
Create toy data mapping car design parameters to drag coefficient (Cd).

Parameters (7 features):
  - body_length: 3.5 - 5.5 m
  - body_width: 1.5 - 2.2 m
  - body_height: 1.2 - 1.8 m
  - front_angle: 10 - 40 degrees (hood angle)
  - rear_angle: 5 - 35 degrees (fastback angle)
  - ground_clearance: 0.1 - 0.3 m
  - wheel_diameter: 0.5 - 0.8 m

Target: Cd (drag coefficient) ~ 0.20 - 0.50

The relationship is synthetic but physically inspired.

Usage:
    python data/generate_toy_cars.py
    python data/generate_toy_cars.py --n_samples 2000
"""

import os
import argparse

import numpy as np
import torch


def compute_cd(params: np.ndarray) -> np.ndarray:
    """
    Compute synthetic drag coefficient from car parameters.
    
    This is a made-up but physically motivated formula:
    - Longer, lower cars → lower Cd
    - Steeper front/rear angles → higher Cd
    - Higher ground clearance → slightly higher Cd (underbody flow)
    
    Args:
        params: (N, 7) array of car parameters
        
    Returns:
        cd: (N,) array of drag coefficients
    """
    length = params[:, 0]
    width = params[:, 1]
    height = params[:, 2]
    front_angle = params[:, 3]
    rear_angle = params[:, 4]
    clearance = params[:, 5]
    wheel_d = params[:, 6]

    # Base Cd from aspect ratio (height/length)
    aspect = height / length
    cd_base = 0.15 + 0.5 * aspect

    # Front angle contribution (in radians)
    cd_front = 0.003 * front_angle  # degrees to contribution

    # Rear angle contribution (separation at rear)
    cd_rear = 0.005 * rear_angle

    # Width penalty (larger frontal area relative to length)
    cd_width = 0.02 * (width - 1.5)

    # Ground clearance (higher = more underbody drag)
    cd_clearance = 0.1 * clearance

    # Wheel exposed area
    cd_wheel = 0.05 * (wheel_d - 0.5)

    cd = cd_base + cd_front + cd_rear + cd_width + cd_clearance + cd_wheel

    # Add small noise for realism
    noise = np.random.randn(len(cd)) * 0.005
    cd = cd + noise

    return np.clip(cd, 0.15, 0.60).astype(np.float32)


def generate_car_data(n_samples: int = 1000, seed: int = 42) -> dict:
    """Generate synthetic car → Cd dataset."""
    rng = np.random.RandomState(seed)

    # Sample parameters uniformly
    params = np.column_stack([
        rng.uniform(3.5, 5.5, n_samples),    # body_length
        rng.uniform(1.5, 2.2, n_samples),    # body_width
        rng.uniform(1.2, 1.8, n_samples),    # body_height
        rng.uniform(10, 40, n_samples),       # front_angle (degrees)
        rng.uniform(5, 35, n_samples),        # rear_angle (degrees)
        rng.uniform(0.1, 0.3, n_samples),    # ground_clearance
        rng.uniform(0.5, 0.8, n_samples),    # wheel_diameter
    ]).astype(np.float32)

    np.random.seed(seed)
    cd = compute_cd(params)

    param_names = ["body_length", "body_width", "body_height",
                   "front_angle", "rear_angle", "ground_clearance",
                   "wheel_diameter"]

    return {
        'params': torch.tensor(params),       # (N, 7)
        'cd': torch.tensor(cd),               # (N,)
        'param_names': param_names,
        'param_ranges': {
            'body_length': (3.5, 5.5),
            'body_width': (1.5, 2.2),
            'body_height': (1.2, 1.8),
            'front_angle': (10.0, 40.0),
            'rear_angle': (5.0, 35.0),
            'ground_clearance': (0.1, 0.3),
            'wheel_diameter': (0.5, 0.8),
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    print(f"Generating {args.n_samples} synthetic car samples...")
    data = generate_car_data(args.n_samples)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "car_aero_data.pt")
    torch.save(data, path)

    print(f"Saved {path}")
    print(f"  params shape: {list(data['params'].shape)}")
    print(f"  Cd range: [{data['cd'].min():.3f}, {data['cd'].max():.3f}]")
    print(f"  Cd mean: {data['cd'].mean():.3f}")
    print(f"  Features: {data['param_names']}")
