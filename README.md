# PhysicsNeMo: From Zero to One

A hands-on tutorial book for physics-informed neural networks (PINNs) and neural operators, inspired by NVIDIA PhysicsNeMo. Every chapter builds from scratch in pure PyTorch so you understand the fundamentals before reaching for frameworks.

## Requirements

**Core (required):**
```
torch >= 2.0
numpy >= 1.24
matplotlib >= 3.7
```

**Optional (enhances experience):**
```
hydra-core >= 1.3    # structured config management
optuna >= 3.0        # Bayesian optimization (ch07)
fastapi >= 0.100     # model serving API (ch07)
uvicorn >= 0.23      # ASGI server for FastAPI
onnx >= 1.14         # ONNX export (ch07)
```

Install core:
```bash
pip install torch numpy matplotlib
```

Install everything:
```bash
pip install torch numpy matplotlib hydra-core optuna fastapi uvicorn onnx
```

## Directory Structure

```
physicsnemo-from-zero-to-one/
├── README.md
├── ch01_basics/              # Environment check & PyTorch warm-up
│   └── check_env.py
├── ch02_heat1d/              # 1D heat equation PINN
│   ├── heat1d_pinn_raw.py    # Raw PyTorch PINN (no framework)
│   ├── heat1d_train.py       # Hydra-based training
│   ├── heat1d_visualize.py   # Visualization from checkpoint
│   └── conf/                 # Hydra config hierarchy
│       ├── config.yaml
│       ├── arch/
│       │   ├── small.yaml
│       │   └── large.yaml
│       └── training/
│           ├── debug.yaml
│           └── full.yaml
├── ch03_heatsink/            # 2D heat sink with CSG geometry
│   ├── heat_sink_geometry.py # Rectangle-based CSG
│   ├── equations.py          # HeatConduction2D + Robin BC
│   ├── heat_sink_train.py    # Multi-constraint training
│   ├── heat_sink_inverse.py  # Inverse problem
│   ├── visualize.py          # Temperature contour plots
│   └── conf/config.yaml
├── ch04_fno_airfoil/         # Fourier Neural Operator
│   ├── dataset.py            # Synthetic data generators
│   ├── fno_model.py          # FNO2D from scratch
│   ├── train_fno_mini.py     # Training script
│   ├── visualize_airfoil.py  # Prediction visualization
│   └── conf/config.yaml
├── ch05_darcy_hybrid/        # Physics-informed FNO
│   ├── darcy_residual.py     # Darcy PDE via finite differences
│   ├── train_data_fno.py     # Pure data baseline
│   ├── train_physics_fno.py  # Hybrid data + physics
│   ├── visualize.py
│   └── conf/config.yaml
├── ch06_fourcastnet_mini/    # Mini FourCastNet (AFNO)
│   ├── scripts/
│   │   └── generate_toy_weather.py
│   ├── dataset.py            # Weather window dataset
│   ├── afno_model.py         # Simplified AFNO
│   ├── train_afno_mini.py    # Training
│   ├── rollout_eval.py       # Autoregressive rollout
│   └── conf/config.yaml
└── ch07_drivaernet_optim/    # Design optimization pipeline
    ├── data/
    │   └── generate_toy_cars.py
    ├── models/
    │   └── cd_mlp.py
    ├── train.py
    ├── optimize.py           # Optuna optimization
    ├── export_onnx.py        # ONNX export
    ├── api/
    │   └── app.py            # FastAPI serving
    └── conf/config.yaml
```

## Quick Start

```bash
# Step 0: Check environment
python ch01_basics/check_env.py

# Step 1: Your first PINN — 1D heat equation
python ch02_heat1d/heat1d_pinn_raw.py

# Step 2: Hydra-based training (optional, needs hydra-core)
cd ch02_heat1d && python heat1d_train.py

# Step 3: 2D heat sink with geometry
cd ch03_heatsink && python heat_sink_train.py

# Step 4: Fourier Neural Operator
cd ch04_fno_airfoil && python train_fno_mini.py

# Step 5: Physics-informed FNO
cd ch05_darcy_hybrid && python train_physics_fno.py

# Step 6: Mini weather forecasting
cd ch06_fourcastnet_mini && python scripts/generate_toy_weather.py
cd ch06_fourcastnet_mini && python train_afno_mini.py

# Step 7: Design optimization pipeline
cd ch07_drivaernet_optim && python data/generate_toy_cars.py
cd ch07_drivaernet_optim && python train.py
cd ch07_drivaernet_optim && python optimize.py
```

## Philosophy

1. **No black boxes** — Every model is built from raw PyTorch first
2. **No data downloads** — Synthetic generators included for every chapter
3. **Gradual complexity** — From 1D PDE → 2D geometry → neural operators → full pipelines
4. **Production-ready patterns** — Hydra configs, checkpointing, ONNX export, REST API

## License

Educational use. Based on concepts from NVIDIA PhysicsNeMo / Modulus.
