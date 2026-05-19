# 分章依赖表

与 [ENVIRONMENT.md](ENVIRONMENT.md) 配套；各章细节见 `chXX_*/README.md`。

| 章 | 目录 | 首选入口 | pip（除 torch/numpy/matplotlib） | 其他 |
|:---|:---|:---|:---|:---|
| 1 | `ch01_hello/` | `pinn_spring.py` | — | — |
| 2 | `ch02_heat1d/` | `heat1d_pinn_raw.py` | `hydra-core`（`heat1d_train.py`） | — |
| 3 | `ch03_heatsink/` | `heat_sink_train.py` | — | — |
| 4 | `ch04_fno_airfoil/` | `train_fno_mini.py` | `hydra-core`（可选） | — |
| 5 | `ch05_darcy_hybrid/` | `train_data_fno.py` | `hydra-core`（可选） | **依赖 `ch04_fno_airfoil/`** |
| 6 | `ch06_fourcastnet_mini/` | `train_afno_mini.py` | `hydra-core`（可选） | 先跑 `scripts/generate_toy_weather.py` |
| 7 | `ch07_drivaernet_optim/` | `train.py` | `optuna`（优化）；`fastapi uvicorn`（API） | — |

**三档脚本后缀**：裸 PyTorch（默认文件名）· `*_sdk.py` · `*_gpu.py`（需 CUDA + 常需 PhysicsNeMo）。
