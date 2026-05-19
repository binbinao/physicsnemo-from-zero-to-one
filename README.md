# PhysicsNeMo: From Zero to One

[![GitHub](https://img.shields.io/badge/GitHub-binbinao%2Fphysicsnemo--from--zero--to--one-blue?logo=github)](https://github.com/binbinao/physicsnemo-from-zero-to-one)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-red?logo=pytorch)](https://pytorch.org/)
[![PhysicsNeMo](https://img.shields.io/badge/PhysicsNeMo-2.0-green?logo=nvidia)](https://github.com/NVIDIA/physicsnemo)

A hands-on tutorial book for physics-informed neural networks (PINNs) and neural operators using NVIDIA PhysicsNeMo. Every chapter provides **three variants**: raw PyTorch (educational), PhysicsNeMo SDK (CPU-verified), and GPU production (DDP + AMP).

> **中文读者请从这里开始 → [docs/START_HERE.md](docs/START_HERE.md)**（教材目录 [book/SUMMARY.md](book/SUMMARY.md)，正文从 [book/ch00.md](book/ch00.md)；第 1 天安装见 [QUICKSTART_DAY1](docs/QUICKSTART_DAY1.md)）

> 📖 **Built with multi-agent orchestration in 2 weeks.** See [Vibe Publishing](#vibe-publishing) below.

---

## Requirements

环境以 **[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)** 为准（Python ≥3.10，PyTorch ≥2.3）。

| 阶段 | 安装 |
|:---|:---|
| **第 1 天（ch01）** | `pip install -r requirements-minimal.txt` — [QUICKSTART_DAY1](docs/QUICKSTART_DAY1.md) |
| **ch01–ch03 裸 PyTorch** | `requirements-minimal.txt` |
| **SDK + ch07** | `pip install -r requirements-full.txt` |
| **验证快照** | [docs/TESTED_ENVIRONMENT.md](docs/TESTED_ENVIRONMENT.md) |

**书—代码命令**：[docs/COMMAND_REFERENCE.md](docs/COMMAND_REFERENCE.md) · **分章说明**：各 `chXX_*/README.md`

## 附录（教材）

| 附录 | 文件 | 内容 |
|:---|:---|:---|
| A | [book/appendix_a_math.md](book/appendix_a_math.md) | 数学与符号速查 |
| B | [book/appendix_b_cloud_gpu.md](book/appendix_b_cloud_gpu.md) | 云 GPU 与环境（速查） |
| C | [book/appendix_c_troubleshooting.md](book/appendix_c_troubleshooting.md) | 常见踩坑 50 问 |
| D | [book/appendix_d_pytorch_mini.md](book/appendix_d_pytorch_mini.md) | PyTorch 30 分钟最小集 |

**学习路径**：[6 周计划](docs/STUDY_PLAN_6WEEKS.md) · **读完后**：[WHATS_NEXT](docs/WHATS_NEXT.md) · **训练基线**：[results/BASELINE.md](results/BASELINE.md)

**无本地 GPU？** [云 GPU 分步指南](docs/CLOUD_GPU_GUIDE.md) · [Colab 笔记本](notebooks/colab_quickstart.ipynb) · [硬件预期表](docs/HARDWARE_EXPECTATIONS.md)

## 读者改进跟踪

- 第一轮（#1–#16）：已完成 → [BEGINNER_ISSUE_BACKLOG](docs/BEGINNER_ISSUE_BACKLOG.md)
- 第二轮审阅（#17–#53）：已完成 → [REVIEW_ROUND2_BACKLOG](docs/REVIEW_ROUND2_BACKLOG.md)
- **CAE 审阅（#54–#88）**：已完成 → [CAE_REVIEW_BACKLOG](docs/CAE_REVIEW_BACKLOG.md) · [START_HERE_CAE](docs/START_HERE_CAE.md)
- **CAE Phase 3（#104–#108）**：CFD 批跑 / 联合反演 / 多目标优化 → [tools/cfd_batch](tools/cfd_batch/README.md)

## Directory Structure

```
physicsnemo-from-zero-to-one/
├── book/                                # Textbook ch00–ch07, appendices, assets/
├── docs/                                # START_HERE, ENVIRONMENT, study plan, …
├── notebooks/                           # Colab quickstart
├── scripts/
│   └── check_env.py                    # Environment self-check
├── ch01_hello/                          # Chapter 1: Hello PINN
│   ├── mlp_spring.py                   # Data-driven MLP baseline
│   ├── pinn_spring.py                  # Physics-informed (raw PyTorch)
│   ├── pinn_spring_sdk.py              # PhysicsNeMo SDK version
│   └── conf/config.yaml
├── ch02_heat1d/                         # Chapter 2: 1D Heat Conduction
│   ├── heat1d_pinn_raw.py             # Raw PyTorch PINN
│   ├── heat1d_pinn_sdk.py             # PhysicsNeMo FullyConnected
│   ├── heat1d_pinn_gpu.py             # GPU + DDP + AMP
│   ├── heat1d_train.py                # Hydra-based training
│   ├── heat1d_visualize.py            # Visualization
│   └── conf/                          # Hydra configs
├── ch03_heatsink/                       # Chapter 3: 2D Heat Sink
│   ├── heat_sink_train.py             # Raw PyTorch (multi-BC)
│   ├── heat_sink_train_sdk.py         # PhysicsNeMo SDK
│   ├── heat_sink_train_gpu.py         # GPU production
│   ├── heat_sink_geometry.py          # CSG geometry
│   ├── heat_sink_inverse.py           # Inverse problem
│   ├── equations.py                   # PDE definitions
│   └── visualize.py
├── ch04_fno_airfoil/                    # Chapter 4: FNO (airfoil narrative; default train = Darcy)
│   ├── CH04_GUIDE.md                 # Why airfoil vs Darcy + two paths
│   ├── fno_model.py                   # FNO from scratch
│   ├── train_fno_mini.py             # Default: Darcy synthetic data
│   ├── train_fno_sdk.py              # PhysicsNeMo FNO
│   ├── train_fno_gpu.py              # GPU + DDP + coord_features
│   ├── dataset.py                    # Darcy + optional airfoil synthetic
│   └── visualize_airfoil.py
├── ch05_darcy_hybrid/                   # Chapter 5: Physics-Informed FNO
│   ├── train_data_fno.py             # Data-only baseline
│   ├── train_physics_fno.py          # Hybrid data+physics (raw)
│   ├── train_physics_fno_sdk.py      # PhysicsNeMo SDK
│   ├── train_physics_fno_gpu.py      # GPU production
│   ├── darcy_residual.py             # PDE residual computation
│   └── visualize.py
├── ch06_fourcastnet_mini/               # Chapter 6: Mini FourCastNet
│   ├── afno_model.py                 # AFNO from scratch
│   ├── train_afno_mini.py            # Raw PyTorch training
│   ├── train_afno_sdk.py             # PhysicsNeMo AFNO
│   ├── train_afno_gpu.py             # GPU + DDP + rollout
│   ├── rollout_eval.py               # Autoregressive evaluation
│   ├── dataset.py                    # Weather window dataset
│   └── scripts/generate_toy_weather.py
└── ch07_drivaernet_optim/               # Chapter 7: End-to-End Pipeline
    ├── train.py                       # Raw PyTorch surrogate
    ├── train_sdk.py                   # PhysicsNeMo FullyConnected
    ├── train_gpu.py                   # GPU + Optuna HPO
    ├── optimize.py                    # Design optimization
    ├── export_onnx.py                 # ONNX export
    ├── api/app.py                     # FastAPI deployment
    ├── data/generate_toy_cars.py
    └── models/cd_mlp.py
```

## Quick Start

```bash
git clone https://github.com/binbinao/physicsnemo-from-zero-to-one.git
cd physicsnemo-from-zero-to-one
python scripts/check_env.py
```

**第 1 天最小路径**（无需 PhysicsNeMo）：见 [docs/QUICKSTART_DAY1.md](docs/QUICKSTART_DAY1.md)

```bash
# Chapter 1: Your first PINN
python ch01_hello/pinn_spring.py

# Chapter 4: FNO (generates data automatically)
python ch04_fno_airfoil/train_fno_mini.py

# Chapter 7: Full pipeline (train → optimize → deploy)
python ch07_drivaernet_optim/train.py
python ch07_drivaernet_optim/optimize.py
```

### GPU Training (requires CUDA)

```bash
# Single GPU with PhysicsNeMo SDK
python ch04_fno_airfoil/train_fno_sdk.py --epochs 30

# Multi-GPU with DDP
torchrun --nproc_per_node=4 ch06_fourcastnet_mini/train_afno_gpu.py --epochs 50 --ddp

# Hyperparameter search
python ch07_drivaernet_optim/train_gpu.py --hpo --hpo_trials 50
```

## Three Variants Per Chapter

| Variant | Suffix | Purpose | Dependencies |
|:---|:---|:---|:---|
| **Raw PyTorch** | `*.py` / `*_raw.py` | Educational, fully transparent | torch only |
| **PhysicsNeMo SDK** | `*_sdk.py` | Industry API, skip_connections, weight_norm | nvidia-physicsnemo |
| **GPU Production** | `*_gpu.py` | DDP, AMP, large-scale, checkpointing | nvidia-physicsnemo + CUDA |

## Philosophy

1. **No black boxes** — Every model is built from raw PyTorch first
2. **No data downloads** — Synthetic generators included for every chapter
3. **Gradual complexity** — From 1D PDE → 2D geometry → neural operators → full pipelines
4. **Production-ready** — Hydra configs, DDP, AMP, ONNX export, REST API
5. **Three-level learning** — Understand (raw) → Use (SDK) → Scale (GPU)

## Vibe Publishing

This entire codebase (56 files, 7,400+ lines) was created in **2 weeks** using multi-agent AI orchestration:

- 🤖 **Agent pipeline**: Topic Scout → Plan → Research → Parallel Writing → Review → Revision
- ⚡ **30× productivity multiplier** vs traditional solo authoring
- ✅ **All CPU variants verified** to converge on real hardware
- 🧑‍💻 **Human-in-the-loop** for architecture, taste, and quality control

Read the full story in the repository commit history and chapter READMEs; multi-agent workflow is summarized in each chapter's delivery notes.

## License

Educational use. Based on concepts from [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo).

## Citation

```bibtex
@misc{physicsnemo-from-zero-to-one,
  author = {robinji},
  title = {PhysicsNeMo: From Zero to One},
  year = {2026},
  url = {https://github.com/binbinao/physicsnemo-from-zero-to-one}
}
```
