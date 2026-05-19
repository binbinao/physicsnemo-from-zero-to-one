# 环境要求（单一事实来源）

> 全仓库以本文档为准；README、`book/ch00.md`、`scripts/check_env.py` 与之对齐。

## Python

| 项 | 要求 |
|:---|:---|
| **最低版本** | Python **3.10** |
| **推荐版本** | Python **3.11** |
| **不支持** | Python 3.9 及以下（`check_env` 会报错） |

## PyTorch

| 项 | 要求 |
|:---|:---|
| **最低版本** | PyTorch **2.3+** |
| **CUDA** | 第 4 章起建议 **CUDA 12.x** 对应 wheel；仅学 ch1–2 可用 CPU |
| **安装** | 见 [pytorch.org](https://pytorch.org/get-started/locally/) |

## 分阶段依赖

### Tier 0 — 第 1 天（ch01 裸 PyTorch）

```bash
pip install "torch>=2.3" numpy matplotlib
```

详见 [QUICKSTART_DAY1.md](QUICKSTART_DAY1.md)。

### Tier 1 — ch01–ch03 全部裸 PyTorch版

```bash
pip install "torch>=2.3" numpy matplotlib
```

### 教材配图（`book/assets/`）

含中文标注的图由 **Matplotlib 程序化出图**（非文生图 AI），须能加载仓库内置 **Noto Sans SC**：

```bash
python scripts/download_book_fonts.py   # 若缺少 book/assets/fonts/NotoSansSC-Regular.otf
python scripts/check_figure_cjk.py
python book/scripts/generate_diagram_figures.py
python book/scripts/generate_all_figures.py
```

详见 [BOOK_FIGURE_ART_QA.md](BOOK_FIGURE_ART_QA.md)、[book/assets/fonts/README.md](../book/assets/fonts/README.md)。

### Tier 2 — SDK 版（ch01–ch07 的 `*_sdk.py`）

```bash
pip install "torch>=2.3" numpy matplotlib nvidia-physicsnemo nvidia-physicsnemo.sym hydra-core
```

### Tier 3 — 全书 + 部署（ch07 API / ONNX）

```bash
pip install "torch>=2.3" numpy matplotlib nvidia-physicsnemo nvidia-physicsnemo.sym \
  hydra-core optuna fastapi uvicorn onnx
```

## 按章依赖速查

| 章 | 首选脚本 | 最低 Tier | GPU |
|:---|:---|:---|:---|
| ch01 | `pinn_spring.py` | 0 | 可选 |
| ch02 | `heat1d_pinn_raw.py` 或 `heat1d_train.py` | 0 / 1 | 可选 |
| ch03 | `heat_sink_train.py` | 1 | 建议 8GB |
| ch04 | `train_fno_mini.py` | 1 | 建议 8GB |
| ch05 | `train_data_fno.py` | 1 + **ch04 目录** | 建议 8GB |
| ch06 | `train_afno_mini.py` | 1 | 建议 8GB |
| ch07 | `train.py` | 1；Optuna 要 Tier 3 | 建议 8GB |

完整表：[DEPENDENCIES_BY_CHAPTER.md](DEPENDENCIES_BY_CHAPTER.md) · 硬件/时长：[HARDWARE_EXPECTATIONS.md](HARDWARE_EXPECTATIONS.md)

## 验证

```bash
python scripts/check_env.py           # 全局核心 + Tier 概览
python scripts/check_env.py --list  # 各章 Tier / GPU 速查
python scripts/check_env.py --chapter 4
python scripts/check_env.py --tier 2
```

`--chapter` 会检查该章必需包（含 ch05 的 `ch04_fno_airfoil/` 目录）及可选 SDK 包。

## 云 GPU / Colab

[CLOUD_GPU_GUIDE.md](CLOUD_GPU_GUIDE.md) · [notebooks/colab_quickstart.ipynb](../notebooks/colab_quickstart.ipynb)

## SDK 装不上时怎么办

| 情况 | 建议 |
|:---|:---|
| 无 NVIDIA GPU / Mac Apple Silicon | 先只装 `requirements-minimal.txt`，跑 ch01–ch03 裸 PyTorch；ch04+ 用 [Colab](CLOUD_GPU_GUIDE.md) |
| `pip install nvidia-physicsnemo` 失败 | 查 [附录 C](../book/appendix_c_troubleshooting.md)；暂跳过 `*_sdk.py`，不影响 raw 脚本 |
| 仅需某一章 SDK | 见 [DEPENDENCIES_BY_CHAPTER.md](DEPENDENCIES_BY_CHAPTER.md)，不必一次装全量 |

**Apple Silicon**：PyTorch 可用 MPS 加速部分算子，但 PhysicsNeMo 官方以 Linux + CUDA 为主；本书微缩实验建议在 Colab T4 或 Linux GPU 上跑 ch04–ch07。

**Windows**：推荐 [WSL2](https://learn.microsoft.com/windows/wsl/) + Ubuntu，在 WSL 内按 Linux 步骤安装；原生 Windows 下 sym 常遇路径/编译问题。

## 依赖文件

| 文件 | 用途 |
|:---|:---|
| [requirements-minimal.txt](../requirements-minimal.txt) | ch01–ch03 裸 PyTorch |
| [requirements-full.txt](../requirements-full.txt) | SDK + ch07 |
| [TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT.md) | 最近验证的 commit / Python / PyTorch |
