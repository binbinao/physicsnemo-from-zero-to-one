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

## 版本锁定（计划）

正式 `requirements-*.txt` 见 Issue #13；当前以本文档 + 各章 README 为准。
