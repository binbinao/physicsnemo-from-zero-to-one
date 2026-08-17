# 第 2 章 · 1D 热传导 PINN

## 先跑这个

```bash
# 始终 argparse（推荐初学）
python heat1d_pinn_raw.py --steps 500
# Hydra 入口：未装 hydra → --steps；已装 hydra → steps=500 / training=debug
python heat1d_train.py
```

> **Hydra 提示**：装了 `hydra-core` 后请用 `key=value`，不要用 `--steps`。见 [COMMAND_REFERENCE](../docs/COMMAND_REFERENCE.md)。

## 三档脚本

| 档位 | 文件 | 依赖 | 说明 |
|:---|:---|:---|:---|
| **首选 · 裸 PyTorch** | `heat1d_pinn_raw.py` | torch | 本章唯一带 `_raw` 后缀的文件 |
| Hydra 训练 | `heat1d_train.py` | torch, hydra-core | 与 raw **同问题族**，配置驱动 |
| SDK | `heat1d_pinn_sdk.py` | physicsnemo | **物理设定不同**（α、IC），**不可直接对比** loss |
| GPU | `heat1d_pinn_gpu.py` | CUDA | 生产档 |

> ⚠️ `heat1d_pinn_raw.py`（α=0.1，高斯 IC）与 `heat1d_pinn_sdk.py`（α=0.01，sin IC）**不是同一 PDE 设定**，勿把两版 loss 曲线直接叠在一起比。

## 可视化

```bash
python heat1d_visualize.py
```

## 命令参考

[docs/COMMAND_REFERENCE.md](../docs/COMMAND_REFERENCE.md)

## 教材

[book/ch02.md](../book/ch02.md)
