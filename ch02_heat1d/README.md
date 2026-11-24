# 第 2 章 · 1D 热传导 PINN

## 先跑这个

```bash
# 始终 argparse（推荐初学）
python heat1d_pinn_raw.py --steps 500
# Hydra 入口：未装 hydra → --steps；已装 hydra → training.steps=500 / training=debug
python heat1d_train.py
```

> **Hydra / argparse 双语法**：装 `hydra-core` 时用 `key=value`（本章如 `training.steps=500`、`arch=large`）；未装时用 `--flag`（如 `--steps 500`）。注意 `steps` / `lr` / `hidden` 在 `training.*` 与 `arch.*` 分组下，要写全 `training.steps=500`（或直接用 `training=debug` 预设）。见 [COMMAND_REFERENCE](../docs/COMMAND_REFERENCE.md)。

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

## 查看训练曲线

训练默认记录曲线，结束后会打印日志目录：

| 脚本 | 默认日志目录 |
|:---|:---|
| `heat1d_train.py`（Hydra） | `outputs/<日期>/<时间>/tb` |
| `heat1d_pinn_raw.py` | `outputs/tb/<时间戳>` |
| `heat1d_pinn_sdk.py` | `outputs/tb/<时间戳>` |
| `heat1d_pinn_gpu.py` | `outputs_gpu/tb/<时间戳>` |

```bash
python scripts/plot_tb_curves.py --logdir <目录> --out curves.png   # 零依赖出图
tensorboard --logdir <目录>                                          # 需 pip install tensorboard
```

DDP 多卡时只有 rank 0 写日志。关闭记录：设 `PN_TB=0`（通用）；非 Hydra 档脚本也可加 `--no-tb`。
详见 [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md#训练曲线tensorboard默认开启)。

## 命令参考

[docs/COMMAND_REFERENCE.md](../docs/COMMAND_REFERENCE.md)

## 教材

[book/ch02.md](../book/ch02.md)
