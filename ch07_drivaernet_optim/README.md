# 第 7 章 · 端到端流水线

> **数据档位**：默认 **toy 合成** `generate_toy_cars.py`，非完整 DrivAerNet。checkpoint `meta.data_tier=toy_synthetic`。

## 先跑这个

```bash
python train.py --epochs 100
python optimize.py --checkpoint outputs/best.pt --n_trials 50
```

## 流程

| 步骤 | 脚本 |
|:---|:---|
| 1. 数据 | `data/generate_toy_cars.py`（按需） |
| 2. 训练 | `train.py` |
| 3. 优化 | `optimize.py`（单目标 Cd）· `optimize_multi.py`（Cd + 迎风面积） |
| 3b. CFD 复核队列 | `hifi_validation_queue.py` → [tools/cfd_batch](../tools/cfd_batch/README.md) |
| 4. 导出 | `export_onnx.py` |
| 5. 部署 | `api/app.py` |

## 三档脚本

| 档位 | 文件 |
|:---|:---|
| **首选** | `train.py` |
| SDK | `train_sdk.py` |
| GPU + HPO | `train_gpu.py` |

## 查看训练曲线

训练默认记录曲线，结束后会打印日志目录：

| 脚本 | 默认日志目录 |
|:---|:---|
| `train.py` / `train_sdk.py` | `outputs/tb/<时间戳>` |
| `train_gpu.py` | `outputs_gpu/tb/<时间戳>`（`--hpo` 时每个 trial 为 `outputs_gpu/tb/trial<N>/<时间戳>`） |

```bash
python scripts/plot_tb_curves.py --logdir <目录> --out curves.png   # 零依赖出图
tensorboard --logdir <目录>                                          # 需 pip install tensorboard
```

DDP 多卡时只有 rank 0 写日志。关闭记录：加 `--no-tb`，或设 `PN_TB=0`。
详见 [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md#训练曲线tensorboard默认开启)。

## 教材

[book/ch07.md](../book/ch07.md)
