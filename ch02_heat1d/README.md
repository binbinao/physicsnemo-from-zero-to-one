# 第 2 章 · 1D 热传导 PINN

## 先跑这个

```bash
python heat1d_pinn_raw.py
# 或（Hydra）
python heat1d_train.py
```

## 三档脚本

| 档位 | 文件 | 依赖 |
|:---|:---|:---|
| **首选 · 裸 PyTorch** | `heat1d_pinn_raw.py` | torch |
| Hydra 训练 | `heat1d_train.py` | torch, hydra-core |
| SDK | `heat1d_pinn_sdk.py` | physicsnemo |
| GPU | `heat1d_pinn_gpu.py` | CUDA |

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

DDP 多卡时只有 rank 0 写日志。关闭记录：加 `--no-tb`，或设 `PN_TB=0`。
详见 [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md#训练曲线tensorboard默认开启)。

## 命令参考

[docs/COMMAND_REFERENCE.md](../docs/COMMAND_REFERENCE.md#ch02hydraconfig-在-conf)

## 教材

[book/ch02.md](../book/ch02.md)
