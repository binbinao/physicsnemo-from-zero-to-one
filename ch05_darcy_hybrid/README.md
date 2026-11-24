# 第 5 章 · Physics-Informed FNO

## ⚠️ 依赖第 4 章

本目录 **import** `../ch04_fno_airfoil/fno_model.py`。请保留完整仓库，并先跑通 ch04 或至少存在该目录。

若报错 `No module named fno_model`：确认未单独拷贝本章文件夹。

## 先跑这个

```bash
# 无 Hydra：--epochs；有 Hydra：epochs=50 n_train=100
python train_data_fno.py --epochs 50 --n_train 100
python train_physics_fno.py --epochs 50 --n_train 100 --lambda_physics 0.1
```

> **Hydra / argparse 双语法**：装 `hydra-core` 时用 `key=value`（如 `epochs=30`、`steps=500`）；未装时用 `--flag`（如 `--epochs 30`、`--steps 500`）。见 [COMMAND_REFERENCE](../docs/COMMAND_REFERENCE.md)。  
> 另需 **scipy**（`requirements-minimal.txt` 已含）+ **ch04 目录**已存在。

## 三档 / 对照脚本

| 档位 | 文件 | 依赖 | 说明 |
|:---|:---|:---|:---|
| **基线 · 纯数据** | `train_data_fno.py` | torch, scipy | 无 PDE 正则（裸 PyTorch） |
| **首选 · 混合** | `train_physics_fno.py` | torch, scipy | 数据 + Darcy 残差 |
| SDK | `train_physics_fno_sdk.py` | physicsnemo | PhysicsNeMo backbone |
| GPU | `train_physics_fno_gpu.py` | CUDA | 生产档 |

> 本章不是简单的 mini/sdk/gpu 三平行，而是 **数据基线 → 混合损失** 的实验对照。

## 查看训练曲线

训练默认记录曲线，结束后会打印日志目录：

| 脚本 | 默认日志目录 |
|:---|:---|
| `train_data_fno.py` / `train_physics_fno.py` / `train_physics_fno_sdk.py` | `outputs/tb/<时间戳>` |
| `train_physics_fno_gpu.py` | `outputs_gpu/tb/<时间戳>` |

```bash
python scripts/plot_tb_curves.py --logdir <目录> --out curves.png   # 零依赖出图
tensorboard --logdir <目录>                                          # 需 pip install tensorboard
```

DDP 多卡时只有 rank 0 写日志。关闭记录：设 `PN_TB=0`（通用）；非 Hydra 档脚本也可加 `--no-tb`。
详见 [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md#训练曲线tensorboard默认开启)。

## 教材

[book/ch05.md](../book/ch05.md)
