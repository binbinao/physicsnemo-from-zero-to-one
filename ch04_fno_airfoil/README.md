# 第 4 章 · FNO（翼型场景 · Darcy 默认训练）

> **困惑「目录叫 airfoil 为何跑 Darcy」？** → 读 [CH04_GUIDE.md](CH04_GUIDE.md)（一页纸 + 路径图）

## 先跑这个（路径 A · 默认）

```bash
# 无 Hydra：--epochs；有 Hydra：epochs=30
python train_fno_mini.py --epochs 30
# Hydra: python train_fno_mini.py epochs=30
```

生成/加载 `data/darcy_data.pt`，checkpoint 一般为 `outputs/fno_darcy.pt`。
需 **scipy**（`requirements-minimal.txt` 已含）；先跑 `python ../scripts/check_env.py --chapter 4`。

> **Hydra / argparse 双语法**：装 `hydra-core` 时用 `key=value`（如 `epochs=30`、`steps=500`）；未装时用 `--flag`（如 `--epochs 30`、`--steps 500`）。见 [COMMAND_REFERENCE](../docs/COMMAND_REFERENCE.md)。

## 两条路径速查

| 路径 | 做什么 | 命令 |
|:---|:---|:---|
| **A · Darcy（默认）** | 练通 FNO 训练循环 | `train_fno_mini.py` |
| **B · 翼型合成** | 仅生成/查看翼型式合成数据 | `dataset.py --type airfoil` → `visualize_airfoil.py`（默认 ckpt 为 Darcy，仅作演示） |
| **B · 完整翼型** | AirfRANS / 真实 CFD 数据 | 见 `book/ch04.md` 完整版节，24GB+ |

## 三档脚本

| 档位 | 文件 | 说明 |
|:---|:---|:---|
| **首选 · 裸 PyTorch** | `train_fno_mini.py` | 路径 A；手写 FNO |
| SDK | `train_fno_sdk.py` | 同 Darcy 数据，换 PhysicsNeMo backbone |
| GPU | `train_fno_gpu.py` | DDP / 生产档 |

## 框架切换

第 4 章起使用 **physicsnemo 主框架**（非 sym）。对比表见 [CH04_GUIDE.md §3](CH04_GUIDE.md#3-框架切换ch3--ch4) 与 [book/ch04.md §4.1](../book/ch04.md)。

## 查看训练曲线

训练默认记录曲线，结束后会打印日志目录：

| 脚本 | 默认日志目录 |
|:---|:---|
| `train_fno_mini.py` / `train_fno_sdk.py` | `outputs/tb/<时间戳>` |
| `train_fno_gpu.py` | `outputs_gpu/tb/<时间戳>` |

```bash
python scripts/plot_tb_curves.py --logdir <目录> --out curves.png   # 零依赖出图
tensorboard --logdir <目录>                                          # 需 pip install tensorboard
```

DDP 多卡时只有 rank 0 写日志。关闭记录：设 `PN_TB=0`（通用）；非 Hydra 档脚本也可加 `--no-tb`。
详见 [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md#训练曲线tensorboard默认开启)。

## 教材

[book/ch04.md](../book/ch04.md)
