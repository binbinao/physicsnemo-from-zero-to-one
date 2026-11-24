# 第 6 章 · Mini FourCastNet (AFNO)

## 先跑这个

```bash
# 数据可自动生成；也可手动：
python scripts/generate_toy_weather.py --n_time 200 --resolution 64
# 无 Hydra：--epochs；有 Hydra：epochs=30
python train_afno_mini.py --epochs 30
python rollout_eval.py --ckpt outputs/afno_weather.pt --rollout_steps 10
```

> **Hydra / argparse 双语法**：装 `hydra-core` 时用 `key=value`（本章如 `epochs=30`）；未装时用 `--flag`（如 `--epochs 30`）。键名以本章 `conf/config.yaml` 为准。见 [COMMAND_REFERENCE](../docs/COMMAND_REFERENCE.md)。  
> **无多卡**：跳过 `train_afno_gpu.py` 的 DDP；单卡/CPU 用 `train_afno_mini.py` 即可。

## 三档脚本

| 档位 | 文件 | 依赖 | 说明 |
|:---|:---|:---|:---|
| **首选 · 裸 PyTorch** | `train_afno_mini.py` | torch | toy 天气 + AFNO |
| SDK | `train_afno_sdk.py` | physicsnemo | PhysicsNeMo AFNO |
| GPU / DDP | `train_afno_gpu.py` | CUDA | **需多卡才有意义**；无多卡可跳过 |

## 数据

- 无 ERA5 时：用 `scripts/generate_toy_weather.py`（仓库未含 `download_era5_subset.sh`）。
- `dataset.py` 缺文件时也会自动调用生成脚本。

## 查看训练曲线

训练默认记录曲线，结束后会打印日志目录：

| 脚本 | 默认日志目录 |
|:---|:---|
| `train_afno_mini.py` / `train_afno_sdk.py` | `outputs/tb/<时间戳>` |
| `train_afno_gpu.py` | `outputs_gpu/tb/<时间戳>` |

```bash
python scripts/plot_tb_curves.py --logdir <目录> --out curves.png   # 零依赖出图
tensorboard --logdir <目录>                                          # 需 pip install tensorboard
```

DDP 多卡时只有 rank 0 写日志。关闭记录：设 `PN_TB=0`（通用）；非 Hydra 档脚本也可加 `--no-tb`。
详见 [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md#训练曲线tensorboard默认开启)。

## 教材

[book/ch06.md](../book/ch06.md)
