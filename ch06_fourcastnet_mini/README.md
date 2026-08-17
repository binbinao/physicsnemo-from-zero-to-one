# 第 6 章 · Mini FourCastNet (AFNO)

## 先跑这个

```bash
# 数据可自动生成；也可手动：
python scripts/generate_toy_weather.py --n_time 200 --resolution 64
# 无 Hydra：--epochs；有 Hydra：epochs=20
python train_afno_mini.py --epochs 20
python rollout_eval.py --ckpt outputs/afno_weather.pt --rollout_steps 10
```

> **Hydra 提示**：见 [COMMAND_REFERENCE](../docs/COMMAND_REFERENCE.md)。  
> **无多卡**：跳过 `train_afno_gpu.py` 的 DDP；单卡/CPU 用 `train_afno_mini.py` 即可。

## 三档脚本

| 档位 | 文件 | 说明 |
|:---|:---|:---|
| **首选 · 裸 PyTorch** | `train_afno_mini.py` | toy 天气 + AFNO |
| SDK | `train_afno_sdk.py` | PhysicsNeMo AFNO |
| GPU / DDP | `train_afno_gpu.py` | **需多卡才有意义**；无多卡可跳过 |

## 数据

- 无 ERA5 时：用 `scripts/generate_toy_weather.py`（仓库未含 `download_era5_subset.sh`）。
- `dataset.py` 缺文件时也会自动调用生成脚本。

## 教材

[book/ch06.md](../book/ch06.md)
