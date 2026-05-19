# 第 6 章 · Mini FourCastNet (AFNO)

## 先跑这个

```bash
python scripts/generate_toy_weather.py --n_time 200 --resolution 64
python train_afno_mini.py --epochs 20
python rollout_eval.py --ckpt outputs/afno_weather.pt --rollout_steps 10
```

## 三档脚本

| 档位 | 文件 |
|:---|:---|
| **首选** | `train_afno_mini.py` |
| SDK | `train_afno_sdk.py` |
| GPU / DDP | `train_afno_gpu.py` |

## 数据

- 无 ERA5 时：用 `scripts/generate_toy_weather.py`（仓库未含 `download_era5_subset.sh`）。

## 教材

[book/ch06.md](../book/ch06.md)
