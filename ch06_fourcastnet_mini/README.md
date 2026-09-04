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

DDP 多卡时只有 rank 0 写日志。关闭记录：加 `--no-tb`，或设 `PN_TB=0`。
详见 [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md#训练曲线tensorboard默认开启)。

## 教材

[book/ch06.md](../book/ch06.md)
