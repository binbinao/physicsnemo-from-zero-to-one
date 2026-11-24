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

- **toy（默认，开箱可跑）**：`scripts/generate_toy_weather.py` 生成 4 通道合成天气场（温度/位势/纬向风/经向风），仅用于验证数据管线与训练/rollout 能跑通，**不代表真实预报技能**。数据文件不存在时 `dataset.py` 会自动调用它生成。
- **真实数据（需自备）**：仓库只提供 toy 生成器，**不含** ERA5 下载脚本（设计如此，不是文件缺失）。ERA5 需自行从 ECMWF Copernicus Climate Data Store（CDS）或 ARCO-ERA5 获取；本仓库不承诺下载链接、账号配额与接口稳定性。
- **落点与格式**：自备数据须与 toy 同构 —— `<data_dir>/toy_weather.pt` = `dict{'data': float32 张量 (T, C, H, W)}`；或把 `conf/config.yaml` 的 `data_dir` 指向自备目录（仅 `train_afno_mini.py` 生效、改配置即可；SDK / GPU 脚本固定读 `ch06_fourcastnet_mini/data/`，需把文件放到该目录）。该 `.pt` 已被 `.gitignore` 忽略，不入库。

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
