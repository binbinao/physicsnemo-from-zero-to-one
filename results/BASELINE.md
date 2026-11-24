# 定性训练基线（参考）

> **非签审标准**；用于判断「训练是否正常」。实测因硬件/种子而异。  
> 硬件与时长见 [docs/HARDWARE_EXPECTATIONS.md](../docs/HARDWARE_EXPECTATIONS.md)。

**验证环境**：Python 3.12 · PyTorch 2.9.x · 各章脚本**默认参数**（下表逐章列出，以脚本 `--help` 为准）。  
**是否本轮复测**：否。本文件只给**定性趋势**；`results/` 下只有 ch01 的一份历史示例记录，其余章的真实记录按 [README.md](README.md) 的模板征集。  
**关于「约 …」**：下表的量级一律是**经验区间（非实测、本轮未复测）**，只用于判断「训练是否跑偏」；它们不是签审数值，也不得被抄成 `results/**` 下的数据记录（那里只收真实运行结果）。

---

## ch01 · 弹簧 PINN

| 指标 | 正常趋势 |
|:---|:---|
| `pde_loss` / IC losses | 随 epoch 下降，终值约 1e-4–1e-2 量级 |
| 曲线 vs 解析解 | visually 重合 |
| 训练时长 | 1–5 min（CPU 可） |

脚本：`pinn_spring.py`（默认 `--epochs 5000`；快速演示可 `--epochs 2000`）  
**示例记录（历史提交，未经本轮复测）**：[ch01_hello/example_record/](ch01_hello/example_record/)（`final_losses.txt`：pde≈4e-5）

---

## ch02 · 1D 热

| 指标 | 正常趋势 |
|:---|:---|
| PDE / IC / BC loss | 均下降，无一条长期横盘 |
| 终值 PDE | 约 1e-4–1e-3（视 `w_ic`, `w_bc`） |
| 训练时长 | 5–15 min（GPU debug） |

脚本：`heat1d_pinn_raw.py`（默认 `--steps 3000`；CPU 短跑 `--steps 500`）或 `heat1d_train.py`（默认 `steps=3000`）+ `training=debug`

---

## ch03 · 2D 散热

| 指标 | 正常趋势 |
|:---|:---|
| 多条 constraint loss | 同量级下降 |
| 温度场 | 热点在芯片附近，无全场常数 |
| 终值 total | 约 1e-3–1e-2 量级（经验区间；微缩步数会更高） |
| 训练时长 | 15–60 min |

脚本：`heat_sink_train.py`（默认 `steps=3000`；CPU 演示可 `--steps 500`）

---

## ch04 · FNO Darcy

| 指标 | 正常趋势 |
|:---|:---|
| train MSE | 随 epoch 下降 |
| test MSE | 与 train 同数量级（微缩数据） |
| 终值 | 约 1e-2–5e-2 量级（经验区间；短程微缩训练通常更高） |
| 训练时长 | 10–30 min（8GB GPU） |

脚本：`train_fno_mini.py`（默认 `--epochs 30`；加长演示可 `--epochs 50`）  
输出：`outputs/fno_darcy.pt`

---

## ch05 · 混合 FNO

| 指标 | 正常趋势 |
|:---|:---|
| 纯数据 baseline | train 很低、test 可能偏高（小数据过拟合） |
| + `lambda_physics=0.1` | test 往往改善或更稳定（经验区间 约 1e-2–5e-2） |
| 训练时长 | 15–40 min |

脚本：`train_data_fno.py` → `train_physics_fno.py`（默认 `epochs=50`、`n_train=100`）

---

## ch06 · AFNO toy 天气

| 指标 | 正常趋势 |
|:---|:---|
| train MSE | 下降（经验区间 约 1e-2–5e-2） |
| rollout RMSE | 随步数增加而上升（误差累积正常） |
| 训练时长 | 15–45 min |

脚本：`train_afno_mini.py`（默认 `epochs=30`）+ `rollout_eval.py`

---

## ch07 · Cd surrogate

| 指标 | 正常趋势 |
|:---|:---|
| val MSE | 随 epoch 下降 |
| val MAE（Cd） | 约 0.01–0.05 量级（toy 数据） |
| 优化后 Cd | 低于参数空间中心点 baseline |
| 训练时长 | 5–20 min |

脚本：`train.py`（默认 `--epochs 200`）

---

## 本地记录模板

```bash
mkdir -p results/ch04_fno_airfoil/$(date +%Y-%m-%d)_mygpu
# 训练后保存：
#   final_losses.txt  — 末 epoch train/test loss
#   notes.md          — GPU 型号、commit、命令行、种子、日期
```

要求与禁止项见 [README.md](README.md)：**只收真实跑出来的数值**，示意值 / 区间值 / 未实测数值一律不入库。
