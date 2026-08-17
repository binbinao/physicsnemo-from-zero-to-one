# 定性训练基线（参考）

> **非签审标准**；用于判断「训练是否正常」。实测因硬件/种子而异。  
> 硬件与时长见 [docs/HARDWARE_EXPECTATIONS.md](../docs/HARDWARE_EXPECTATIONS.md)。

**验证环境**：Python 3.12 · PyTorch 2.9.x · 微缩默认 epoch · commit `5de0cb7` 前后文档批次。  
**样例目录**：各章 `results/chXX_*/example_record/`（看趋势与数量级，勿死磕小数点）。

---

## ch01 · 弹簧 PINN

| 指标 | 正常趋势 |
|:---|:---|
| `pde_loss` / IC losses | 随 epoch 下降，终值约 1e-4–1e-2 量级 |
| 曲线 vs 解析解 | visually 重合 |
| 训练时长 | 1–5 min（CPU 可） |

脚本：`pinn_spring.py --epochs 2000`（演示可缩短；默认脚本为 5000）  
**实测样例**：[ch01_hello/example_record/](ch01_hello/example_record/)（`final_losses.txt`：pde≈4e-5）

---

## ch02 · 1D 热

| 指标 | 正常趋势 |
|:---|:---|
| PDE / IC / BC loss | 均下降，无一条长期横盘 |
| 终值 PDE | 约 1e-4–1e-3（视 `w_ic`, `w_bc`） |
| 训练时长 | 5–15 min（GPU debug） |

脚本：`heat1d_pinn_raw.py` 或 `heat1d_train.py` + `training=debug`  
**参考样例**：[ch02_heat1d/example_record/](ch02_heat1d/example_record/)

---

## ch03 · 2D 散热

| 指标 | 正常趋势 |
|:---|:---|
| 多条 constraint loss | 同量级下降 |
| 温度场 | 热点在芯片附近，无全场常数 |
| 终值 total | 约 1e-3–1e-2 量级（微缩步数会更高） |
| 训练时长 | 15–60 min |

脚本：`heat_sink_train.py`  
**参考样例**：[ch03_heatsink/example_record/](ch03_heatsink/example_record/)

---

## ch04 · FNO Darcy

| 指标 | 正常趋势 |
|:---|:---|
| train MSE | 随 epoch 下降 |
| test MSE | 与 train 同数量级（微缩数据） |
| 终值 | 约 1e-2–5e-02 量级（50 epoch, n=200） |
| 训练时长 | 10–30 min（8GB GPU） |

脚本：`train_fno_mini.py --epochs 50`  
输出：`outputs/fno_darcy.pt`  
**参考样例**：[ch04_fno_airfoil/example_record/](ch04_fno_airfoil/example_record/)

---

## ch05 · 混合 FNO

| 指标 | 正常趋势 |
|:---|:---|
| 纯数据 baseline | train 很低、test 可能偏高（小数据过拟合） |
| + `lambda_physics=0.1` | test 往往改善或更稳定（约 1e-2–5e-2） |
| 训练时长 | 15–40 min |

脚本：`train_data_fno.py` → `train_physics_fno.py`  
**参考样例**：[ch05_darcy_hybrid/example_record/](ch05_darcy_hybrid/example_record/)

---

## ch06 · AFNO toy 天气

| 指标 | 正常趋势 |
|:---|:---|
| train MSE | 下降（约 1e-2–5e-2） |
| rollout RMSE | 随步数增加而上升（误差累积正常） |
| 训练时长 | 15–45 min |

脚本：`train_afno_mini.py` + `rollout_eval.py`  
**参考样例**：[ch06_fourcastnet_mini/example_record/](ch06_fourcastnet_mini/example_record/)

---

## ch07 · Cd surrogate

| 指标 | 正常趋势 |
|:---|:---|
| val MSE | 随 epoch 下降 |
| val MAE（Cd） | 约 0.01–0.05 量级（toy 数据） |
| 优化后 Cd | 低于参数空间中心点 baseline |
| 训练时长 | 5–20 min |

脚本：`train.py --epochs 200`  
**参考样例**：[ch07_drivaernet_optim/example_record/](ch07_drivaernet_optim/example_record/)

---

## 本地记录模板

```bash
mkdir -p results/ch04_fno_airfoil/$(date +%Y-%m-%d)_mygpu
# 训练后保存：
#   final_losses.txt  — 末 epoch train/test loss
#   notes.md          — GPU 型号、commit、命令行
```

欢迎 PR 补充「实测数值 + 环境」到对应子目录（勿提交大 checkpoint）。
