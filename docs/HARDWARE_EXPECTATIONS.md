# 全书硬件与训练时长预期

> **用途**：选型云 GPU、安排学习计划、判断「慢是否正常」。  
> **基准环境**：微缩默认 epoch / 分辨率；在 **RTX 4070 12GB** 与 **Colab T4 16GB** 上定性验证，非严格 benchmark。  
> **原则**：看**数量级与趋势**；换 CPU、驱动、PyTorch 版本会有偏差。  
> **CAE 免责（C29）**：表中 ch03 等为 **2D 微缩 PINN**；对标 Icepak **3D 瞬态封装** 通常需更大显存与更长训练，不在本书微缩验证范围内。

相关文档：[ENVIRONMENT.md](ENVIRONMENT.md) · [CLOUD_GPU_GUIDE.md](CLOUD_GPU_GUIDE.md) · [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md)

---

## 1. 总览表

| 章 | 首选脚本 | Tier | 最低显存 | CPU | 8GB GPU | 24GB GPU | 训练 wall-clock（微缩） | 训练后单次推理 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 1 | `ch01_hello/pinn_spring.py` | 0 | — | ✅ | ✅ | ✅ | 1–5 min | <1 s |
| 2 | `ch02_heat1d/heat1d_pinn_raw.py` | 0–1 | — | ✅ 慢 | ✅ | ✅ | 5–15 min (CPU 8–20 min) | <0.1 s |
| 3 | `ch03_heatsink/heat_sink_train.py` | 1 | ~6GB | 勉强 | ✅ | ✅ | 15–60 min | <0.5 s |
| 4 | `ch04_fno_airfoil/train_fno_mini.py` | 1 | ~6GB | ❌ | ✅ | ✅ | 10–30 min | <0.1 s |
| 5 | `ch05_darcy_hybrid/train_data_fno.py` | 1 | ~6GB | ❌ | ✅ | ✅ | 15–40 min | <0.1 s |
| 6 | `ch06_fourcastnet_mini/train_afno_mini.py` | 1 | ~8GB | ❌ | ✅ | ✅ 更快 | 15–45 min | 0.1–1 s / step |
| 7 | `ch07_drivaernet_optim/train.py` | 1–3 | ~4GB | ❌ | ✅ toy | ✅ | 5–20 min | <1 ms |

**符号**：✅ 推荐可跑通微缩版；❌ 不建议；**—** 无硬性显存要求（CPU 即可）。

---

## 2. 硬件档位说明

| 配置 | 适合章节 | 说明 |
|:---|:---|:---|
| **CPU 16GB+** | ch01–ch02 完整；ch03 简化 | 无 CUDA 时先完成 PINN 入门 |
| **8GB 消费级 GPU**（3060/4060/4070） | ch01–ch07 微缩版 | 本书默认基线 |
| **16GB**（T4 / 4080） | 同上，batch 可略大 | Colab 免费档约在此档 |
| **24GB+**（3090/4090/A10） | 更大分辨率、更多 epoch、ch06 rollout | 完整版实验 |
| **多卡 A100** | ch06 `train_afno_gpu.py --ddp` | 可选，非必读 |

---

## 3. 分章明细

### 第 1 章 · 弹簧（ch01_hello）

| 项目 | 值 |
|:---|:---|
| 默认规模 | MLP 1k epoch；PINN 2k–5k epoch |
| CPU | ✅ 两个 demo 均可 |
| GPU | 可选，加速不明显 |
| 磁盘 | <100 MB |
| 入口 | `python pinn_spring.py --epochs 2000` |

### 第 2 章 · 1D 热（ch02_heat1d）

| 项目 | 值 |
|:---|:---|
| 默认规模 | `heat1d_pinn_raw` ~3k step；`heat1d_train` + `training=debug` ~500 step |
| CPU | ✅ 约 8–20 min（书中「约 8 分钟」为 GPU/debug 档） |
| 8GB GPU | ✅ 约 5–15 min |
| Hydra multirun | 4 组实验顺序约 **20–40 min**（单卡） |
| 磁盘 | `outputs/` 数十 MB 级 |

### 第 3 章 · 2D 散热片（ch03_heatsink）

| 项目 | 值 |
|:---|:---|
| 默认规模 | 2D three-fin，数千配点 |
| 显存 | 建议 **≥6GB**；CPU 仅适合极小网络调试 |
| 8GB GPU | ✅ 微缩版 15–60 min |
| 行业对比 | Icepak 单工况 ~1h；PINN 训练 30min–2h，推理 ~0.1s（见附录 C.4） |

### 第 4 章 · FNO / Darcy（ch04_fno_airfoil）

| 项目 | 值 |
|:---|:---|
| 数据 | 合成 Darcy 64×64，~200 样本（可自动生成为 `data/darcy_data.pt`） |
| 默认 | `train_fno_mini.py --epochs 30` |
| CPU | ❌ 不推荐（能跑但极慢） |
| 8GB GPU | ✅ 10–30 min |
| 完整 AirfRANS | 建议 **24GB+**，不在微缩默认范围内 |

### 第 5 章 · 混合 FNO（ch05_darcy_hybrid）

| 项目 | 值 |
|:---|:---|
| 前置 | **必须**保留 `ch04_fno_airfoil/`（`check_env --chapter 5` 会检查） |
| 默认 | `train_data_fno.py` / `train_physics_fno.py`，`epochs=50`，`n_train=100` |
| 8GB GPU | ✅ 15–40 min |
| λ 扫描 multirun | 可达 **1–2 h** |

### 第 6 章 · AFNO 天气微缩（ch06_fourcastnet_mini）

| 项目 | 值 |
|:---|:---|
| 数据 | 先 `scripts/generate_toy_weather.py --n_time 200 --resolution 64` |
| 默认 | `train_afno_mini.py --epochs 20–30` |
| 8GB GPU | ✅ 15–45 min |
| Rollout | `rollout_eval.py --rollout_steps 10` 约 **1–5 min** |
| 完整 ERA5 / 多卡 | 云 GPU A100 档，见 [CLOUD_GPU_GUIDE.md](CLOUD_GPU_GUIDE.md) |

### 第 7 章 · 端到端（ch07_drivaernet_optim）

| 项目 | 值 |
|:---|:---|
| 训练 | `train.py --epochs 100–200`，toy 车数据 |
| 优化 | `optimize.py --n_trials 50–500`（Optuna 为 Tier 3） |
| 8GB GPU | ✅ toy 全流程 |
| API | `uvicorn` 启动 `api/app.py`，CPU 即可演示 |
| 磁盘 | 建议预留 **≥5GB**（含 checkpoint） |

---

## 4. 三档脚本与资源

| 档位 | 额外依赖 | 显存 / 时间 |
|:---|:---|:---|
| 裸 PyTorch（`*_raw.py` / 默认 `train.py`） | 仅 Tier 0–1 | 最低 |
| SDK（`*_sdk.py`） | Tier 2 | 与裸 PyTorch 同量级或略高 |
| GPU 生产（`*_gpu.py`、DDP） | Tier 2 + CUDA | 多卡时按 GPU 数近线性扩展 |

---

## 5. 磁盘与网络

| 项 | 建议 |
|:---|:---|
| 仓库 + venv | ~2GB |
| 各章 `outputs/`、`data/*.pt` | 每章 0.1–2GB，已在 `.gitignore` |
| 全书本地开发 | **≥30GB** 空闲（见附录 B） |
| 数据下载 | 微缩版**无需**外网数据集；Colab 需能 `git clone` |

---

## 6. 参考 loss / 指标（定性）

> 非签审基线；仅用于判断「是否离谱」。实测请在本机记录。

| 章 | 现象（正常趋势） |
|:---|:---|
| ch01 | PINN 三条 loss 均下降；曲线与解析解重合 |
| ch02 | PDE / IC / BC loss 同量级下降；最终 PDE ~1e-4–1e-3 量级（视权重而定） |
| ch04 | train/test MSE 随 epoch 下降；test 不应长期高于 train 一个数量级以上（小数据除外） |
| ch05 | 加 `lambda_physics` 后小数据 val 往往优于纯数据 |
| ch07 | val MSE 随 epoch 下降；优化后 Cd 低于 baseline 中心点 |

更细的数值对照见 [results/BASELINE.md](../results/BASELINE.md)。

---

## 7. 与 `check_env` 联动

```bash
python scripts/check_env.py --list      # Tier + GPU 建议
python scripts/check_env.py --chapter 4 # 该章必需包 + GPU 提示
```

---

## 8. 书中阅读时长（参考）

| 章 | 快速通道 | 深入阅读 |
|:---|:---|:---|
| ch00–ch01 | 30–45 min | 1–1.5 h |
| ch02 | 15 min 跑通 | ~90 min |
| ch03–ch04 | 20–30 min | 2 h+ |
| ch05–ch07 | 30 min+ | 2 h+ |

全书 6 周计划见 [STUDY_PLAN_6WEEKS.md](STUDY_PLAN_6WEEKS.md)。
