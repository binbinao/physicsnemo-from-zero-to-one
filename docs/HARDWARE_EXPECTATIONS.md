# 全书硬件与训练时长预期

> 供云 GPU 选型与学习计划参考；实际时间因机器、随机种子而异。

| 章节 | 脚本（首选） | CPU | 8GB GPU | 24GB GPU | 大致 wall-clock |
|:---|:---|:---|:---|:---|:---|
| ch01 | `mlp_spring.py` / `pinn_spring.py` | ✅ | ✅ | ✅ | 1–5 min |
| ch02 | `heat1d_pinn_raw.py` | ✅ 慢 | ✅ | ✅ | 5–15 min |
| ch03 | `heat_sink_train.py` | 简化可跑 | ✅ | ✅ | 15–60 min |
| ch04 | `train_fno_mini.py` | ❌ 不推荐 | ✅ | ✅ | 10–30 min |
| ch05 | `train_data_fno.py` | ❌ | ✅ | ✅ | 15–40 min |
| ch06 | `train_afno_mini.py` | ❌ | ✅ | ✅ 更快 | 15–45 min |
| ch07 | `train.py` | ❌ | ✅ toy | ✅ | 5–20 min |

**符号**：✅ 可跑通微缩版；❌ 不建议。

**云 GPU 分步指南**：[CLOUD_GPU_GUIDE.md](CLOUD_GPU_GUIDE.md)
