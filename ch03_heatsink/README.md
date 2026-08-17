# 第 3 章 · 2D 散热片

## 先跑这个

```bash
# 无 Hydra：--steps；有 Hydra：steps=500
python heat_sink_train.py --steps 500   # CPU 演示可缩短
```

> **Hydra 提示**：见 [COMMAND_REFERENCE](../docs/COMMAND_REFERENCE.md)。

## 三档脚本

| 档位 | 文件 | 依赖 | 说明 |
|:---|:---|:---|:---|
| **首选 · 裸 PyTorch** | `heat_sink_train.py` | torch | 无 `_raw` 后缀；3 鳍、mm 域 |
| SDK | `heat_sink_train_sdk.py` | physicsnemo.sym | **几何不同**（5 鳍、米制），**不可直接对比** |
| GPU | `heat_sink_train_gpu.py` | CUDA | 生产档 |

> ⚠️ raw 与 SDK **不是同一几何问题**；对照学习看 API / Domain·Constraint，勿比绝对 loss。

## 其他

| 文件 | 用途 |
|:---|:---|
| `heat_sink_geometry.py` | CSG 几何 |
| `heat_sink_inverse.py` | 反问题 · 参数扫描（对照） |
| `heat_sink_inverse_joint.py` | 反问题 · **联合训练**（推荐） |
| `validator.py` | CAE 残差报告 → `outputs/validation_report.json` |
| `visualize.py` | 结果可视化 |

训练结束自动生成验证报告。2D 微缩，非 Icepak 3D 签审级。

## 教材

[book/ch03.md](../book/ch03.md)
