# 第 3 章 · 2D 散热片

## 先跑这个

```bash
python heat_sink_train.py
```

## 三档脚本

| 档位 | 文件 | 依赖 |
|:---|:---|:---|
| **首选 · 裸 PyTorch** | `heat_sink_train.py` | torch |
| SDK | `heat_sink_train_sdk.py` | physicsnemo.sym |
| GPU | `heat_sink_train_gpu.py` | CUDA |

## 其他

| 文件 | 用途 |
|:---|:---|
| `heat_sink_geometry.py` | CSG 几何 |
| `heat_sink_inverse.py` | 反问题 · 参数扫描（对照） |
| `heat_sink_inverse_joint.py` | 反问题 · **联合训练**（推荐） |
| `validator.py` | CAE 残差报告 → `outputs/validation_report.json` |
| `visualize.py` | 结果可视化 |
| `inference/` | **生产推理包**：ONNX 制品 + FastAPI 服务（见 [inference/README.md](inference/README.md)） |
| `EXPERIMENTS_PASS.md` | **追 PASS 实验结论文档**：死锁机制/配方/Fourier 证据链/robin 量纲假说（后续改进项） |

训练结束自动生成验证报告。2D 微缩，非 Icepak 3D 签审级。

## 查看训练曲线

训练默认记录曲线，结束后会打印日志目录：

| 脚本 | 默认日志目录 |
|:---|:---|
| `heat_sink_train.py` / `heat_sink_train_sdk.py` | `outputs/tb/<时间戳>` |
| `heat_sink_train_gpu.py` | `outputs_gpu/tb/<时间戳>` |
| `heat_sink_inverse.py` / `heat_sink_inverse_joint.py` | `outputs/tb/<时间戳>` |

```bash
python scripts/plot_tb_curves.py --logdir <目录> --out curves.png   # 零依赖出图
tensorboard --logdir <目录>                                          # 需 pip install tensorboard
```

DDP 多卡时只有 rank 0 写日志。关闭记录：加 `--no-tb`，或设 `PN_TB=0`。
详见 [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md#训练曲线tensorboard默认开启)。

## 教材

[book/ch03.md](../book/ch03.md)
