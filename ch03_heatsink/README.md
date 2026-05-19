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
| `heat_sink_inverse.py` | 反问题 |
| `visualize.py` | 结果可视化 |

## 教材

[book/ch03.md](../book/ch03.md)
