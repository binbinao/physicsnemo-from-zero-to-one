# 第 2 章 · 1D 热传导 PINN

## 先跑这个

```bash
python heat1d_pinn_raw.py
# 或（Hydra）
python heat1d_train.py
```

## 三档脚本

| 档位 | 文件 | 依赖 |
|:---|:---|:---|
| **首选 · 裸 PyTorch** | `heat1d_pinn_raw.py` | torch |
| Hydra 训练 | `heat1d_train.py` | torch, hydra-core |
| SDK | `heat1d_pinn_sdk.py` | physicsnemo |
| GPU | `heat1d_pinn_gpu.py` | CUDA |

## 可视化

```bash
python heat1d_visualize.py
```

## 命令参考

[docs/COMMAND_REFERENCE.md](../docs/COMMAND_REFERENCE.md#ch02hydraconfig-在-conf)

## 教材

[book/ch02.md](../book/ch02.md)
