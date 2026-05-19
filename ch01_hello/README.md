# 第 1 章 · Hello PINN

## 先跑这个

```bash
python pinn_spring.py --epochs 2000
```

零基础第 1 天只需：`pip install torch numpy matplotlib` → 见 [docs/QUICKSTART_DAY1.md](../docs/QUICKSTART_DAY1.md)。

## 三档脚本

| 档位 | 文件 | 依赖 |
|:---|:---|:---|
| **首选 · 裸 PyTorch** | `pinn_spring.py` | torch |
| 对照 · 数据驱动 | `mlp_spring.py` | torch |
| SDK | `pinn_spring_sdk.py` | nvidia-physicsnemo.sym |

## 配置

- `conf/config.yaml` — 仅 SDK 版使用

## 教材

[book/ch01.md](../book/ch01.md)
