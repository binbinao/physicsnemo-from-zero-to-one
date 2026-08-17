# 第 1 章 · Hello PINN

## 先跑这个

```bash
python pinn_spring.py --epochs 5000
```

（脚本默认亦为 5000；快速演示可改 `--epochs 2000`。）

零基础第 1 天只需：`pip install torch numpy matplotlib` → 见 [docs/QUICKSTART_DAY1.md](../docs/QUICKSTART_DAY1.md)。

## 三档脚本

| 档位 | 文件 | 依赖 | 说明 |
|:---|:---|:---|:---|
| **首选 · 裸 PyTorch** | `pinn_spring.py` | torch | 本章无 `_raw` 后缀 |
| 对照 · 数据驱动 | `mlp_spring.py` | torch | 与 PINN 对比用 |
| SDK | `pinn_spring_sdk.py` | nvidia-physicsnemo.sym | 声明式 API 预览 |

> 本章无独立 `*_gpu.py`；CPU 即可。
## 配置

- `conf/config.yaml` — 仅 SDK 版使用

## 教材

[book/ch01.md](../book/ch01.md)
