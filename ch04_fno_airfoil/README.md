# 第 4 章 · FNO（Darcy 微缩 + 翼型数据）

## 先跑这个

```bash
python train_fno_mini.py --epochs 30
# 或 Hydra: python train_fno_mini.py epochs=30
```

**说明**：`train_fno_mini.py` 默认训练 **Darcy 渗流**合成数据（`data/darcy_data.pt`）。章名 `airfoil` 表示工业场景；翼型压力场可用 `dataset.py --type airfoil` 生成后由 `visualize_airfoil.py` 查看。

## 两条路径

| 路径 | 命令 | 显存 |
|:---|:---|:---|
| **微缩（默认）** | `train_fno_mini.py` | ~8GB |
| 翼型合成数据 | `python dataset.py --type airfoil` | 仅数据生成 |
| SDK / GPU | `train_fno_sdk.py` / `train_fno_gpu.py` | 8GB+ |

## 框架切换

第 4 章起使用 **physicsnemo 主框架**（非 sym）。见 [book/ch04.md](../book/ch04.md) §4.1。

## 教材

[book/ch04.md](../book/ch04.md)
