# 第 5 章 · Physics-Informed FNO

## ⚠️ 依赖第 4 章

本目录 **import** `../ch04_fno_airfoil/fno_model.py`。请保留完整仓库，并先跑通 ch04 或至少存在该目录。

若报错 `No module named fno_model`：确认未单独拷贝本章文件夹。

## 先跑这个

```bash
python train_data_fno.py --epochs 50 --n_train 100
python train_physics_fno.py --epochs 50 --n_train 100 --lambda_physics 0.1
```

## 三档脚本

| 档位 | 文件 | 说明 |
|:---|:---|:---|
| **基线 · 纯数据** | `train_data_fno.py` | 无 PDE 正则 |
| **首选 · 混合** | `train_physics_fno.py` | 数据 + Darcy 残差 |
| SDK / GPU | `train_physics_fno_sdk.py` / `train_physics_fno_gpu.py` | |

## 教材

[book/ch05.md](../book/ch05.md)
