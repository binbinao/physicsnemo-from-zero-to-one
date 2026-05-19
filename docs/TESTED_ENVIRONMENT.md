# 测试通过环境记录

> 用于复现本书微缩实验时的参考快照。**依赖仍以 `>=` 为主**；若安装失败，以 [ENVIRONMENT.md](ENVIRONMENT.md) 为准并提 Issue。

## 最近验证

| 项 | 值 |
|:---|:---|
| **Git commit** | `5de0cb7`（2026-05，小白 issue 批次完成后） |
| **Python** | 3.12 |
| **PyTorch** | 2.9.x（CUDA 与 CPU 均测过 check_env） |
| **OS** | macOS / Linux |

```bash
git checkout 5de0cb7   # 可选：锁定到验证提交
pip install -r requirements-minimal.txt
python scripts/check_env.py --chapter 1
```

## 安装命令

```bash
# 第 1 天
pip install -r requirements-minimal.txt

# SDK + ch07
pip install -r requirements-full.txt
```

## 说明

- PhysicsNeMo 版本随 NVIDIA 发布变化，**不锁死小版本**；SDK 脚本以仓库 `*_sdk.py` 为准。
- 严格复现请在 PR/Issue 中附上 `pip freeze` 与 GPU 型号。
