# 测试通过环境记录

> 用于复现本书微缩实验时的参考快照。**依赖仍以 `>=` 为主**；若安装失败，以 [ENVIRONMENT.md](ENVIRONMENT.md) 为准并提 Issue。

## 最近验证

| 项 | 值 |
|:---|:---|
| **代码基线** | 以 `main` 最新为准（`git rev-parse --short HEAD`），不记录固定 commit |
| **记录时点** | 2026-05-19（本文件最近一次人工记录）。此后 `main` 的改动**未纳入**本快照 |
| **Python** | 3.12 |
| **PyTorch** | 2.9.x（CUDA 与 CPU 均测过 check_env） |
| **OS** | macOS / Linux |

> 新鲜度自查：`grep -m1 '记录时点' docs/TESTED_ENVIRONMENT.md`，与 `date -I` 对比即知该快照有多旧。

```bash
git fetch origin && git checkout main && git pull   # 与 Colab/协作者保持一致
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
