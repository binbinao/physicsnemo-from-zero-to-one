# 第 1 天最小安装（仅 ch01 · 裸 PyTorch）

> 目标：**30 分钟内**完成环境 + 跑通弹簧 PINN，**无需**安装 PhysicsNeMo / CUDA。

## 1. 检查 Python

```bash
python3 --version   # 需要 >= 3.10
```

## 2. 创建环境（推荐）

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
```

## 3. 只装最小依赖

```bash
pip install -r requirements-minimal.txt
# 或等价：pip install "torch>=2.3" numpy matplotlib
```

不要在这一步安装 `nvidia-physicsnemo`（留给第 2 天或跑 SDK 版时再说）。

## 4. 克隆并自检

```bash
git clone https://github.com/binbinao/physicsnemo-from-zero-to-one.git
cd physicsnemo-from-zero-to-one
python scripts/check_env.py --tier 0
# 或：python scripts/check_env.py --chapter 1
```

**预期**：Python ✅、PyTorch ✅；CUDA / PhysicsNeMo ❌ 也**没关系**。

## 5. 跑通两个 demo

```bash
cd ch01_hello
python mlp_spring.py --epochs 500
python pinn_spring.py --epochs 2000
```

会弹出 loss 曲线图；若在无图形界面服务器上，可忽略弹窗或改用 `MPLBACKEND=Agg`。

## 6. 第 2 天起

| 天 | 建议 |
|:---|:---|
| Day 2 | ch02 `heat1d_pinn_raw.py`；安装 `hydra-core` 可选 |
| Day 3+ | 有 GPU 再进 ch04；见 [ENVIRONMENT.md](ENVIRONMENT.md) Tier 2 |
| 无 GPU | [Colab 笔记本](../notebooks/colab_quickstart.ipynb) |

## 常见问题

- **没有 GPU**：ch01–ch02 用 CPU 即可。  
- **想装 PhysicsNeMo**：见 [ENVIRONMENT.md](ENVIRONMENT.md) Tier 2。  
- **报错**：见 [附录 C](../book/appendix_c_troubleshooting.md)。
