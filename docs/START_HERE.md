# 从这里开始（中文读者）

> 第一次打开本仓库？按下面顺序走，**不要**从 README 英文目录树开始迷路。

## 1. 读教材前言（10 分钟）

[book/ch00.md](../book/ch00.md) — 为什么学 PhysicsNeMo、全书地图、硬件预期。

## 2. 第 1 天只装最小依赖（30 分钟）

```bash
git clone https://github.com/binbinao/physicsnemo-from-zero-to-one.git
cd physicsnemo-from-zero-to-one
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-minimal.txt
python scripts/check_env.py --chapter 1
python ch01_hello/pinn_spring.py --epochs 500
```

详细步骤：[QUICKSTART_DAY1.md](QUICKSTART_DAY1.md)

## 3. 按周学习（可选）

[STUDY_PLAN_6WEEKS.md](STUDY_PLAN_6WEEKS.md) · 命令对照：[COMMAND_REFERENCE.md](COMMAND_REFERENCE.md)

## 4. 关键「别踩坑」一页纸

| 困惑 | 去看 |
|:---|:---|
| 目录 `fno_airfoil` 为何跑 Darcy？ | [ch04_fno_airfoil/CH04_GUIDE.md](../ch04_fno_airfoil/CH04_GUIDE.md) |
| ch3 用 sym，ch4 换主框架？ | [FRAMEWORK_SWITCH.md](FRAMEWORK_SWITCH.md) |
| 没 GPU | [CLOUD_GPU_GUIDE.md](CLOUD_GPU_GUIDE.md) · [Colab](../notebooks/colab_quickstart.ipynb) |
| 装不上 PhysicsNeMo | [ENVIRONMENT.md](ENVIRONMENT.md) §「SDK 装不上」 |
| 训练 loss 是否正常？ | [results/BASELINE.md](../results/BASELINE.md) |
| 每章一页纸 | [cheatsheets/](cheatsheets/README.md) |
| SDK 版本 | [SDK_COMPATIBILITY.md](SDK_COMPATIBILITY.md) |

## 5. 仓库里有什么

| 路径 | 内容 |
|:---|:---|
| `book/` | 教材正文 ch00–ch07 + 附录 A–D |
| `book/assets/` | 正文插图 PNG |
| `assets/` | 仓库级资源（如公众号二维码占位图） |
| `ch01_hello/` … `ch07_*/` | 可运行代码（每章 README 写首选脚本） |
| `docs/` | 环境、学习计划、命令表、本页 |

## 6. 版本与 Colab

- 环境快照：[TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT.md)
- **请使用 GitHub 上 `main` 最新提交**；若 Colab/同学 clone 的代码较旧，在仓库首页核对最新 commit 时间。
- Colab：打开 [notebooks/colab_quickstart.ipynb](../notebooks/colab_quickstart.ipynb) 前，可先 `git pull` 或重新 clone。

## 7. 读完全书

[WHATS_NEXT.md](WHATS_NEXT.md)

---

**English quick start**: see [README](../README.md) · **Issue 跟踪**：[REVIEW_ROUND2_BACKLOG.md](REVIEW_ROUND2_BACKLOG.md)
