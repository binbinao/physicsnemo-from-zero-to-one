# 6 周学习计划

> 假设每周 **6–8 小时**（含阅读 + 跑代码）。时间紧可走 🟢 快速通道；想发表/落地走 🔵 深入。

相关：[HARDWARE_EXPECTATIONS.md](HARDWARE_EXPECTATIONS.md) · [QUICKSTART_DAY1.md](QUICKSTART_DAY1.md) · [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md)

---

## 读者自测（先选路径）

勾选最符合你的一项：

1. 我主要做 **CAE/仿真**，想先能跑通 demo 再给同事看。→ **🟢 快速通道为主**
2. 我有 **DL 基础**，想搞懂 PINN/FNO 原理与调参。→ **🔵 深入阅读**
3. 我 **几乎零基础**（Python 勉强、没用过 PyTorch）。→ **🟡 先附录 D + Day1，再跟 🟢**

4. 我每周能投入 **<4 小时** → 把计划拉长到 **8–10 周**，每周减半任务即可。  
5. 我有 **GPU 云主机** → 第 4 周起用 [Colab 笔记本](../notebooks/colab_quickstart.ipynb) 或 [CLOUD_GPU_GUIDE.md](CLOUD_GPU_GUIDE.md)。

---

## 第 1 周 · 环境与 PINN 入门

| 目标 | 任务 |
|:---|:---|
| 环境 | [QUICKSTART_DAY1](QUICKSTART_DAY1.md) + `check_env --tier 0` |
| 阅读 | ch00 前言、ch01 快速通道 + 1.3–1.6 |
| 代码 | `mlp_spring.py`、`pinn_spring.py` |
| 可选 | [附录 D](../book/appendix_d_pytorch_mini.md) |

**周末检查**：能解释 PINN 三件套损失是什么。

---

## 第 2 周 · 1D 热传导 + Hydra

| 目标 | 任务 |
|:---|:---|
| 阅读 | ch02 快速通道 + PDE/配点节 |
| 代码 | `heat1d_pinn_raw.py` 或 `heat1d_train.py` |
| 文档 | `w_ic` 调参实验（正文 2.7 可 skim） |

**周末检查**：能改 `conf/training/debug.yaml` 并重新训练。

---

## 第 3 周 · 2D 散热片

| 目标 | 任务 |
|:---|:---|
| 阅读 | ch03 快速通道 + 几何/BC 节 |
| 代码 | `heat_sink_train.py` |
| 硬件 | 无 GPU 可用 Colab 跑 ch03 |

**周末检查**：说出 Domain / Constraint 在 sym 里各代表什么。

---

## 第 4 周 · FNO 与框架切换 ★

| 目标 | 任务 |
|:---|:---|
| 必读 | [CH04_GUIDE.md](../ch04_fno_airfoil/CH04_GUIDE.md)（翼型 vs Darcy） |
| 阅读 | ch04 §4.1 框架切换 + 快速通道 |
| 代码 | `train_fno_mini.py`（**需 GPU 或 Colab**） |
| 环境 | `check_env --chapter 4` |

**周末检查**：能区分 sym（ch1–3）与 main 框架（ch4–7）。

---

## 第 5 周 · 物理信息 FNO

| 目标 | 任务 |
|:---|:---|
| 前置 | 确认 ch04 目录存在 |
| 阅读 | ch05 快速通道 |
| 代码 | `train_data_fno.py` → `train_physics_fno.py` |

**周末检查**：能解释 λ_physics 变大时 val loss 可能如何变化。

---

## 第 6 周 · 时序 + 端到端 + 收尾

| 目标 | 任务 |
|:---|:---|
| 代码 | ch06 `generate_toy_weather` + `train_afno_mini` |
| 代码 | ch07 `train.py` → `optimize.py`（可选 API） |
| 阅读 | [WHATS_NEXT.md](WHATS_NEXT.md) |
| 复习 | 附录 A/B/C，[BEGINNER_ISSUE_BACKLOG](BEGINNER_ISSUE_BACKLOG.md) 全绿 |

**毕业检查**：独立完成「从 clone 到 ch04 跑通」不查书超过 3 处。

---

## 三条路径时间分配（参考）

| 路径 | 每周阅读 : 代码 | 略读章节 |
|:---|:---|:---|
| 🟢 快速 | 30% : 70% | 调参推导、Failure 长文 |
| 🔵 深入 | 60% : 40% | 全读 + 做笔记 |
| 🟡 零基础 | 先附录 D，再按周表 | ch00 可拆两周 |

---

## 进度跟踪（可复制）

```markdown
- [ ] W1 ch01
- [ ] W2 ch02
- [ ] W3 ch03
- [ ] W4 ch04 + CH04_GUIDE
- [ ] W5 ch05
- [ ] W6 ch06 + ch07 + WHATS_NEXT
```

---

## 章节小测验（自测，答案见下）

1. **ch01**：PINN 相比 MLP 多出来的损失项是什么？（提示：PDE / IC）
2. **ch02**：Hydra 里加大 `w_ic` 一般为了压哪类误差？
3. **ch03**：`Domain` 与 `Constraint` 各描述什么？
4. **ch04**：为何目录叫 `airfoil` 默认脚本却训 Darcy？
5. **ch05**：`lambda_physics` 变大时，物理残差项在总损失里占什么变化？

<details>
<summary>参考答案（先自己做再点开）</summary>

1. PDE 残差损失（+ 边界/初值），可不依赖标注数据。  
2. 初值（IC）误差；过大可能导致 PDE 项相对变弱。  
3. Domain=求解域与方程；Constraint=边界/初值等约束。  
4. 工业叙事用翼型；教学默认用更小 Darcy 合成数据跑通 FNO 循环（见 CH04_GUIDE）。  
5. 物理残差权重上升；小数据时 test 可能更稳，但过大可能拖累数据拟合。

</details>
