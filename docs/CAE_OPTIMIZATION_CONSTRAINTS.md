# 优化约束说明（C21）

本书 `optimize.py` 仅含 **演示用** 惩罚项（车长、离地间隙）。工程落地应增加：

| 约束类型 | 示例 |
|:---|:---|
| 几何 | 包络、最小离地间隙、视野法规 |
| 制造 | 冲压角度、曲率半径 |
| 性能 | 升力下限、侧风稳定性 |
| 热 / 结构 | 格栅风量、骨架碰撞（需另模块） |

实现建议：在 `objective` 中对违反约束的设计返回大罚值，或改用约束优化器；**优化结果必须**进入高保真复核流程。

本书已实现：

- [design_constraints.py](../ch07_drivaernet_optim/design_constraints.py) — 演示级硬/软约束  
- [optimize_multi.py](../ch07_drivaernet_optim/optimize_multi.py) `--export-hifi` — 仅导出可行 Pareto 解到 `hifi_queue.csv`  
- [CAE_CLOSED_LOOP_DEMO.md](CAE_CLOSED_LOOP_DEMO.md) — mock CFD 闭环（或 `hifi_validation_queue.py` 单目标队列）
