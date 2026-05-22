# 资深 CAE 工程师导读

> 已熟悉 ANSYS / STAR-CCM+ / Icepak？本页是 **[START_HERE.md](START_HERE.md)** 的工程补充。

## 阅读顺序

1. [本书物理场范围](#范围) — ch00 §0.6  
2. [代理模型决策树](CAE_SURROGATE_DECISION_TREE.md) — 何时 PINN / FNO / 全量 CFD  
3. 教材 + 命令：[COMMAND_REFERENCE.md](COMMAND_REFERENCE.md)  
4. **签审前必读**：[VV_REPORT_TEMPLATE.md](VV_REPORT_TEMPLATE.md) · [CAE_CONSERVATION_CHECKLIST.md](CAE_CONSERVATION_CHECKLIST.md) · [CAE_OPTIMIZATION_CONSTRAINTS.md](CAE_OPTIMIZATION_CONSTRAINTS.md) · [SAFETY_CRITICAL_LIMITATIONS.md](SAFETY_CRITICAL_LIMITATIONS.md)

## 范围

| 包含 | 不包含 |
|:---|:---|
| PINN / FNO **微缩**与合成数据流程 | 替代签审级 FEM/CFD |
| 代理筛选 + **高保真复核**工作流（文档+ch07 队列脚本） | 自动画网格、求解器 deck 生成 |
| 单位/无量纲、数据 SOP、许可证说明 | NAFEMS 认证课程 |

## 关键脚本（工程向）

| 章 | 验证 / 复核 |
|:---|:---|
| ch03 | `python ch03_heatsink/validator.py` → `outputs/validation_report.json` |
| ch07 | `optimize.py` / `optimize_multi.py` → `hifi_validation_queue.py` → `tools/cfd_batch/` |
| ch03 反演 | `heat_sink_inverse_joint.py`（联合训练）· `heat_sink_inverse.py`（扫描对照） |
| **闭环演示** | `python scripts/run_cae_closed_loop_demo.py` — 见 [CAE_CLOSED_LOOP_DEMO.md](CAE_CLOSED_LOOP_DEMO.md)（mock CFD，无需商业许可证） |

## 数据档位（诚实标注）

| 档位 | ch07 | 用途 |
|:---|:---|:---|
| **toy** | `generate_toy_cars.py` 默认 | 流程演示 |
| **mini/full** | 真实 DrivAerNet / 客户 CFD | 需自建 [CAE_DATA_GENERATION_SOP.md](CAE_DATA_GENERATION_SOP.md) |

## 延伸阅读

- [CAE_UNITS_AND_NONDIM.md](CAE_UNITS_AND_NONDIM.md)  
- [CAE_REFERENCE_VALIDATION.md](CAE_REFERENCE_VALIDATION.md)  
- [LICENSE_NOTES.md](LICENSE_NOTES.md)  
- ASME V&V 40、NAFEMS AI/ML in simulation（见 [WHATS_NEXT.md](WHATS_NEXT.md)）
