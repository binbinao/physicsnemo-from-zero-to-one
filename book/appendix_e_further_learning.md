# 附录 E · 学完之后：继续学习路线图

> **用途**：读完前言 + 第 1–7 章后，用本附录做「能力盘点」与「下一程选路」。  
> **配套行动清单**（勾选表、30 天计划）：[docs/WHATS_NEXT.md](../docs/WHATS_NEXT.md)。  
> **不是**第二本教材；下列推荐按「先会用、再加深、再上生产」排序。

---

## E.0 全书审阅小结：你已经走完什么

本书的主线不是「堆模型名」，而是一条可交付的 AI4Science 能力栈：

| 阶段 | 章节 | 你实际练过的能力 |
|:---|:---|:---|
| 动机与地图 | 前言 | AI4Science 价值主张、PhysicsNeMo 定位、硬件预期 |
| 坐标 → 导数 → 残差 | ch01–ch02 | ODE/PDE PINN、损失权重、Hydra 配置实验 |
| 工业几何与反问题 | ch03 | 多 BC、Domain/Constraint、反问题直觉 |
| 框架换挡 + 算子学习 | ch04 | `physicsnemo-sym` → 主框架、FNO 训练循环 |
| 数据 + 物理 | ch05 | 混合损失、λ_physics 权衡 |
| 时空与规模 | ch06 | AFNO/自回归、短 rollout、分布式入门 |
| 闭环交付 | ch07 | 代理 → 优化 → ONNX/API；V&V 意识；UQ 概念 |

**教学设计上的刻意取舍（审阅结论）**：

1. **默认数据多为合成 / toy**：保证 CPU/小 GPU 可复现；真实 CFD 网格与全量数据集留给读后项目。  
2. **叙事场景 ≠ 默认脚本**（尤其 ch04 翼型叙事 / Darcy 默认、ch07 自定义方法论 / DrivAerNet 目录名）：降低入门摩擦，但要求读者读 `CH04_GUIDE` 与章首「读前须知」。  
3. **UQ、完整 Triton 生产、多物理耦合、非结构网格 GNN**：正文有概念或 Failure 提示，**未做成必跑作业**——这是进阶主战场。  
4. **安全关键边界已写清**：见 [SAFETY_CRITICAL_LIMITATIONS.md](../docs/SAFETY_CRITICAL_LIMITATIONS.md)；代理模型默认是「筛选 + 复核」，不是最终签审替身。

若能力矩阵里仍有 ❌，先回 [WHATS_NEXT §0](../docs/WHATS_NEXT.md) 与对应章 cheatsheet，再读本附录的进阶路径。

---

## E.1 先巩固：把本书吃透到「能改能讲」

在跳进新架构之前，建议用 **2–3 周** 做三件事（比再学一个新名词更值钱）：

| 动作 | 验收标准 | 资源 |
|:---|:---|:---|
| 对照命令再跑一轮 | 不查书完成 ch01→ch04 最小路径 | [COMMAND_REFERENCE](../docs/COMMAND_REFERENCE.md) |
| 对照 baseline | 自己的 loss / 指标落在合理区间 | [results/BASELINE.md](../results/BASELINE.md) |
| 能讲清框架切换 | 3 分钟说清 sym vs 主框架何时用 | [FRAMEWORK_SWITCH](../docs/FRAMEWORK_SWITCH.md) |
| 能讲清 ch04 口径 | 翼型叙事 vs Darcy 默认各服务什么 | [CH04_GUIDE](../ch04_fno_airfoil/CH04_GUIDE.md) |

仍卡环境：附录 B + [ENVIRONMENT.md](../docs/ENVIRONMENT.md)；仍卡 PyTorch：附录 D。

---

## E.2 按目标选路：四条进阶路径（单选聚焦）

> **原则**：同时走两条 = 两条都走不远。先选一条主路径 30–90 天。

### 路径 α · 官方生态复现（最稳）

把本书能力迁移到 NVIDIA 官方 example，再改成你的 PDE。

| 优先级 | 建议 | 与本书的衔接 |
|:---|:---|:---|
| 1 | PhysicsNeMo 文档 + `examples/` 里与 ch04/ch05 同族的 Darcy / CFD | FNO、混合损失 |
| 2 | `examples/weather/` 一类 AFNO / FCN | ch06 |
| 3 | 部署类 example（ONNX / 推理服务） | ch07 |

- 仓库：https://github.com/NVIDIA/physicsnemo  
- 文档：https://docs.nvidia.com/deeplearning/physicsnemo/

### 路径 β · 真实数据项目（最接近工业痛点）

| 领域 | 数据集 / 方向 | 从本书哪章出发 |
|:---|:---|:---|
| 翼型 / RANS | AirfRANS | ch04 叙事 |
| 汽车气动 | DrivAerNet（完整数据，非 toy） | ch07 |
| 天气 | ERA5 / ARCO-ERA5；对照 GraphCast / FourCastNet 论文 | ch06 |
| 多孔介质 | 开源 Darcy 基准与更高分辨率场 | ch05 |

**纪律**：先保留本书合成数据流水线当「调试基线」，再换真实数据；数据管线通常比调模型更耗时。

### 路径 γ · CAE / 签审与 V&V（资深工程师）

| 主题 | 去哪 |
|:---|:---|
| 签审工作流入口 | [START_HERE_CAE.md](../docs/START_HERE_CAE.md) |
| 验收报告模板 | [VV_REPORT_TEMPLATE.md](../docs/VV_REPORT_TEMPLATE.md) |
| 代理模型决策树 | [CAE_SURROGATE_DECISION_TREE.md](../docs/CAE_SURROGATE_DECISION_TREE.md) |
| 行业标准 | ASME V&V 40；[NAFEMS](https://www.nafems.org/) AI/ML 研讨 |

目标能力：能写「适用范围 / 失效模式 / 高保真复核策略」，而不是只交一张漂亮云图。

### 路径 δ · 研究向加深（论文与理论）

适合要发论文或做方法创新的读者。优先顺序建议：

1. **PINN 失败模式与修复**：NTK / 因果训练 / 自适应权重（ch02 延伸阅读已列）  
2. **神经算子统一视角**：Kovachki et al., *Neural Operator*, JMLR 2023  
3. **物理信息算子**：PINO / Physics-informed FNO（接 ch05）  
4. **UQ**：Deep Ensemble、MC-Dropout（ch07 §7.10 已点题，需自己实现）

---

## E.3 方法族扩展：本书之外还该认识谁

下列条目**按「与本书距离」由近到远**排列。先近后远，避免一上来换范式。

| 方向 | 为什么学 | 入门锚点 |
|:---|:---|:---|
| **DeepONet / Branch-Trunk** | 非均匀传感器、多分辨率输入常见 | Lu et al., *Nat. Mach. Intell.* 2021；PhysicsNeMo DeepONet 示例 |
| **Geo-FNO / 变形域算子** | 把 FNO 从规则网格推到变形网格 | Li et al. Geo-FNO 系列 |
| **GNN / MeshGraphNet 类** | 非结构网格、工业 STL/体网格主流表示 | Pfaff et al., *ICML* 2021；汽车/结构代理常见选择 |
| **Transformer / AFNO 变体** | 长程依赖、天气与多尺度场 | 接 ch06；对照 GraphCast、Pangu-Weather |
| **扩散 / 生成式物理模型** | 不确定性采样、超分辨、条件生成 | PhysicsNeMo 文档中的 diffusion 相关示例（版本以官方为准） |
| **多物理 / 共轭传热** | 真实产品几乎都是耦合场 | 从 ch03 散热片叙事扩展到 CHT / 流固热 |
| **贝叶斯优化 / 多目标** | 设计空间搜索不止 Optuna 网格 | BoTorch；接 ch07 `optimize.py` |
| **主动学习 + UQ 驱动采样** | 少标高保真点 | 接 ch07 §7.10 的「高不确定 → 进 CFD 队列」 |

> ⚠️ **选型提醒**：网格形态决定表示学习路线。规则网格优先 FNO 族；非结构网格优先 GNN / 点云类；传感器稀疏优先 DeepONet。不要用「最新论文」硬套错误离散表示。

---

## E.4 工程化下一程：从 demo 到可交付

| 层级 | 建议掌握 | 本书已有基础 |
|:---|:---|:---|
| 导出与校验 | ONNX 数值对齐、输入 schema、版本锁定 | ch07 `export_onnx.py` |
| 推理服务 | Triton model repo、动态 batch、GPU 锁 | ch07 延伸阅读 + [CAE_DEPLOYMENT_NOTES](../docs/CAE_DEPLOYMENT_NOTES.md) |
| 观测与回流 | 预测日志、漂移监控、失败样本回灌 | ch07 后记「验证闭环」 |
| 约束与可制造性 | 设计约束检查、单位/无量纲纪律 | [CAE_OPTIMIZATION_CONSTRAINTS](../docs/CAE_OPTIMIZATION_CONSTRAINTS.md) · [CAE_UNITS_AND_NONDIM](../docs/CAE_UNITS_AND_NONDIM.md) |
| 安全边界 | 何时禁止仅用代理签审 | [SAFETY_CRITICAL_LIMITATIONS](../docs/SAFETY_CRITICAL_LIMITATIONS.md) |

**最小生产清单（读后项目自检）**：

- [ ] holdout / 外推分区测试  
- [ ] 物理 sanity check（守恒、边界、量纲）  
- [ ] 不确定性或至少 ensemble 方差可见  
- [ ] Top-K 设计必须过一次高保真复核  
- [ ] API/服务有输入校验与模型版本号  

---

## E.5 精选继续阅读（论文 · 课程 · 社区）

### E.5.1 必读论文（每条路径最多先啃 3 篇）

| 主题 | 文献 |
|:---|:---|
| PINN 开山 | Raissi, Perdikaris, Karniadakis, *JCP* 2019 |
| PIML 全景 | Karniadakis et al., *Nat. Rev. Phys.* 2021 |
| FNO | Li et al., *ICLR* 2021 |
| 神经算子综述视角 | Kovachki et al., *JMLR* 2023 |
| 天气 AI | Lam et al., GraphCast, *Science* 2023；Pathak et al., FourCastNet |
| UQ 实用 | Lakshminarayanan et al., Deep Ensembles, *NeurIPS* 2017 |

各章「延伸阅读」已按主题拆开；本表是**跨章精简清单**，避免重复扫七次。

### E.5.2 课程与公开材料

| 类型 | 建议 |
|:---|:---|
| 官方 | NVIDIA PhysicsNeMo / Modulus 教程与 NGC 容器文档 |
| 深度学习补强 | PyTorch 官方 tutorials（超过附录 D 之后） |
| 数值方法补强 | 任意一本 CFD/FEM 入门（理解网格、残差、收敛——代理模型的「老师」） |
| 社区 | PhysicsNeMo GitHub Discussions；本书 Issues / PR |

### E.5.3 本书仓库内可继续挖的「第二层」

| 资源 | 适合谁 |
|:---|:---|
| [docs/START_HERE_CAE.md](../docs/START_HERE_CAE.md) 与 `CAE_*` 系列 | 要把 demo 接到 CAE 流程的人 |
| [docs/CAE_CLOSED_LOOP_DEMO.md](../docs/CAE_CLOSED_LOOP_DEMO.md) | 想看闭环故事线 |
| `tools/cfd_batch` 等工具说明 | 需要批跑 / 联合反演扩展的人 |
| 各章 `*_gpu.py` + DDP | 要从单卡迈向多卡的人 |

---

## E.6 30 / 90 天行动模板

### 30 天（单选一条路径）

```text
Week 1–2  官方 example 复现（与 ch04 或 ch07 同族）
Week 3    接入一个真实小数据集（建议 <10GB）
Week 4    ONNX 对齐 或 集成进现有 CAE/脚本；写一页 V&V 笔记
```

### 90 天（交付一个「你的问题」最小闭环）

```text
Month 1  问题定义 + 数据契约 + 合成基线可复现
Month 2  真实数据 / 混合物理 + 误差报告模板
Month 3  优化或服务化 + Top-K 高保真复核 + 失败案例分析
```

详细勾选版见 [docs/WHATS_NEXT.md](../docs/WHATS_NEXT.md)。若仍在 6 周通读阶段，先完成 [STUDY_PLAN_6WEEKS.md](../docs/STUDY_PLAN_6WEEKS.md) 再进本附录。

---

## E.7 给不同读者的一句话建议

| 你是 | 读完后优先 |
|:---|:---|
| CAE / 仿真工程师 | 路径 γ + β：真实几何 + V&V，少追新架构 |
| DL 背景研究者 | 路径 δ + α：PINO/UQ/算子统一视角，再上真实网格 |
| 学生 / 转行 | 路径 α：官方 example 复现，再做一个小真实数据项目 |
| 团队技术负责人 | 路径 γ：标准、安全边界、部署与数据回流；用 ch07 做内部分享 |

---

## E.8 本附录与其他文档的分工

| 文档 | 职责 |
|:---|:---|
| **本附录 E** | 教材内「审阅结论 + 选路 + 方法地图」 |
| [WHATS_NEXT.md](../docs/WHATS_NEXT.md) | 仓库侧勾选清单与短行动表 |
| [STUDY_PLAN_6WEEKS.md](../docs/STUDY_PLAN_6WEEKS.md) | 通读本书的周计划（读前/读中） |
| 各章「延伸阅读」 | 该章主题的论文与官方 example |

> 模型会换、SDK 会换。本书希望你带走的是：**物理 + 数据 + 模型 + 优化 + 验证 + 部署** 串成解决方案的能力。  
> 下一程的关键词只有一个：**Your Problem**。

---

*附录 E · v1.0 · 更新：2026-08-17*
