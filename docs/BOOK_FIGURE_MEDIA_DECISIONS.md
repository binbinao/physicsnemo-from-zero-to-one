# 全书配图与表格 · Gemini 采用决策（已确认）

> **主编确认**：2026-05-19 · **方式 1（整类拍板）**  
> - **22 张** 已 Gemini 试稿 → **定稿**（合并 PR #114 后入库）  
> - **26 张** Matplotlib 数据/场图 → **维持不动**  
> - **12 张** 概念/流程图 → **下一批 Gemini**（见 Issue P3）  
> - **3 张** F4.5 / F3.6 /（可选 F1.8）→ **Figma 矢量**，不纳入 Gemini 批次  
> - **全部表格** → **Markdown**，不用 Gemini  

> 维护说明：默认建议基于插图编辑二审 + 路线 A/B 试稿经验。  
> **图内文字策略**：Gemini 图内尽量 **英文短标签**；**中文图题/表题** 留在 Markdown 正文。  
> **表格**：原则上 **不用 Gemini**（数字与中文须精确）；仅「概念关系图」可例外改为信息图。

**图例（建议列）**

| 标记 | 含义 |
|:---|:---|
| **Gemini** | 建议用 Gemini / 同类文生图（或 Cursor 图像生成，提示词见 [BOOK_FIGURE_GEMINI_PROMPTS.md](BOOK_FIGURE_GEMINI_PROMPTS.md)） |
| **Matplotlib** | 保持 `generate_all_figures.py` / `generate_diagram_figures.py` |
| **Figma** | 建议矢量工具（框图、FFT 链路比文生图更准） |
| **Markdown** | 保持正文表格，不转图片 |
| **已 Gemini** | 已定稿（22 张，PR #114） |
| **P3 Gemini** | 下一批 12 张（待 Issue 执行） |

---

## 一、配图（54 张 PNG + 1 张章外）

### 前言 ch00（3 图）

| 图号 | 文件 | 类型 | 当前 | **建议** | 理由 | 确认 |
|:---|:---|:---|:---|:---|:---|:---:|
| F0.1 | `f0_1_ai4s_timeline.png` | 时间线 | 已 Gemini | **已 Gemini** | 叙事信息图，非数据 | ☐ |
| F0.2 | `f0_2_cfd_vs_ai_comparison.png` | 对数柱图 | Matplotlib | **Matplotlib** | 数值对比，须可复现 | ☐ |
| F0.3 | `f0_4_book_roadmap.png` | 全书路线 | 已 Gemini | **已 Gemini** | 地铁线图适合插画 | ☐ |

### 第 1 章 ch01（8 图 + 表见第二节）

| 图号 | 文件 | 类型 | 当前 | **建议** | 理由 | 确认 |
|:---|:---|:---|:---|:---|:---|:---:|
| F1.0 | `f1_0_banner.png` | 章横幅 | 已 Gemini | **已 Gemini** | 章首第一印象 | ☐ |
| F1.1 | `f1_1_roadmap.png` | 章路线 | 已 Gemini | **已 Gemini** | 流程图，非数据 | ☐ |
| F1.2 | `f1_4_mlp_results.png` | 训练结果 | Matplotlib | **Matplotlib** | 真实 loss/曲线 | ☐ |
| F1.3 | `f1_5_pinn_results.png` | 快速通道 | Matplotlib | **Matplotlib** | 同上 | ☐ |
| F1.4 | `f1_3_spring_diagram.png` | 受力示意 | 已 Gemini | **已 Gemini** | 教学示意 | ☐ |
| F1.5 | `f1_4_mlp_results.png` | MLP 三联 | Matplotlib | **Matplotlib** | 与 F1.2 同数据 | ☐ |
| F1.6 | `f1_5_pinn_results.png` | PINN 三联 | Matplotlib | **Matplotlib** | 同上 | ☐ |
| F1.7 | `f1_5_2_extrapolation.png` | 外推对比 | Matplotlib | **Matplotlib** | 定量对比 | ☐ |
| F1.8 | `f1_7_physicsnemo_arch.png` | 双框架栈 | Matplotlib | **Gemini** 或 **Figma** | 现图为色块+字，偏素；架构图用矢量更稳 | ☐ |

### 第 2 章 ch02（8 图）

| 图号 | 文件 | 类型 | 当前 | **建议** | 理由 | 确认 |
|:---|:---|:---|:---|:---|:---|:---:|
| F2.1 | `f2_1_roadmap.png` | 章路线 | 已 Gemini | **已 Gemini** | | ☐ |
| F2.2 | `f2_2_loss_curves.png` | loss 曲线 | Matplotlib | **Matplotlib** | | ☐ |
| F2.3 | `f2_3_temperature_evolution.png` | 场演化 | Matplotlib | **Matplotlib** | 场数据 | ☐ |
| F2.4 | `f2_4_physics_intuition.png` | 物理直觉 | Matplotlib | **Gemini**（可选） | 现为简单曲线叠加，可升级为示意插画 | ☐ |
| F2.5 | `f2_5_collocation_points.png` | 散点分布 | Matplotlib | **Matplotlib** | 配点坐标须准确 | ☐ |
| F2.6 | `f2_7_depth_sweep.png` | 调参 | Matplotlib | **Matplotlib** | | ☐ |
| F2.7 | `f2_7_3_lr_sweep.png` | 调参 | Matplotlib | **Matplotlib** | | ☐ |
| F2.8 | `f2_7_4_weight_sweep.png` | 调参 | Matplotlib | **Matplotlib** | | ☐ |

### 第 3 章 ch03（10 图）

| 图号 | 文件 | 类型 | 当前 | **建议** | 理由 | 确认 |
|:---|:---|:---|:---|:---|:---|:---:|
| F3.0 | `f3_0_banner.png` | 章横幅 | 已 Gemini | **已 Gemini** | | ☐ |
| F3.1 | `f3_1_progression.png` | 维度递进 | Matplotlib | **Gemini** | 方框流程过素 | ☐ |
| F3.2 | `f3_2_temperature_field.png` | 温度场 | Matplotlib | **Matplotlib** | 仿真场 | ☐ |
| F3.3 | `f3_3_geometry.png` | 几何 | 已 Gemini | **已 Gemini** | | ☐ |
| F3.4 | `f3_4_csg.png` | CSG | 已 Gemini | **已 Gemini** | | ☐ |
| F3.5 | `f3_5_boundary_conditions.png` | BC 示意 | Matplotlib | **Gemini** 或 **Figma** | 多边界标注图 | ☐ |
| F3.6 | `f3_6_domain_constraint.png` | Sym 结构 | Matplotlib | **Figma**（首选） | API 框图宜矢量 | ☐ |
| F3.7 | `f3_7_loss_curves.png` | loss | Matplotlib | **Matplotlib** | | ☐ |
| F3.8 | `f3_8_inverse_flow.png` | 反问题流 | Matplotlib | **Gemini** | 与 f3_9 同系列 | ☐ |
| F3.9 | `f3_9_design_loop.png` | 设计闭环 | 已 Gemini | **已 Gemini** | | ☐ |

### 第 4 章 ch04（10 图）

| 图号 | 文件 | 类型 | 当前 | **建议** | 理由 | 确认 |
|:---|:---|:---|:---|:---|:---|:---:|
| F4.0 | `f4_0_banner.png` | 章横幅 | 已 Gemini | **已 Gemini** | | ☐ |
| F4.1 | `f4_1_framework_switch.png` | 框架切换 | 已 Gemini | **已 Gemini** | | ☐ |
| F4.2 | `f4_2_fno_prediction.png` | 预测场 | Matplotlib | **Matplotlib** | | ☐ |
| F4.3 | `f4_3_pinn_vs_fno.png` | 对比 | 已 Gemini | **已 Gemini** | | ☐ |
| F4.4 | `f4_4_neural_operator.png` | 算子映射 | Matplotlib | **Gemini** 或 **Figma** | 概念图 | ☐ |
| F4.5 | `f4_5_fno_block.png` | FNO 结构 | Matplotlib | **Figma**（首选） | FFT 链路需准确，文生图易错 | ☐ |
| F4.6 | `f4_6_data_pipeline.png` | 数据管线 | Matplotlib | **Gemini** | 流程图 | ☐ |
| F4.7 | `f4_7_pressure_field.png` | 压力场 | Matplotlib | **Matplotlib** | | ☐ |
| F4.8 | `f4_8_cp_curve.png` | Cp 曲线 | Matplotlib | **Matplotlib** | | ☐ |
| F4.9 | `f4_9_tuning.png` | 调参三联 | Matplotlib | **Matplotlib** | | ☐ |

### 第 5 章 ch05（9 图）

| 图号 | 文件 | 类型 | 当前 | **建议** | 理由 | 确认 |
|:---|:---|:---|:---|:---|:---|:---:|
| F5.0 | `f5_0_banner.png` | 章横幅 | 已 Gemini | **已 Gemini** | | ☐ |
| F5.1 | `f5_1_triangle.png` | 三角关系 | Matplotlib | **Gemini** 或 **Figma** | PINN/FNO/数据三角 | ☐ |
| F5.2 | `f5_2_comparison.png` | 场对比 | Matplotlib | **Matplotlib** | | ☐ |
| F5.3 | `f5_3_darcy_physics.png` | 物理示意 | Matplotlib | **Gemini**（可选） | 可插画化渗流 | ☐ |
| F5.4 | `f5_4_darcy_samples.png` | 样本对 | Matplotlib | **Matplotlib** | k→u 场 | ☐ |
| F5.5 | `f5_6_small_data.png` | 小数据曲线 | Matplotlib | **Matplotlib** | | ☐ |
| F5.6 | `f5_7_lambda_sweep.png` | λ 扫描 | Matplotlib | **Matplotlib** | | ☐ |
| F5.7 | `f5_8_industry_map.png` | 行业映射 | 已 Gemini | **已 Gemini** | | ☐ |

### 第 6 章 ch06（6 图）

| 图号 | 文件 | 类型 | 当前 | **建议** | 理由 | 确认 |
|:---|:---|:---|:---|:---|:---|:---:|
| F6.0 | `f6_0_banner.png` | 章横幅 | 已 Gemini | **已 Gemini** | | ☐ |
| F6.1 | `f6_1_afno_block.png` | AFNO 块 | Matplotlib | **Gemini** 或 **Figma** | 模块结构图 | ☐ |
| F6.2 | `f6_2_rollout_result.png` | rollout 场 | Matplotlib | **Matplotlib** | | ☐ |
| F6.3 | `f6_3_autoregressive.png` | 自回归 | Matplotlib | **Gemini** | 时间步示意 | ☐ |
| F6.4 | `f6_4_error_growth.png` | 误差曲线 | Matplotlib | **Matplotlib** | | ☐ |
| F6.5 | `f6_5_weather_industry.png` | 行业 | 已 Gemini | **已 Gemini** | | ☐ |

### 第 7 章 ch07（2 图）

| 图号 | 文件 | 类型 | 当前 | **建议** | 理由 | 确认 |
|:---|:---|:---|:---|:---|:---|:---:|
| F7.0 | `f7_0_banner.png` | 章横幅 | 已 Gemini | **已 Gemini** | | ☐ |
| F7.1 | `f7_1_pipeline.png` | 端到端 | Matplotlib | **Gemini** | 六步 pipeline 过素 | ☐ |

### 章外 / 非正文图号

| 编号 | 文件 | **建议** | 理由 | 确认 |
|:---|:---|:---|:---|:---:|
| — | `../assets/wechat_qrcode.png` | **不用 Gemini** | 须真实公众号二维码，占位待换 | ☐ |

---

## 二、配图建议汇总（便于拍板）

| 建议 | 数量 | 说明 |
|:---|:---:|:---|
| **已 Gemini（维持定稿）** | **22** | P0+P1+部分 P2 已试稿 |
| **新增 Gemini（建议下一批）** | **12** | f2_4, f3_1, f3_5, f3_8, f4_4, f4_6, f5_1, f5_3?, f6_1, f6_3, f7_1, f1_8 |
| **Figma 矢量（优先于 Gemini）** | **3** | f4_5, f3_6, 可选 f1_8 |
| **Matplotlib（不变）** | **26** | 全部数据/场/调参图 |
| **章外** | **1** | 微信二维码 |

---

## 三、表格（正文 + 附录）— 均不建议 Gemini

> 表格含 **精确数字、中文、公式**，应保留 **Markdown 表格** 或将来用 LaTeX/Word 排版。  
> 若要做「信息图」，仅 **无数字的概念关系表** 可考虑，本书暂无刚需。

### 3.1 带编号的正式表（T 表）

| 表号 | 位置 | 标题/用途 | **建议** | 确认 |
|:---|:---|:---|:---|:---:|
| T2.1 | ch02 §2.1 | 第 1→2 章升级清单 | **Markdown** | ☐ |
| T4.0 | ch04 章首 | 两条路径 A/B | **Markdown** | ☐ |
| T4.1 | ch04 | sym vs main 框架 | **Markdown**（与 F4.1 图分工） | ☐ |
| T4.2 | ch04 | PINN vs FNO | **Markdown**（与 F4.3 图分工） | ☐ |
| T1.1 | ch01 | MLP vs PINN 决策 | **Markdown** | ☐ |
| T1.2 | ch01 | 三行业对应 | **Markdown** | ☐ |
| T1.3 | ch01 | 工时对比 | **Markdown** | ☐ |

### 3.2 正文内嵌表（无 T 编号，按章统计）

| 章节 | 约数量 | 典型用途 | **建议** | 确认 |
|:---|:---:|:---|:---|:---:|
| ch00 | 6 | 对比框架、硬件、能力 | **Markdown** | ☐ |
| ch01 | 3 | 读法路径、行业、工时 | **Markdown**（T1.x 已列） | ☐ |
| ch02 | 8+ | Icepak 工时、调参结果、Failure | **Markdown** | ☐ |
| ch03 | 6+ | 几何尺寸、BC、反问题、行业 | **Markdown** | ☐ |
| ch04 | 6+ | 路径、PINN/FNO、调参、CFD 工时 | **Markdown** | ☐ |
| ch05 | 5+ | PINO 对比、小数据、λ 扫描 | **Markdown** | ☐ |
| ch06 | 3+ | FNO/AFNO、变量量级、DDP 坑 | **Markdown** | ☐ |
| ch07 | 3+ | 能力回收、数据档位、验收项 | **Markdown** | ☐ |
| 附录 A | 4 | 符号、损失对照、推荐阅读 | **Markdown** | ☐ |
| 附录 B | 2 | 云 GPU 场景、厂商对比 | **Markdown** | ☐ |

**表格合计**：正式 T 表 **7** 张；内嵌 Markdown 表约 **40+** 处 — **全部建议保持 Markdown，不用 Gemini。**

---

## 四、Mermaid 流程图（正文内，非 PNG）

| 位置 | **建议** | 理由 | 确认 |
|:---|:---|:---|:---:|
| ch01 §1.1 路线图 | **保留 Mermaid** 或导出 PNG 与 F1.1 二选一 | 避免双份重复；定稿可删 mermaid 只留图 | ☐ |
| ch02/ch03/ch04/ch05/ch06/ch07 部分节 | 同上 | 与对应 Gemini 路线图统一即可 | ☐ |

---

## 五、执行状态

| 动作 | 状态 |
|:---|:---|
| 合并 PR #114（22 张 Gemini 定稿） | ✅ 已合并 |
| Issue #115 P3：12 张 Gemini 重绘 | ✅ 试稿已入库 |
| Figma/SVG：F4.5、F3.6 | #117、#118（见 [OUTPUT_FORMS](BOOK_FIGURE_OUTPUT_FORMS.md)） |
| 正文去 text↓/mermaid 三叠 | #119 |
| Mermaid→SVG 印前（可选） | #120 |
| Matplotlib 26 张 | 冻结，仅 CJK/数据更新 |

---

## 六、编辑立场（供参考）

- **Gemini 明显优于 Matplotlib 的**：章横幅、全书/章路线图、行业 hub、闭环流程、受力/几何教学示意。  
- **不应 Gemini 的**：凡带 **坐标轴数值、loss 曲线、误差场、Cp、调参扫描** 的图。  
- **表格永远 Markdown**；图题中文、图内英文，是本书最稳的印前策略。

---

*文档版本：v1.0-confirmed · 2026-05-19*
