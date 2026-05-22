# 配图生产说明（维护者用）

正文仅保留 `![F{章}.{序} …](assets/…)` 图题；详细分镜与生成提示集中于此，便于重绘与版本管理。

| 图号 | 文件 | 说明摘要 | source |
|:---|:---|:---|:---|
| F0.1 | f0_1_ai4s_timeline.png | 2020–2026 AI4S 关键事件横向时间线 | **gemini** |
| F0.2 | f0_2_cfd_vs_ai_comparison.png | 传统 CFD vs AI 代理：网格/求解/扫参工时对比（对数柱图） | matplotlib |
| F0.3 | f0_3_book_roadmap.png | 全书 7 章地铁线路图（sym/main + 行业） | **gemini**（试稿见 `gemini/`） |
| F1.0 | f1_0_banner.png | 弹簧 + ODE + PINN 章横幅 | **gemini** |
| F1.1 | f1_1_roadmap.png | 第 1 章学习路线（与 mermaid 一致） | **gemini** |
| F1.2 | f1_2_mlp_results.png | 快速通道 MLP 终端/结果 | matplotlib |
| F1.3 | f1_3_pinn_results.png | 快速通道 PINN 与真值对照 | matplotlib |
| F1.4 | f1_4_spring_diagram.png | 弹簧受力示意图 | **gemini** |
| F1.5 | f1_2_mlp_results.png | MLP loss + 外推失败三联图（同 F1.2） | matplotlib |
| F1.6 | f1_3_pinn_results.png | PINN 三件套 loss + 拟合（同 F1.3） | matplotlib |
| F1.7 | f1_7_extrapolation.png | MLP vs PINN 外推对比 | matplotlib |
| F1.8 | f1_8_physicsnemo_arch.png | PhysicsNeMo 双框架堆栈 | **gemini** |
| F2.1 | f2_1_roadmap.png | 第 2 章学习路线 | **gemini** |
| F2.2–F2.3 | f2_2, f2_3 | loss、场演化 | matplotlib |
| F2.4 | f2_4_physics_intuition.png | 热扩散直觉 | **gemini** |
| F2.5–F2.8 | f2_5, f2_6*, f2_7*, f2_8* | 配点、调参 | matplotlib |
| F3.0 | f3_0_banner.png | 散热片章横幅 | **gemini** |
| F3.1 | f3_1_progression.png | 维度递进 | **gemini** |
| F3.3 | f3_3_geometry.png | three-fin 几何 | **gemini** |
| F3.4 | f3_4_csg.png | CSG 并集示意 | **gemini** |
| F3.5 | f3_5_boundary_conditions.png | 边界条件示意 | **gemini** |
| F3.6 | f3_6_domain_constraint.png | Sym 结构 | **matplotlib-vector**（#118） |
| F3.7 | f3_7_loss_curves.png | loss | matplotlib |
| F3.8 | f3_8_inverse_flow.png | 反问题流 | **gemini** |
| F3.9 | f3_9_design_loop.png | 散热设计闭环 | **gemini** |
| F4.0 | f4_0_banner.png | 翼型/FNO 章横幅 | **gemini** |
| F4.1 | f4_1_framework_switch.png | sym/main 框架切换 | **gemini** |
| F4.3 | f4_3_pinn_vs_fno.png | 单工况 vs 多工况 | **gemini** |
| F4.4 | f4_4_neural_operator.png | 算子映射 | **gemini** |
| F4.5 | f4_5_fno_block.png | FNO 结构 | **matplotlib-vector**（#117；频域色带 + 英文标签） |
| F4.6 | f4_6_data_pipeline.png | 数据管线 | **gemini** |
| F4.2, F4.7–F4.9 | f4_2, f4_7–9 | 预测场、Cp、调参 | matplotlib |
| F5.0 | f5_0_banner.png | Darcy 章横幅 | **gemini** |
| F5.1 | f5_1_triangle.png | PINN/FNO/数据三角 | **gemini** |
| F5.3 | f5_3_darcy_physics.png | Darcy 渗流示意 | **gemini** |
| F5.7 | f5_8_industry_map.png | 多孔介质行业映射 | **gemini** |
| F5.2, F5.4–F5.6 | f5_2, f5_4–6 | 场对比、小数据、λ | matplotlib |
| F6.0 | f6_0_banner.png | FourCastNet 章横幅 | **gemini** |
| F6.1 | f6_1_afno_block.png | AFNO block | **gemini** |
| F6.3 | f6_3_autoregressive.png | 自回归 rollout | **gemini** |
| F6.5 | f6_5_weather_industry.png | 天气 AI 行业 | **gemini** |
| F6.2, F6.4 | f6_2, f6_4 | rollout 场、误差曲线 | matplotlib |
| F7.0 | f7_0_banner.png | 汽车气动章横幅 | **gemini** |
| F7.1 | f7_1_pipeline.png | 端到端 pipeline | **gemini** |

## 主编决策（2026-05-19）

| 档位 | 数量 | 策略 |
|:---|:---:|:---|
| **定稿 gemini** | **34** | P0–P2（PR #114）+ P3（PR 待开） |
| **冻结 matplotlib** | **~20** | loss/场/调参/预测对比 |
| **Figma（待绘）** | 2 | F4.5, F3.6 |
| **表格** | 全部 | Markdown，不转图 |

**路线 B 提示词**：[docs/BOOK_FIGURE_GEMINI_PROMPTS.md](../../docs/BOOK_FIGURE_GEMINI_PROMPTS.md)  
**决策全文**：[docs/BOOK_FIGURE_MEDIA_DECISIONS.md](../../docs/BOOK_FIGURE_MEDIA_DECISIONS.md)  
**输出形式**：[docs/BOOK_FIGURE_OUTPUT_FORMS.md](../../docs/BOOK_FIGURE_OUTPUT_FORMS.md)（架构图勿用 `` ```text` `` 竖排）

批量重绘脚本与历史记录见 [docs/FIGURE_UPGRADE_SUMMARY.md](../../docs/FIGURE_UPGRADE_SUMMARY.md)。
