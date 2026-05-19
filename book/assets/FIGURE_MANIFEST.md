# 配图生产说明（维护者用）

正文仅保留 `![F{章}.{序} …](assets/…)` 图题；详细分镜与生成提示集中于此，便于重绘与版本管理。

| 图号 | 文件 | 说明摘要 | source |
|:---|:---|:---|:---|
| F0.1 | f0_1_ai4s_timeline.png | 2020–2026 AI4S 关键事件横向时间线 | **gemini** |
| F0.2 | f0_2_cfd_vs_ai_comparison.png | 传统 CFD vs AI 代理：网格/求解/扫参工时对比（对数柱图） | matplotlib |
| F0.3 | f0_4_book_roadmap.png | 全书 7 章地铁线路图（sym/main + 行业） | **gemini**（试稿见 `gemini/`） |
| F1.0 | f1_0_banner.png | 弹簧 + ODE + PINN 章横幅 | **gemini** |
| F1.1 | f1_1_roadmap.png | 第 1 章学习路线（与 mermaid 一致） | **gemini** |
| F1.2 | f1_4_mlp_results.png | 快速通道 MLP 终端/结果 | matplotlib |
| F1.3 | f1_5_pinn_results.png | 快速通道 PINN 与真值对照 | matplotlib |
| F1.4 | f1_3_spring_diagram.png | 弹簧受力示意图 | **gemini** |
| F1.5 | f1_4_mlp_results.png | MLP loss + 外推失败三联图 | matplotlib |
| F1.6 | f1_5_pinn_results.png | PINN 三件套 loss + 拟合 | matplotlib |
| F1.7 | f1_5_2_extrapolation.png | MLP vs PINN 外推对比 | matplotlib |
| F1.8 | f1_7_physicsnemo_arch.png | PhysicsNeMo 双框架堆栈 | matplotlib |
| F2.1 | f2_1_roadmap.png | 第 2 章学习路线 | **gemini** |
| F2.2–F2.8 | f2_*（除 f2_1） | loss、场演化、配点、调参 | matplotlib |
| F3.0 | f3_0_banner.png | 散热片章横幅 | **gemini** |
| F3.1 | f3_1_progression.png | 维度递进 | matplotlib |
| F3.3 | f3_3_geometry.png | three-fin 几何 | **gemini** |
| F3.4 | f3_4_csg.png | CSG 并集示意 | **gemini** |
| F3.x | f3_*（其余） | BC、训练、反问题等 | matplotlib |
| F3.9 | f3_9_design_loop.png | 散热设计闭环 | **gemini** |
| F4.0 | f4_0_banner.png | 翼型/FNO 章横幅 | **gemini** |
| F4.1 | f4_1_framework_switch.png | sym/main 框架切换 | **gemini** |
| F4.3 | f4_3_pinn_vs_fno.png | 单工况 vs 多工况 | **gemini** |
| F4.x | f4_*（其余） | 预测场、Cp、调参 | matplotlib |
| F5.0 | f5_0_banner.png | Darcy 章横幅 | **gemini** |
| F5.7 | f5_8_industry_map.png | 多孔介质行业映射 | **gemini** |
| F5.x | f5_*（其余） | 混合 FNO、λ 扫描等 | matplotlib |
| F6.0 | f6_0_banner.png | FourCastNet 章横幅 | **gemini** |
| F6.5 | f6_5_weather_industry.png | 天气 AI 行业 | **gemini** |
| F6.x | f6_*（其余） | AFNO、rollout、误差 | matplotlib |
| F7.0 | f7_0_banner.png | 汽车气动章横幅 | **gemini** |
| F7.1 | f7_1_pipeline.png | 端到端 pipeline | matplotlib |

**路线 B 提示词**：[docs/BOOK_FIGURE_GEMINI_PROMPTS.md](../../docs/BOOK_FIGURE_GEMINI_PROMPTS.md)

批量重绘脚本与历史记录见 [docs/FIGURE_UPGRADE_SUMMARY.md](../../docs/FIGURE_UPGRADE_SUMMARY.md)。
