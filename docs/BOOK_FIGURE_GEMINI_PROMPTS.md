# 教材配图 · Gemini 路线 B 提示词手册

> **用途**：装饰性/叙事性配图（章横幅、路线图、行业映射、教学示意）。  
> **不适用**：loss 曲线、场图、调参扫描等数据图（继续 `generate_all_figures.py`）。  
> **关联**：[BOOK_FIGURE_ART_QA.md](BOOK_FIGURE_ART_QA.md) · [FIGURE_MANIFEST.md](../book/assets/FIGURE_MANIFEST.md)

## 全局约束（每条提示词末尾追加）

```
Style: industrial technical textbook illustration, clean flat vector with subtle depth, blue (#2E86AB) and orange (#E07A2F) accent palette, white or very light gray background, 16:9 or 3:1 wide banner aspect ratio, high resolution, no watermark, no photorealistic faces, no sci-fi neon. Minimal English labels only (2-4 words max per label); do NOT render Chinese characters in the image.

AVOID: cherry blossoms, flowers, pastel floral frames, decorative nature borders, romantic or artistic poster style. Prefer engineering diagrams, CFD/FEA aesthetic, dark-line technical drawing.
```

## 交付路径

| 优先级 | 输出目录 | 合并到正文 |
|:---|:---|:---|
| 试稿 | `book/assets/gemini/` | 人工验收后 `cp` 覆盖 `book/assets/{name}.png` |
| 定稿 | `book/assets/` | 更新 `FIGURE_MANIFEST` 的 `source` 列 |

## P0 — 章横幅（7 张）

| 文件 | 图号 | 提示词主体（English prompt body） |
|:---|:---|:---|
| `f1_0_banner.png` | F1.0 | Horizontal chapter opener banner: a stylized mass-spring-damper on the left, equation fragment "m x'' + kx = 0" subtly in background, on the right a small dual-panel sketch comparing data points (MLP) vs smooth physics curve (PINN). Industrial engineering textbook cover feel. |
| `f2_0_banner.png` | — | *(ch02 无独立 banner 文件；可选新增或跳过)* |
| `f3_0_banner.png` | F3.0 | Banner: 3D-cutaway heatsink on a red-hot chip package, heat flow arrows rising through aluminum fins, subtle grid hinting 2D simulation. Semiconductor thermal theme. |
| `f4_0_banner.png` | F4.0 | Banner: airfoil profile in wind tunnel flow, pressure field color wash (blue low, red high), faint Fourier spectrum motif along top edge. CFD surrogate / FNO theme. |
| `f5_0_banner.png` | F5.0 | Banner: porous media cross-section with flow through channels, overlay of neural network nodes merging with Darcy flow arrows. "Data + physics" visual metaphor. |
| `f6_0_banner.png` | F6.0 | Banner: Earth globe with weather layers, temperature field stripes, small AFNO block diagram inset. Medium-range forecast AI theme. |
| `f7_0_banner.png` | F7.0 | Banner: car silhouette with streamlines, drag coefficient gauge, small API/cloud icon at the end of a design loop arrow. Automotive aerodynamic optimization theme. |

## P0 — 全书路线图（1 张）

| 文件 | 图号 | 提示词主体 |
|:---|:---|:---|
| `f0_4_book_roadmap.png` | F0.3 | Infographic metro-line roadmap left-to-right: 5 stations labeled in English only — "Ch0 Intro", "Ch1-3 PINN", "Ch4-5 FNO", "Ch6 Weather", "Ch7 Deploy". Each station has a tiny icon (book, flame, wing, globe, car). Sym branch tint blue, main framework tint orange. Clear arrows between stations, no overlapping text. |

## P1 — 行业 / 闭环 / 架构对比（5 张）

| 文件 | 图号 | 提示词主体 |
|:---|:---|:---|
| `f3_9_design_loop.png` | F3.9 | Circular workflow diagram: CAD → Icepak → PINN surrogate → optimizer → validation, icons for each step, chip cooling context. English labels only. |
| `f4_1_framework_switch.png` | F4.1 | Layered architecture diagram: lower layer "physicsnemo-sym" (PINN, PDE), upper layer "physicsnemo" (FNO, train), bridge arrow "Ch4 switch". Clean stack, not 3D. |
| `f4_3_pinn_vs_fno.png` | F4.3 | Split comparison: left single airfoil one CFD run (PINN), right grid of many airfoil silhouettes with fast inference rays (FNO). Labels "single case" vs "many cases". |
| `f5_8_industry_map.png` | F5.7 | Hub-and-spoke: center "Darcy flow", spokes to oil & gas, battery electrode, ceramics, geology icons. Professional icon style, not bar chart. |
| `f6_5_weather_industry.png` | F6.5 | Hub "Weather AI" connected to energy, shipping, insurance, agriculture icons around a subtle global temperature map background. |

## P2 — 教学示意（可选，6 张）

| 文件 | 图号 | 提示词主体 |
|:---|:---|:---|
| `f1_3_spring_diagram.png` | F1.4 | Free-body diagram: block mass m, horizontal spring to wall, force arrow F=-kx, coordinate x. Clean physics textbook line art. |
| `f3_3_geometry.png` | F3.3 | Labeled three-fin heat sink side view: base plate, three fins with spacing arrows, chip heat source below. English labels "base", "fin", "chip". |
| `f3_4_csg.png` | F3.4 | Three panels: rectangle A, rectangle B, union A+B for CSG teaching, soft colors. |
| `f1_1_roadmap.png` | F1.1 | Chapter 1 flow: Hook → Fast track → PINN → SDK → Industry, horizontal chevron steps, English short labels. |
| `f2_1_roadmap.png` | F2.1 | Chapter 2 flow: Hook → PDE → 3 losses → Hydra → Tuning, same visual language as f1_1. |
| `f0_1_ai4s_timeline.png` | F0.1 | Timeline 2020-2026 with icons: protein, graph, modulus logo style blocks, weather, car; events AlphaFold, PIML, Modulus, GraphCast, PhysicsNeMo, v2.0. English event names only. |

## P3 — 概念/流程图（12 张，Issue #115）

| 文件 | 图号 | 提示词主体 |
|:---|:---|:---|
| `f2_4_physics_intuition.png` | F2.4 | Three overlaid Gaussian heat pulses on a rod (narrow to wide), arrows showing diffusion over time t=0, t=0.1, t=0.3. Label "heat diffusion". No axis numbers. |
| `f3_1_progression.png` | F3.1 | Three-step progression left to right: 1D rod sketch → 2D finned heatsink → 3D chip package cutaway. Labels "1D", "2D", "3D". |
| `f3_5_boundary_conditions.png` | F3.5 | Heatsink cross-section with color-coded edges: red fixed T, blue heat flux, green convection h. English BC labels only. |
| `f3_8_inverse_flow.png` | F3.8 | Flow: measured temperature field → invert conductivity k → design update. Icons for sensor, neural net, CAD. |
| `f4_4_neural_operator.png` | F4.4 | Left: one input shape; center: neural operator block; right: fan of many output fields. Label "one model, many cases". |
| `f4_6_data_pipeline.png` | F4.6 | Linear pipeline: raw CFD meshes → preprocessing → tensor dataset → FNO training. CFD/ML engineering icons. |
| `f5_1_triangle.png` | F5.1 | Equilateral triangle with vertices PINN, FNO, Data; center label PINO. Clean infographic, not 3D. |
| `f5_3_darcy_physics.png` | F5.3 | Porous media slice with flow arrows; permeability field a(x) as shaded band; Darcy law hint. |
| `f6_1_afno_block.png` | F6.1 | Block diagram: input weather field → AFNO spectral blocks → output field; small FFT motif. |
| `f6_3_autoregressive.png` | F6.3 | Time sequence t, t+Δ, t+2Δ with weather maps feeding back into model arrow loop. |
| `f7_1_pipeline.png` | F7.1 | Six chevrons: Data → Train → HPO → Optimize → ONNX → API. Automotive Cd context subtle. |
| `f1_7_physicsnemo_arch.png` | F1.8 | Two-layer stack: bottom physicsnemo-sym (PINN/PDE), top physicsnemo (FNO/train). Bridge arrow between layers. |

## 负面提示词（Negative prompt，Gemini / Imagen 可用时附加）

```
blurry text, Chinese characters, kanji, watermark, logo clutter, anime, cyberpunk neon, misleading physics, wrong airfoil shape, offensive content, low resolution, jpeg artifacts
```

## 验收清单（美术编辑签字）

- [ ] 图内无方块字、无乱码英文
- [ ] 与 Matplotlib 数据图并置时色温不突兀
- [ ] 物理示意无原则性错误（翼型、弹簧方向、热流向上等）
- [ ] 宽图 ≥ 1200 px；横幅约 3:1
- [ ] `LICENSE_NOTES.md` 已含 AI 配图声明

## 自动化试稿（Cursor 图像生成 / Gemini）

本仓库 P0 试稿使用 **Cursor 内置图像生成**（提示词按上表），输出至 `book/assets/gemini/`。在 [Google AI Studio](https://aistudio.google.com/) 用同一提示词可换 **Gemini 2.0 Flash 图像** 重出并对比。

仓库内试稿可放在 `book/assets/gemini/`。验收通过后：

```bash
cp book/assets/gemini/f1_0_banner.png book/assets/f1_0_banner.png
```

正文图题保持中文，不依赖图内汉字。
