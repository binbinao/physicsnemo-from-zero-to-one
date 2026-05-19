# 教材内容 · 推荐输出形式（图 / 表 / 流程）

> **背景**：主编反馈 ch04 §4.5.1「FNO block」等节在阅读器里呈现为 **深色 `text` 代码块 + 简陋 Matplotlib + Mermaid** 三重叠加，印前质量差。  
> **关联**：[BOOK_FIGURE_MEDIA_DECISIONS.md](BOOK_FIGURE_MEDIA_DECISIONS.md) · [BOOK_FIGURE_ART_QA.md](BOOK_FIGURE_ART_QA.md) · [book/STYLE.md](../book/STYLE.md)

---

## 一、决策矩阵（按内容类型）

| 内容类型 | 典型位置 | ❌ 不宜作为主呈现 | ✅ 推荐输出 | 备注 |
|:---|:---|:---|:---|:---|
| **多步算法 / FFT 管线** | ch04 §4.5.1 FNO block | `` ```text` `` 竖排 `↓`、Gemini 文生图 | **Figma / draw.io → SVG/PNG**（印前首选）；次选 **Matplotlib 结构化框图**（脚本可复现） | 须区分「空间域 / 频域」色带，标签英文 |
| **API / 框架堆栈** | F3.6、F1.8（已定 Gemini 可接受） | 纯 `text` 流程 | **Figma 矢量**（F3.6）；Gemini 插画（F1.8 已定稿） | F3.6 框图要准，不宜文生图 |
| **章路线图 / 行业 hub** | F1.1、F3.1、F5.7 等 | `text` 竖排 | **Gemini 定稿 PNG**（已定） | 正文可保留 Mermaid 作 Web 折叠，印前以 PNG 为准 |
| **教学示意（几何/BC）** | F3.3–F3.5 | `text` | **Gemini PNG**（P3 已定） | |
| **数据 / 场 / 调参** | loss、Cp、误差 | 任何插画 | **Matplotlib**（冻结） | 数值须可复现 |
| **终端 / 训练日志** | 快速通道输出 | 做成正式图 | **`` ```text` `` 或 `` ```console` ``** | 保留即可 |
| **单行 I/O 示意** | `input: … output: …` | 单独占版面的图 | **正文一句 / 表格单元格** | ch07 §7.2.2 等 |
| **对比表** | T4.1、T4.2 | 信息图替代 | **Markdown 表**；印前可 **LaTeX tabular** | 不用 Gemini |
| **Mermaid 流程图** | ch03/04/07 多处 | 与定稿 PNG **同时** 当正文主图 | **Web 辅助**；印前 **导出 SVG** 或 **删除**（见 §二） | GitHub 渲染友好，PDF 常失败 |

---

## 二、三重冗余问题（你截图的那类）

### 2.1 现状（以 ch04 §4.5.1 为例）

同一段落依次出现：

1. `` ```text` `` — 竖排「输入场 x ↓ FFT …」（阅读器里像终端截图，**不适合纸质书**）
2. `![F4.5](f4_5_fno_block.png)` — Matplotlib 五色横条（~30KB，**信息密度低**）
3. `` ```mermaid` `` — 与上两者内容重复

### 2.2 目标形态（单一事实来源）

| 渠道 | 呈现 |
|:---|:---|
| **纸质 / PDF / 静态 EPUB** | 仅 **F4.5 定稿矢量图**（图题中文，图内英文） |
| **GitHub / 在线阅读** | 定稿图 + 可选「展开 Mermaid 源码」折叠块（`details`）或链到附录 |
| **正文** | **删除** `` ```text` `` 竖排流程（或改为一行摘要，不单独占版） |

### 2.3 同类待清理节（Issue #119）

| 章节 | 节 | 定稿图 | 冗余 |
|:---|:---|:---|:---|
| ch04 | §4.5.1 | F4.5 | text + mermaid |
| ch04 | §4.1 | F4.1 | mermaid（PNG 已定） |
| ch03 | §3.1 | F3.1 | mermaid |
| ch03 | §3.6 | F3.6 | mermaid + 简陋 PNG |
| ch06 | §6.x AFNO | F6.1 | mermaid（若与 F6.1 重复） |
| ch07 | §7.6 | F7.1 | mermaid |

---

## 三、F4.5 / F3.6 矢量图规格（Issue #117 / #118）

**画布**：宽 ≥ 1400px，横向流程；白底或浅灰底（与全书 Gemini 图一致）。

**F4.5 必含节点（英文图内标签）**：

`Input field` → `Lifting` → `FFT` → `Truncate modes (k)` → `Spectral Conv` → `iFFT` → `Activation + Pointwise` → `Output`

- 频域段用 **不同背景色**（如浅蓝带）与空间域区分  
- 可选小插图：2D 频谱截断示意（modes 框）

**F3.6 必含**：`Geometry` → `Domain` → `InteriorConstraint` / `BoundaryConstraint` → `Solver` → `Validator`

**交付**：`book/assets/figma/` 或 `book/assets/svg/` 源文件 + 导出 `f4_5_fno_block.png`、`f3_6_domain_constraint.png`

---

## 四、印前管线（可选，Issue #120）

```bash
# 示例：Mermaid CLI 导出（CI 未接入）
npx @mermaid-js/mermaid-cli -i book/diagrams/f4_5.mmd -o book/assets/svg/f4_5.svg
```

- 长期：**架构图用 Figma/SVG 主稿**；Mermaid 仅维护在 `book/diagrams/*.mmd` 作 Web 同步  
- 见 [book/STYLE.md](../book/STYLE.md) 待增「图 vs 代码块」小节

---

## 五、Issue 索引

| GH | 标题 | 优先级 |
|:---|:---|:---|
| [#117](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/117) | F4.5 FNO block 矢量定稿 | ✅ done |
| [#118](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/118) | F3.6 Domain/Constraint 矢量定稿 | ✅ done |
| [#119](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/119) | 正文去 text↓/mermaid 三重冗余 | ✅ done（首批节） |
| [#120](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/120) | Mermaid→SVG 印前导出（可选） | P2 |

---

*版本：v1.0 · 2026-05-19*
