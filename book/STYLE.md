# 教材体例说明（`book/`）

全书 Markdown 遵循本文件。修订时请先改体例，再改单章。

## 章标题

```markdown
# 第 N 章 · 中文标题
```

前言使用 `# 前言：…`，不称「第 0 章」；全书口径为 **前言 + 第 1–7 章 + 附录 A–D**。

## 章首元数据（ch01–ch07 必填）

```markdown
> **预计阅读**：正文 XX 分钟｜跑通代码 XX 分钟｜深入 XX 分钟
> **配套代码**：[`chNN_xxx/`](https://github.com/binbinao/physicsnemo-from-zero-to-one/tree/main/chNN_xxx)
> **难度**：⭐…
> **本章关键词**：`…`
> **环境基线**：见 [ENVIRONMENT.md](../docs/ENVIRONMENT.md) · PhysicsNeMo v2.0 · PyTorch ≥ 2.3
```

可选：`> **📌 读前须知**`（叙事与默认代码不一致时，如 ch04、ch07）。

## 节标题

| 类型 | 格式 |
|:---|:---|
| 钩子 | `## N.0 钩子：…` |
| 快速通道 | `## N.x 🟢 快速通道：…` |
| 深入 | `## N.x 🔵 …` |
| 行业映射 | `## N.x 🏭 行业映射：…` |
| Failure | `## N.x 🔵 Failure Case：…` |
| 章末 | `## N.x ➡️ 下章预告 + 章末 CTA` |

## 图与表

- **图号**：`F{章号}.{序}`，子图 `F2.7.1` 仅当同一节多图且需细分子序。
- **表号**：`T{章号}.{序}`，标题 `### T1.1 …`。
- **图题**：`![F1.0 简短说明](assets/f1_0_banner.png)`；生产用描述写在 [assets/FIGURE_MANIFEST.md](assets/FIGURE_MANIFEST.md)，不进正文。
- **Mermaid**：源文件可保留；读者版以 `assets/*.png` 为准（与图题并存时，PNG 优先）。

## 章末块（顺序固定）

1. `## N.x ➡️ 下章预告 + 章末 CTA`（终章为后记 + 附录指引）
2. 章末 CTA 引用块（📘 💬 🔔）
3. 视频脚本说明（链到 [video_scripts/README.md](video_scripts/README.md)）
4. `### 延伸阅读`
5. 页脚：`*本章字数：约 X,XXX 字 · 图表数：N 张 · 版本：v1.0 · 更新：YYYY-MM-DD*`

## 附录

- 标题：`# 附录 X · 标题`
- 节：`## X.1 …`，节次连续
- 页脚：`*附录 X · v1.0 · 更新：YYYY-MM-DD*`

## 标点与引号

- 中文正文优先使用「」标注强调与引用。
- 英文专名、代码、路径保持原文。
