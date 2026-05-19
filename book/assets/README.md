# 教材配图 (`book/assets/`)

本目录存放各章 Markdown 引用的插图（PNG）。在 GitHub 上阅读 `book/ch*.md` 时可直接显示。

## 重新生成

```bash
# 数据类图表（loss 曲线、场图、调参扫描等，26 张）
python book/scripts/generate_all_figures.py

# 路线图 / 横幅 / 结构示意图（其余配图）
python book/scripts/generate_diagram_figures.py
```

**依赖**：`matplotlib`、`numpy`；可选 `scienceplots`（未安装时自动回退 `ggplot` 样式）。

## 两个 assets 目录

| 路径 | 用途 |
|:---|:---|
| **`book/assets/`** | 教材插图（`ch*.md` 里 `![](assets/...)`） |
| **`assets/`（仓库根）** | 非正文插图，如章末公众号二维码占位 [`wechat_qrcode.png`](../../assets/wechat_qrcode.png) |

章末「扫码关注」图为**占位**，可替换为正式二维码后覆盖根目录文件。

## 说明

- 配图为教学示意，部分为合成数据可视化，非论文级实验复现图。
- 图号与生产分镜见 [FIGURE_MANIFEST.md](FIGURE_MANIFEST.md)；体例见 [../STYLE.md](../STYLE.md)。
