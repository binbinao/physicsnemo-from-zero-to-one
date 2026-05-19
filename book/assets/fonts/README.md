# 教材配图中文字体

| 文件 | 许可 |
|:---|:---|
| `NotoSansSC-Regular.otf` | [SIL Open Font License 1.1](https://scripts.sil.org/OFL)（Google Noto Sans SC） |

## 获取字体

若仓库未包含 OTF（体积较大可能未提交），执行：

```bash
python scripts/download_book_fonts.py
# 或手动：
curl -fsSL -o book/assets/fonts/NotoSansSC-Regular.otf \
  "https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf"
```

## 使用

```bash
python scripts/check_figure_cjk.py
python book/scripts/generate_diagram_figures.py
python book/scripts/generate_all_figures.py
```

出图脚本通过 `book/scripts/cjk_font.py` 强制加载本字体，避免方块字。
