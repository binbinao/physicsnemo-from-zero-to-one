# 配图生成脚本

| 脚本 | 输出 | 说明 |
|:---|:---|:---|
| `generate_all_figures.py` | 26 张数据可视化图 | loss、场图、FNO 预测、调参曲线等 |
| `generate_diagram_figures.py` | 路线图 / 横幅 / 结构图 | 时间线、pipeline、几何示意等 |

输出目录：`book/assets/`（相对仓库根目录，勿使用绝对路径）。

```bash
python book/scripts/generate_diagram_figures.py
python book/scripts/generate_all_figures.py
```
