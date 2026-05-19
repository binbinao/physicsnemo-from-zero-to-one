# 框架切换：physicsnemo-sym → physicsnemo 主框架

> **单一入口**：全书只在这一页 + [ch04 CH04_GUIDE](../ch04_fno_airfoil/CH04_GUIDE.md) 展开细节。

## 何时切换

| 章节 | 框架 | 典型 import |
|:---|:---|:---|
| ch01–ch03 | **physicsnemo.sym**（符号 PDE / PINN） | `physicsnemo.sym.*` |
| ch04–ch07 | **physicsnemo 主框架**（算子、AFNO、部署） | `physicsnemo.*` |

切换点在第 3 章末 → 第 4 章初（见 [ch00 §0.4](../book/ch00.md)）。

## 对读者的影响

1. **ch01–ch03** 可只装 `requirements-minimal.txt` 跑裸 PyTorch；SDK 版再装 sym。
2. **ch04+** 建议 GPU；默认训练任务见 CH04_GUIDE（目录名 `airfoil`，默认脚本为 **Darcy**）。
3. `check_env --chapter 4` 会检查主框架相关依赖与 `ch04` 数据目录。

## 命令速查

```bash
python scripts/check_env.py --chapter 3   # sym 相关提示
python scripts/check_env.py --chapter 4   # 主框架 + ch04 目录
```

## 延伸阅读

- [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md) ch04 节
- [book/ch04.md §4.1](../book/ch04.md) 正文叙述
