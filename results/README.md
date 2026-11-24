# 可复现基线（参考）

本目录用于存放各章训练的参考输出（loss 曲线、checkpoint 说明等），**不提交大文件**（见根目录 `.gitignore`）。

## 当前状态

- **硬件与训练时长**：见 [docs/HARDWARE_EXPECTATIONS.md](../docs/HARDWARE_EXPECTATIONS.md)（含分章 wall-clock、显存、定性 loss 趋势）。
- **逐章数值 baseline**：[BASELINE.md](BASELINE.md)（定性 loss 趋势与脚本）；欢迎 PR 在子目录附上实测数值。

## 建议本地记录格式

```
results/
  ch02_heat1d/
    YYYY-MM-DD_rtx4070/
      config.yaml
      final_losses.txt
      notes.md
```

## 提交真实记录的要求（`example_record/` 或 `YYYY-MM-DD_设备/`）

想给仓库补一份参考记录，请**真的跑一遍**，并在 `notes.md` 里写清：

| 项 | 要求 | 例 |
|:---|:---|:---|
| 命令 | 完整可复制的命令行（含 `--epochs` / `epochs=` 语法） | `python train_fno_mini.py --epochs 30` |
| 硬件 | CPU 型号或 GPU 型号 + 显存 | `RTX 4070 12GB` |
| 种子 | 随机种子（脚本未固定时写「未固定」） | `seed=0` |
| 日期 | 跑这次训练的日期 | `2026-08-20` |
| commit | 当时的仓库 commit | `git rev-parse --short HEAD` |

**禁止**：示意值（`illustrative`）、区间值（`≈1e-3–5e-3`）、未实际运行的估算值。这类内容会被当作伪造证据，PR 不予合并。

判读方法见 [BASELINE.md](BASELINE.md)（只给定性趋势，不承诺具体数字）。

