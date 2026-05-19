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
