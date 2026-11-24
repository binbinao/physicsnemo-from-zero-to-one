# outputs_exp7_legacy 制品溯源（2026-11-24 · 重训基线 / 对照）

| 项 | 值 |
|:---|:---|
| 来源 | `ch03_heatsink/heat_sink_train.py`（仓内公开脚本） |
| 用途 | **before 对照**：复现修复前（缺 `1/L0`）的方程，用来给出可比的"修复前后"数字。**不是**新的推荐制品 |
| 配方 | exp7 同配方（192×6 / SiLU / hard_bottom / RFF 64-6 / 8000 步 / 同权重同采样），**唯一差异** `robin_scale=0`（legacy 口径：`-k∂T/∂ñ - hΔT`，等效 `h_eff = h/L0 = 333 W/(m²·K)`） |
| 机器 | 8 vCPU / CPU-only，约 66 min |
| 复评 | 归档 `validation_report.json` 由 `validator.py --seed 0`（**不带** `--robin_scale`，即 ckpt 自身口径 `0`）生成：`robin_residual_rms 14.33`（旧字段）/ `robin_residual_rms_reference 1087.12` / `robin_relative_rms 1.0871` / `pde_relative_rms 0.26685` / `T_fin_root|mid|tip = 20.03 / 0.73 / 0.025 ℃`（冷鳍）/ `fin_tip_1d_from_root 8.71 ℃` |
| 与 exp7 原制品的关系 | 本 ckpt 自身口径（`robin_scale=0`）复评得 robin rms 9.86–14.33（采样相关），与 `outputs_prod/`（exp7）的 10.5 同一量级 → 说明旧制品可复现，且其"冷鳍"是方程系数问题而非训练随机性 |
| 判定 | `pass_physical=false`（鳍顶 0.025 ℃ 远低于 15 ℃ 下界）。注意其 `pass_1d_consistency` 为 **true**（8.69 ℃ 差）：冷鳍场"1D 自洽"能过，说明单项判据都不足以定物理，三条要一起看 |

## 复现

```bash
cd ch03_heatsink
python3 heat_sink_train.py --output_dir outputs_exp7_legacy \
  --steps 8000 --hidden 192 --depth 6 --lr 0.003 \
  --n_interior 6000 --n_bottom 600 --n_side 500 --n_robin 900 --n_base_top 300 \
  --w_pde 1 --w_bottom 30 --w_side 8 --w_robin 30 --w_base_top 12 \
  --activation silu --hard_bottom True --fourier_maps 64 --fourier_scale 6 \
  --seed 0 --robin_scale 0
python3 validator.py --checkpoint outputs_exp7_legacy/heat_sink.pt --seed 0 --out_dir outputs_exp7_legacy
```

## 注意

- `heat_sink.pt` 不入库（`*.pt` 被忽略）；保留本目录的目的是让"修复前"这组数字有可复算来源（R-8：exp7 原始 ckpt 未入库）。
- 因果表述：这不是"PINN 掉进冷鳍伪解"，而是"模型忠实解了一个系数写错的方程"——见 `EXPERIMENTS_PASS.md` §6.2。
