# outputs_exp7_fix 制品溯源（2026-11-24）

| 项 | 值 |
|:---|:---|
| 来源 | `ch03_heatsink/heat_sink_train.py`（仓内公开脚本，argparse 分支；未装 Hydra） |
| 配方 | **exp7 同配方**：hidden 192 × depth 6、SiLU、`hard_bottom`、RFF Fourier 64 maps / scale 6 / seed 0、lr 3e-3 + cosine、8000 步、6k 内点采样、w_bottom=30 / w_side=8 / w_robin=30 / w_base_top=12、无 pde ramp、无 weight decay |
| 唯一差异 | `robin_scale=0.03`（**正确物理口径**：`-(k/L0)∂T/∂ñ - hΔT`，`L0=0.03 m`） |
| 机器 | 8 vCPU / 15 GB / CPU-only（torch 2.15.0+cpu），约 66 min |
| 复评 | `validator.py --seed 0`（同点集）：`pde_relative_rms 2.1305` / `robin_relative_rms 0.003746`（`pass_robin_relative=true`）/ `T_fin_root|mid|tip = 89.17 / 78.40 / 68.98 ℃` / `fin_tip_1d_from_root 38.78 ℃` / `pass_physical=false` / `pass_all=false` |
| 判定 | **robin 口径修复达标**（相对特征通量 0.37%），但 pde 回退、鳍剖面过平（`T_fin_tip` 超 55 ℃ 上界、与 1D 差 30.2 ℃）→ `pass_all` 与 `pass_all_relative` 均仍 false。**残差门是必要非充分条件**，物理判据看 `pass_physical`；分析与 gate 组成见 `EXPERIMENTS_PASS.md` §10.1–§10.4 |

## 复现

```bash
cd ch03_heatsink
python3 heat_sink_train.py --output_dir outputs_exp7_fix \
  --steps 8000 --hidden 192 --depth 6 --lr 0.003 \
  --n_interior 6000 --n_bottom 600 --n_side 500 --n_robin 900 --n_base_top 300 \
  --w_pde 1 --w_bottom 30 --w_side 8 --w_robin 30 --w_base_top 12 \
  --activation silu --hard_bottom True --fourier_maps 64 --fourier_scale 6 \
  --seed 0 --robin_scale 0.03
python3 validator.py --checkpoint outputs_exp7_fix/heat_sink.pt --seed 0 --out_dir outputs_exp7_fix
```

## 注意

- `heat_sink.pt`（1.3 MB）被 `.gitignore` 的 `*.pt` 忽略，**不入库**；入库的是 `validation_report.json`、`temperature_field.png` 与本文件。
- 该 ckpt 含 Fourier 缓冲 `B`，加载须用 `heat_sink_train.HeatSinkNet`（已支持）或 `validator.load_and_validate`；`inference/net.py` 尚未同步 RFF（backlog 2），对它直接跑 `inference/export.py` 会失败。
- 对照制品见 `../outputs_exp7_legacy/PROVENANCE.md`（同配方 + `robin_scale=0`，即修复前方程）。
