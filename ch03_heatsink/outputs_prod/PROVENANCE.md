# outputs_prod 制品溯源（2026-09-07 更新）

| 项 | 值 |
|:---|:---|
| 来源实验 | `outputs/_exp/out_exp7/`（train_hp.py，tag=exp7，8000 步） |
| 网络 | HeatSinkNet：hidden 192 × depth 6，SiLU，**hard_bottom**，**RFF Fourier 64 maps / scale 6.0 / seed 0** |
| 训练 | lr 3e-3、6k 内点采样、w_bottom=30/w_robin=30/w_base_top=12、无 pde ramp |
| scale | 30.0（mm→归一化输入） |
| 复跑验证 | pde rms 16.5 / robin rms 10.5 / T∈[0.016, 99.98] · `pass_all=false`（门槛 pde/robin <0.1，仍差 ~100×） |

详细实验证据与结论见 [`ch03_heatsink/EXPERIMENTS_PASS.md`](../EXPERIMENTS_PASS.md)。

## 注意

- 该 ckpt 含 Fourier 缓冲 `B`；加载须用 `heat_sink_train.HeatSinkNet`（已支持）或按 ckpt config 重建。
- `inference/net.py` 复刻网络**尚未**同步 Fourier 支持 → 直接对该 ckpt 跑 `inference/export.py` 会失败；属后续改进项（默认无 Fourier 训练流程不受影响）。
