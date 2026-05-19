# 书—代码命令对照表

> 正文以本表为准；发现不一致请提 Issue。  
> **Hydra 章**：在章节目录下执行；无 Hydra 时用 `--参数` 形式。

## ch01（argparse）

```bash
cd ch01_hello
python mlp_spring.py --epochs 1000
python pinn_spring.py --epochs 5000
```

## ch02（Hydra，config 在 `conf/`）

```bash
cd ch02_heat1d
python heat1d_train.py
python heat1d_train.py arch=large training=full
python heat1d_train.py w_ic=1000
python heat1d_train.py -m arch=small,large lr=1e-3,1e-4
```

无 Hydra 时：`python heat1d_train.py --hidden 64 --steps 3000`（见脚本 `--help`）。

**注意**：配置项为 `w_ic` / `n_pde`，不是正文旧写法 `loss_weights.lam_ic` / `sampling.n_interior`。

## ch03

```bash
cd ch03_heatsink
python heat_sink_train.py          # 结束写 outputs/validation_report.json
python validator.py --checkpoint outputs/heat_sink.pt
python visualize.py
```

## ch04（FNO：默认 Darcy，翼型为扩展路径）

> 目录名 `fno_airfoil` = 工业场景；**默认训练 = Darcy**。见 [ch04_fno_airfoil/CH04_GUIDE.md](../ch04_fno_airfoil/CH04_GUIDE.md)。

**路径 A（默认）**

```bash
cd ch04_fno_airfoil
python train_fno_mini.py --epochs 50
# Hydra: python train_fno_mini.py epochs=50
```

**路径 B（翼型合成，可选）**

```bash
python dataset.py --type airfoil --n_samples 100
# 注意：fno_darcy.pt 是路径 A 的 Darcy 权重；翼型可视化仅看几何/流场样式，勿与 Darcy 精度对比
python visualize_airfoil.py --ckpt outputs/fno_darcy.pt
```

## ch05（依赖 ch04）

```bash
cd ch05_darcy_hybrid
python train_data_fno.py epochs=50 n_train=100
python train_physics_fno.py epochs=50 n_train=100 lambda_physics=0.1
```

## ch06

```bash
cd ch06_fourcastnet_mini
python scripts/generate_toy_weather.py --n_time 200 --resolution 64
python train_afno_mini.py epochs=30
python rollout_eval.py --ckpt outputs/afno_weather.pt --rollout_steps 10
```

## ch07（argparse）

```bash
cd ch07_drivaernet_optim
python data/generate_toy_cars.py   # 若 data/ 为空
python train.py --epochs 200
python optimize.py --checkpoint outputs/best.pt --n_trials 100
python hifi_validation_queue.py --checkpoint outputs/best.pt --top_k 5
python export_onnx.py --checkpoint outputs/best.pt
```

## 已知的正文历史写法（勿用）

| 正文旧命令 | 仓库正确命令 |
|:---|:---|
| `train_fno_mini.py dataset=darcy_mini` | `train_fno_mini.py` 或 `epochs=50` |
| `train_data_fno.py data.train_size=100` | `n_train=100` |
| `train_physics_fno.py loss.lambda_pde=0.1` | `lambda_physics=0.1` |
| `train_afno_mini.py data=toy_weather training=debug` | 先 `generate_toy_weather.py`，再 `train_afno_mini.py epochs=30` |
| `rollout_eval.py outputs/fcn_mini/best.pt` | `--ckpt outputs/afno_weather.pt` |
| `train.py model=mlp data=toy_car` | `train.py --epochs 200` |
| `optimize.py checkpoint=outputs/...` | `--checkpoint outputs/best.pt` |
