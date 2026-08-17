# 书—代码命令对照表

> 正文以本表为准；发现不一致请提 Issue。

## Hydra vs argparse（必读）

部分脚本（`heat1d_train.py`、`heat_sink_train.py`、`train_fno_mini.py`、
`train_data_fno.py` / `train_physics_fno.py`、`train_afno_mini.py`）在检测到
`hydra-core` 时走 **Hydra**，否则走 **argparse**：

| 模式 | 写法示例 | 何时生效 |
|:---|:---|:---|
| **argparse** | `python train_fno_mini.py --epochs 50` | 未安装 `hydra-core` |
| **Hydra** | `python train_fno_mini.py epochs=50` | 已安装 `hydra-core`（如 `requirements-full.txt`） |

装了 Hydra 后，`--epochs` / `--steps` **会被忽略或报错**。入口脚本启动时会打印一行提示。
纯 argparse 脚本（如 `heat1d_pinn_raw.py`、`pinn_spring.py`、`train.py`）始终用 `--参数`。

## ch01（argparse）

```bash
cd ch01_hello
python mlp_spring.py --epochs 1000
python pinn_spring.py --epochs 5000
```

## ch02（Hydra，config 在 `conf/`）

```bash
cd ch02_heat1d
# 首选（始终 argparse）：
python heat1d_pinn_raw.py --steps 500
# Hydra 入口：
python heat1d_train.py
python heat1d_train.py arch=large training=full
python heat1d_train.py w_ic=1000
python heat1d_train.py -m arch=small,large lr=1e-3,1e-4
```

无 Hydra 时：`python heat1d_train.py --hidden 64 --steps 3000`（见脚本 `--help`）。

**注意**：配置项为 `w_ic` / `n_pde`，不是正文旧写法 `loss_weights.lam_ic` / `sampling.n_interior`。

## ch03

> `heat_sink_train.py` 的 checkpoint 始终写在 `ch03_heatsink/outputs/`（与当前工作目录无关）。

```bash
cd ch03_heatsink
python heat_sink_train.py          # 结束写 outputs/validation_report.json
# 无 Hydra：--steps；有 Hydra：steps=500
python heat_sink_train.py --steps 500   # CPU 演示可缩短步数
python validator.py --checkpoint outputs/heat_sink.pt
python heat_sink_inverse_joint.py --target_temp 40 --steps 2000
python visualize.py
```

## ch04（FNO：默认 Darcy，翼型为扩展路径）

> 目录名 `fno_airfoil` = 工业场景；**默认训练 = Darcy**。见 [ch04_fno_airfoil/CH04_GUIDE.md](../ch04_fno_airfoil/CH04_GUIDE.md)。

**路径 A（默认）**

```bash
cd ch04_fno_airfoil
# 无 Hydra：
python train_fno_mini.py --epochs 50
# 有 Hydra：
python train_fno_mini.py epochs=50
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
# 无 Hydra 用 --epochs；有 Hydra 用 epochs=50 n_train=100
python train_data_fno.py epochs=50 n_train=100
python train_physics_fno.py epochs=50 n_train=100 lambda_physics=0.1
```

## ch06

```bash
cd ch06_fourcastnet_mini
python scripts/generate_toy_weather.py --n_time 200 --resolution 64
# 无 Hydra：--epochs；有 Hydra：epochs=30
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
python optimize_multi.py --checkpoint outputs/best.pt --n_trials 100
python ../../tools/cfd_batch/import_hifi_queue.py --csv outputs/hifi_queue.csv
python export_onnx.py --checkpoint outputs/best.pt
```

## CAE 闭环（根目录）

```bash
python scripts/run_cae_closed_loop_demo.py
python scripts/run_cae_closed_loop_demo.py --skip-train --skip-ch03
```

见 [CAE_CLOSED_LOOP_DEMO.md](CAE_CLOSED_LOOP_DEMO.md)。

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
