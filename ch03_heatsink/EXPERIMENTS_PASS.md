# ch03 散热片 PINN · 追 PASS 实验报告（固化存档）

> 状态：**收尾固化 + robin 量纲已修复**（第 10 节为修复后的复跑结论；`outputs_prod/` 的 exp7 制品按决定保留为历史 FAIL 存档）
> 日期：2026-09-07 起 · 修复与复跑：2026-11-24 · 实验场：`outputs/_exp/`（训练 runner `outputs/_exp/train_hp.py`，不随教材走）
> 复现只需仓内公开脚本：`heat_sink_train.py` / `validator.py` / `tests/test_robin_scale.py`（见 §9、§10.3）

## 1. 目标与质量门

`validator.py` 的 CAE 检查门槛：

| 指标 | 门槛 | 含义 |
|:---|:---|:---|
| pde 残差 RMS | < 0.1 | 内点 `∇·(k∇T)=0`（稳态无热源）近似满足 |
| robin 残差 RMS | < 0.1 | 鳍对流面 `-k∂T/∂n = h(T−T∞)` 近似满足 |
| 温度范围 | T∈[−5, 150] | 场无负温/超温杂散 |

2D 微缩模型，不做 Icepak 3D 签审级对标（见 validator note）。

## 2. 结论速览

- **朴素 SiLU-MLP（192×6）+ 硬底边**已把早期"训不动/死锁"问题解决，但残差收敛到平台后**不再随采样数、步数下降**：pde rms ≈ 26–33，robin rms ≈ 10–18，离 0.1 门槛差 **100–330×**。
- **随机 Fourier 特征（RFF，64×，scale=6）在 8000 步即把 pde rms 压到 17.8**（比 60k 步无 Fourier 的 25.9 更好且仍在下降），证实瓶颈确为 MLP 谱偏差；但 robin 项几乎不动。
- **robin 平台的根因定位**：鳍根 0~1 mm 热边界层 + `equations.py: robin_boundary` 无量纲化漏 `1/scale` 因子（对流强度被放大 ~33×），把解逼向"冷鳍"形态。
- **该假说已于 2026-11-24 修复并经有限体积参考解闭环验证**（`h=10` → 鳍顶 31.3 ℃；现码等效 `h_eff=333` → 0.01 ℃，与 exp7 实测逐点吻合）。修复后复跑结论见 **§10**；`outputs_prod/`（exp7）保留为历史 FAIL 存档。

## 3. 阶段一：死锁机制观测（exp1–exp3b）

早期 tanh 深层 + 软 BC 配方反复"训不动"，日志特征如下（均取自 `outputs/_exp/*.log` / 对应 `validation_report.json`）：

| 实验 | 现象（证据） | 诊断 |
|:---|:---|:---|
| exp1（8000 步） | ramp 到 w_pde=1 后，bottom mse 卡 3.2e3、pde mse 1.9e4、robin mse ~400 三高并存 | 底部 Dirichlet **软约束拔河**：热源上不去，PDE/边界互相拉扯 |
| exp2（12000 步） | 600 步后 loss **恒为 5.9113e+05 数百步**，全场 T≈1.48（`bottom-T mean=1.48 range=[1.48,1.48]`） | **常值场冻结**：梯度消失/局部极小，热源完全未注入 |
| exp3a（1200 步） | 硬底后 `pde≈4e-19`（**平凡满足**）、robin/base_top mse 各 ~1e5、鳍上部负温 | hard_bottom 仅解除底部死锁；**平场退化**使对流面被放弃、负温出现 |
| exp3b（800 步） | robin rms 43.8、鳍上部 −0.4℃（`T(0,25)=-0.36`） | Robin/鳍温**塌缩**，边界耦合仍未建立 |

**配方修正（对 exp4 生效）**：① `hard_bottom` 把底部 Dirichlet `T(y=0)=T_source` 精确进网络结构（`T=(1−y)T_s + y·U`）；② 激活函数 tanh→SiLU（解冻常值场死锁）；③ Robin/底部损失重加权。三处死锁同时解除。

## 4. 阶段二：exp4 配方生效（研究基线）

配置：hidden=160 / depth=6 / SiLU / hard_bottom / 5000 步 / 6k 内点采样（w_bottom=60, w_robin=30, w_base_top=18）。

| 指标 | exp4 | 门槛 |
|:---|:---|:---|
| pde rms | 52.8 | 0.1 |
| robin rms | 20.5 | 0.1 |
| 温度范围 | ✓（−0.02~99.98） | ✓ |

死锁解除、温度场形态正常，进入"能否把残差压到 0.1"的追门槛阶段。

## 5. 阶段三：追 0.1 门槛的证据链（exp5–exp7）

| 实验 | 采样/步数 | 关键变化 | pde rms | robin rms |
|:---|:---|:---|:---|:---|
| exp5 | 6k 采样 / 20k 步 | 加宽至 192×6，长训 | 33.4 | 18.5 |
| exp6 | **20k 采样** / 60k 步 | 采样×3.3、步数×3 | 25.9 | 18.3 |
| exp7 | 6k 采样 / 8k 步 | **RFF Fourier 64×/scale 6** | **17.8**（仍在下探） | 18.3 |
| 门槛 | — | — | 0.1 | 0.1 |

> 可比性注：exp4/5 权重档 w_bottom=60/w_side=12/w_base_top=18；exp6/7 档为 30/8/12。同档内（exp5→exp6）对比结论不受影响。

**结论**：
- 采样×3.3 只平滑早期下降，平台高度不变 → 平台不是采样噪声所致。
- 步数 ×3 只给 pde 少量改善（33→26），robin 几乎不动。
- **Fourier 8k 步即优于 60k 步无 Fourier**（pde 17.8 < 25.9），且训练末段 pde mse 低至 ~474–759 仍在降 → 表达瓶颈（谱偏差）确认，方向有效但本次未继续长训。

## 6. robin 平台定位：鳍根边界层 + 量纲缺陷（已修复）

### 6.1 分层诊断（exp7 场，robin 边界 4000 点）

| 位置（距鳍根） | robin rms | 鳍面温度 |
|:---|:---|:---|
| 0~1 mm | **68.9** | ~17 ℃ |
| 1~2 mm | 9.8 | ~11 ℃ |
| 2~6 mm | 4.4 | ~7 ℃ |
| 6~12 mm | 1.6 | ~2.6 ℃ |
| >12 mm / 鳍顶面 | 0.31 / 0.18 | ~0.1 ℃ |

robin 残差几乎全部来自**鳍根 0~1 mm 一条热边界层**；鳍顶面已接近门槛（0.18）。同时诊断显示模型鳍内整体 ≈0.1–4 ℃，而按几何同款 1D 鳍估算鳍顶应 ≈31–46 ℃（绝热上限 `T_source/cosh(mL) ≈ 46.5 ℃`；`fin_tip_1d_bound=61.7` 不是上限，它是 `m=√(h/(kL))`（等价 P/A=1/L）的 fin-efficiency 粗估，只作量级参考）——鳍没有真正导热。

### 6.2 量纲缺陷（已结案，修复记录见 §10）

`equations.robin_boundary` 在**归一化坐标** `ỹ=y/scale`（scale=30 mm）上求导：

- 物理法向导数 `∂T/∂n_phys = (1/scale)·∂T/∂ñ`，scale=0.03 m。
- 物理 Robin：`−k·∂T/∂n_phys = h(T−T∞)` ⇒ `−(k/scale)·∂T/∂ñ = h(T−T∞)`。
- 现代码算的是 `−k·∂T/∂ñ − h(T−T∞)`，导数项**缺 1/scale（≈1/33）** ⇒ 等效把对流强度放大约 33×。

**因果表述已修正**（早期版本的"冷鳍伪解"说法不再使用）：这不是"物理真解不可满足 → MLP 退化"，而是"**模型忠实地解了一个写错系数的方程**"。判据是与同一几何的有限体积参考解逐点对齐（纯 stdlib，1 mm 网格，`python tests/test_robin_scale.py --with-fv`，约 2 s；0.5 mm 细版见需求阶段脚本）：

| 位置 (mm) | ① 正确物理 `h=10` | ② 现码等效 `h_eff=h/L0=333` | exp7 实测（§6.1） |
|:---|:---:|:---:|:---:|
| 底座顶 (0, 9.5) | 78.2 | — | —（未采样） |
| 鳍根 (0, 10.5) | 73.7 | 20.1 | ~17 |
| 鳍中 (0, 20) | 41.6 | 0.47 | ~2.6–7 |
| 鳍顶 (0, 29.5) | **31.3** | **0.01** | ~0.1 |

> 0.5 mm 网格版本给出 76.85 / 74.94 / 42.30 / **31.38** ℃ 与 24.06 / 20.07 / 0.47 / **0.01** ℃，两套网格差 < 1.5 ℃；1D 鳍解析解（`mL=1.414`、`h/(mk)=0.141`）在鳍根 74.9 ℃ 给出鳍顶 30.55 ℃，三方交叉校验一致。

修复方式：`robin_boundary` 增加可选 `scale`（默认 `None` 保持旧行为**逐位不变**），训练/验证/反问题链传 `L0_SI = 0.03`（米，**不是** 30）；配置键 `robin_scale`（`0` = legacy 逃生阀）。回归保护 `tests/test_robin_scale.py` C1（逐位）/ C3（把 `0.03` 与 `30` 的 1000× 差异钉死）/ C6（FV 参考）。

## 7. 制品与代码状态

| 项 | 状态 |
|:---|:---|
| 历史制品（FAIL 存档） | **`outputs_prod/` = exp7**（Fourier RFF 64 / hard_bottom / SiLU / 192×6 / 8000 步；复跑报告 pde rms 16.5、robin rms 10.5、T∈[0.016, 99.98]，`pass_all=false`）。**不再作为"最佳制品"**：鳍顶 0.1 ℃ 是量纲缺陷产物 |
| 修复后制品 | **`outputs_exp7_fix/`**（同配方 + `robin_scale=0.03`）/ 同时重训的对照 **`outputs_exp7_legacy/`**（同配方 + `robin_scale=0`），见 §10 |
| `heat_sink_train.py` | 已扩展 `HeatSinkNet(hard_bottom, T_source, activation, fourier_maps, fourier_scale, seed)`，默认值保持原 tanh/无 Fourier 行为，向后兼容；本轮新增 `robin_scale` 配置键与 ckpt 字段 |
| `validator.py` | `load_and_validate` 改为按 ckpt config 重建全部网络字段（含 Fourier），旧 ckpt 走默认值不受影响；本轮新增双口径残差 + 物理探针 + `--robin_scale`/`--seed` |
| `inference/` | 默认导出链（无 Fourier ckpt）不受影响；`net.py` 复刻网络尚未同步 Fourier，fourier ckpt 导出列为后续项（backlog 2） |

## 8. 后续改进项（Backlog，按优先级）

1. **[高] ~~修复 robin_boundary 无量纲 scale 因子~~ — ✅ 已完成（2026-11-24，见 §6.2 / §10）**。`robin_boundary(..., scale=L0_SI=0.03)`，训练/验证/反问题链全部传参，回归保护 `tests/test_robin_scale.py`（C1–C6）。
2. **[中] 若 Fourier 转为默认配方**：`inference/net.py` 与 `export.py` 同步 RFF 支持，保证制品可走生产导出链（现状：`inference/test_smoke.py` 对 Fourier ckpt 报"ONNX artifact missing"，属预期缺件提示）。
3. **[低] validator 残差统计稳定性**：已补 `--seed`（显式传参才固定采样，默认行为不变）与固定物理探针（鳍根/中/顶，不受采样噪声影响）；`n_bc` 默认仍为 400，如需更稳可传更大值。
4. **[信息] ~~物理形态核对~~ — ✅ 已完成（见 §6.2 表与 `fin_tip_1d_from_root` 闭环字段）**。
5. **[中·新] SDK / GPU 档存在同类单位不自洽**：`heat_sink_train_sdk.py:169`、`heat_sink_train_gpu.py:188` 内联 Robin，坐标**未归一化**（0–60 mm 直接进网）却用 SI 的 `k`、`h` → 等效 `h_eff = h·(1 mm→m)` 缺失同一类 `1/L0` 量级问题。本轮按 D5 只出文档、不改代码（依赖 PhysicsNeMo/CUDA，本环境无法验证）。
6. **[低·新] `heat_sink_train.py` 采样未全局播种**：`seed` 只控制 RFF 频率，几何采样走全局 `np.random`。对比实验请用 `validator.py --seed 0`（同点集复评），训练间差异属采样噪声。

## 9. 复现命令

exp7 配方已可通过仓库内公开脚本 `heat_sink_train.py` 复现（`hard_bottom` / `activation` /
`fourier_*` / 采样与权重键均已进入 `conf/config.yaml` 与 `get_default_cfg()`，默认值不变）：

```bash
# 公开训练脚本（Hydra 档），等价于 exp7 配方
python ch03_heatsink/heat_sink_train.py \
  output_dir=outputs_exp7 \
  steps=8000 hidden=192 depth=6 lr=0.003 \
  n_interior=6000 n_bottom=600 n_side=500 n_robin=900 n_base_top=300 \
  w_pde=1 w_bottom=30 w_side=8 w_robin=30 w_base_top=12 \
  activation=silu hard_bottom=true \
  fourier_maps=64 fourier_scale=6 seed=0
# 验证复跑（口径同 PROVENANCE）
python ch03_heatsink/validator.py --checkpoint outputs_exp7/heat_sink.pt
```

说明：① 实验 runner 的 `--wdecay 0` 与 pde ramp 关闭即公开脚本的默认行为（Adam 无 weight
decay、cosine 退火），无需传参；② `seed` 只控制 RFF 频率抽样，采样序列不全局固定，残差存
在复跑波动（见 §7 与 backlog 3），数值不应逐位对齐；③ 原始实验 runner `outputs/_exp/train_hp.py`
不随教材走，仅作为历史实验场保留。

产物（ckpt / temperature_field）写入 `ch03_heatsink/outputs_exp7/`，训练曲线在其下 `tb/`。

## 10. 修复后结论（2026-11-24 · exp7 同配方前/后复跑）

> 本次把 backlog 1（`robin_boundary` 缺 `1/L0`）修掉并重跑，**只改量纲，不改配方**：网络、采样、权重、步数、`seed` 全部与 exp7 相同，唯一差异是 `robin_scale`（`0.03` = 正确物理 / `0` = 修复前口径）。
> 两次训练各 8000 步、约 66 min（8 vCPU CPU-only），产物在 `outputs_exp7_fix/` 与 `outputs_exp7_legacy/`。

### 10.1 前后对比（同配方、同点集 `--seed 0`）

| 指标 | **before**（`outputs_exp7_legacy/`，`robin_scale=0`） | **after**（`outputs_exp7_fix/`，`robin_scale=0.03`） | 变化 |
|:---|:---:|:---:|:---|
| `robin_relative_rms`（口径 A，新） | **1.0871** | **0.003746** | **↓290×**（AC11 要求 ≥10×；AC12 门槛 0.1 ✅） |
| `robin_residual_rms_reference`（正确口径原始值） | 1087.12 | 3.75 | ↓290× |
| `robin_residual_rms`（旧字段，按各自 ckpt 口径） | 14.33（训练时采样为 9.86） | 3.75 | 旧口径不可直接比（见 §10.2） |
| `pde_relative_rms` | 0.26685 | 2.1305 | **↑8.0×**（AC13 ✗，见 §10.3） |
| `pde_residual_rms`（旧字段） | 26.69 | 213.05 | ↑8.0×（同一件事） |
| `T_fin_root` / `T_fin_mid` / `T_fin_tip` (℃) | 20.03 / 0.73 / **0.025** | 89.17 / 78.40 / **68.98** | 冷鳍 → 鳍恢复导热 |
| `fin_tip_1d_from_root` (℃)（跨度 19 mm = 鳍高 − 2×0.5 mm 探针内缩） | 8.71 | 38.78 | — |
| `T_min` / `T_max` (℃) | 0.011 / 99.99 | 67.57 / 100.00 | — |
| `pass_robin_relative` / `pass_pde_relative` / `pass_all_relative` | false / false / false | **true** / false / false | robin 单独过关 |
| `pass_tip_band` / `pass_fin_profile` / `pass_1d_consistency` / **`pass_physical`** | false / true / true / **false** | false / true / false / **false** | 物理判据未过（见 §10.1.1） |
| `pass_all`（旧语义） | false | false | **预期仍 false**（旧字段按新口径算，见 §10.2） |

> 表中两份 JSON 都由 `validator.py --seed 0` 生成（**同点集**，仓库内 `outputs_exp7_fix/validation_report.json` 与 `outputs_exp7_legacy/validation_report.json`）；不带 `--seed` 的训练时报告数值略有采样波动（after: pde 210.6 / robin 3.34；before: robin 9.86），结论不变。

#### 10.1.1 残差门是必要条件，不是物理证书

`pass_all_relative = pass_pde_relative ∧ pass_robin_relative ∧ pass_temperature_range ∧ pass_physical`，其中 `pass_physical = pass_tip_band ∧ pass_fin_profile ∧ pass_1d_consistency`（`T_fin_tip ∈ T_inf+[0.15, 0.55]ΔT`、`root>mid>tip`、`|T_tip − fin_tip_1d_from_root| ≤ 10 ℃`）。这不是保险丝式冗余，而是实测出来的：

| 场 | `robin_relative_rms` | `T_fin_tip` | `pass_robin_relative` | `pass_physical` |
|:---|:---:|:---:|:---:|:---:|
| 600 步（exp7 配方 + `robin_scale=0.03`） | **0.0193（过 0.1 门）** | **5.14 ℃** | true | **false** |
| 2k pilot | 0.0106 | 20.84 ℃ | true | false（与 1D 差 10.6 ℃） |
| 8k `outputs_exp7_fix` | 0.003746 | 68.98 ℃ | true | false（超 55 ℃ 上界、与 1D 差 30.2 ℃） |
| 8k `outputs_exp7_legacy`（冷鳍） | 1.0871 | 0.025 ℃ | false | false |

反向的极端例子更直接：常数场 `T ≡ T_inf` 的 PDE 残差与 Robin 残差都**严格为 0**（旧 `pass_all` 会判 true），只有探针判据能识破。回归测试固化在 `tests/test_validator_gates.py`（V2）。


参照值（同几何有限体积参考解，`tests/test_robin_scale.py --with-fv`）：鳍根 73.7 / 鳍中 41.6 / **鳍顶 31.3 ℃**；1D 鳍解析（根 74.9）→ 30.55 ℃。

**修前场（`T_fin_tip = 0.025 ℃`）与 `h_eff = h/L0 = 333` 的参考解（0.01 ℃）逐点吻合**——这坐实了"模型忠实解了错系数方程"，而修后场（鳍根 89 ℃ → 鳍顶 69 ℃）已经**没有冷鳍**。

### 10.2 为什么 `robin_residual_rms`（旧字段）不能横比

旧字段的含义是"按本 ckpt 的口径算残差"。`outputs_exp7_legacy` 用 `robin_scale=0`（修复前口径，等效 `h_eff=333`），所以它的 9.86 与 old exp7 的 10.5 同量纲、可对（这也复核了旧制品可复现）；而 `outputs_exp7_fix` 用 `0.03`（物理口径，法向导数前多乘 `1/0.03`），数值天然是另一个量纲。跨 ckpt 比对必须用 `robin_residual_rms_reference`（永远按 `0.03` 重算，同点集）或 `robin_relative_rms`。

**因此 `pass_all` 在修复后仍为 false 是预期行为**，不是回退：`pass_pde/pass_robin` 仍按旧口径 `< 0.1` 判定。新叙事看 `pass_all_relative`。

### 10.3 `pde` 为什么回退（如实记录，未调门槛）

`pde_relative_rms` 0.26685 → 2.1305（8.0×）。这不是实现错误，而是**目标场变陡**：

1. 修复前那个"冷鳍"场在鳍内梯度极小（≈等温 0.1–4 ℃），`k∇²T` 天然接近 0 → pde 残差小且"好看"；exp7 的 16.5 就是这么来的。
2. 修复后鳍内必须从 89 ℃ 降到 69 ℃ 并维持向空气的横向热流，真实解在鳍根/鳍面附近有 ~1–2 mm 热边界层；冷鳍场在该薄层里梯度本来就小，`k∇²T` 天然接近 0。因此"冷鳍场的 pde 好看、正确物理场的 pde 变大"是**同一份误差在不同目标场下的口径差**，定量论证见第 4 条（`≈315` 与实测 213 同阶）。
3. 训练末段的损失配比：`pde mse ≈ 4.5e4`（w=1）vs `robin mse ≈ 15–23`（w=30，贡献 ≈5e2）→ **pde 已占损失 99%**，说明剩余 pde 不是权重失衡（R5 的扫描触发条件不成立：`loss_robin/loss_pde ≈ 3e-4 ≪ 30`），而是**表达/采样能力**：模型没有把鳍的轴向导热解开。
4. 量化证据（同一组固定探针：鳍根 (0,10.5) / 鳍中 (0,20) / 鳍顶 (0,29.5) mm）：

   | | 鳍根 10.5 mm | 鳍中 20 mm | 鳍顶 29.5 mm | tip/root |
   |:---|:---:|:---:|:---:|:---:|
   | FV 参考解（`h=10`） | 73.73 | 41.64 | **31.33** | 0.425 |
   | 修复后 8k（`outputs_exp7_fix`） | 89.17 | 78.40 | **68.98** | **0.774** |
   | 修复后 2k（pilot，仅作趋势参考） | 72.22 | 46.06 | **20.84** | 0.289 |
   | 修复前 8k（`outputs_exp7_legacy`） | 20.03 | 0.73 | **0.025** | 0.001 |

   8k 场的鳍剖面**过平**（tip/root 0.77 vs 解析 0.41）→ 轴向 `k t T''` 没被解析出来：单纯按鳍方程 `k t T'' = 2h(T−T∞)` 折算，缺失项在归一化坐标下约 `2h(T−T∞)L0²/(k t) ≈ 315`，与实测 `pde rms ≈ 210` 同量级。**结论：量纲修好后，限制项从"方程写错"变成了"鳍内薄层/轴向曲率的表达与采样"。**

   **步数不是解药（三点单调）**：把 600 步补进来看，`T_fin_tip` 是 5.14 → 20.84 → 68.98 ℃（600/2k/8k），参考解 31.3 ℃ —— 单调上升并越过参考解继续偏离；同一路线上 `robin_relative_rms` 0.0193 → 0.0106 → 0.0037、`pde_relative_rms` 9.02 → 4.40 → 2.13 单调改善。**残差单调变好、物理剖面单调变坏**，这是损失面的系统性偏置（走向"热而近等温"的鳍），不是围绕参考解的振荡：2k 落在带内只是轨迹上的瞬态，继续加步数只会更远。方向应是 §10.5 第 1 条（鳍内采样/表达），不是"再多训一会儿"。

### 10.4 与验收条款的对账（真实判定）

| AC | 判定 | 依据 |
|:---|:---:|:---|
| AC7 鳍顶 15–55 ℃ | **✗（8k）** | `T_fin_tip = 68.98 ℃`（超出上界 55）。2k pilot 为 20.84 ℃（带内）、600 步为 5.14 ℃（低于下界）；三点随步数**单调远离**参考解 31.3 ℃（§10.3），故"带内"只是轨迹瞬态 |
| AC8 `root>mid>tip` 且 mid∈[30,60]、root∈[55,95] | **✗（8k）** | 89.2 > 78.4 > 69.0 ✅ 单调，但 `T_fin_mid = 78.4` 超上界 60 |
| AC9 `\|T_fin_tip − fin_tip_1d_from_root\| ≤ 10 ℃` | **✗（8k）** | 68.98 vs 38.78 → **30.2 ℃**（2k pilot：20.84 vs 31.41 → 10.6 ℃，也 ✗）。注意修前冷鳍场该差值为 8.69 ℃（"过"），说明单项 1D 自洽不足以判物理 |
| AC10 温度范围不回退 | ✅ | `T_max = 99.99 ≤ 105`，`pass_temperature_range = true` |
| AC11 robin 改善 ≥10× 且 ≤1.0 | ✅ | 1.0871 → 0.003746（290×） |
| AC12 robin ≤ 0.1 | ✅（需限定） | 0.003746。**限定**：600 步场 `robin_relative_rms = 0.0193` 已过门而鳍顶只有 5.14 ℃ → 残差门是必要非充分条件，物理判据看 `pass_physical`（§10.1.1） |
| AC13 pde ≤ 1.3× 基线 | **✗** | 2.1305 vs 0.26685×1.3 = 0.347（原因见 §10.3，未改门槛、未做 3 组扫描，理由见 §10.3 第 3 条） |
| AC14 旧字段仍可复核 | ✅ | 两份 `validation_report.json` 都保留旧字段，本节给出与 exp7（16.5/10.5）的对照 |
| `results/BASELINE.md` 物理判据 | **✗** | `T_fin_tip = 68.98`（判据 15–55）、`T_fin_mid = 78.40`；`pass_physical = false` |

> **一句话结论**：量纲修复**达成**了它的目标——robin 残差降 290×（相对特征通量 0.37%）、鳍从 0 ℃ 冷铁片恢复成"数十 ℃ 且单调"的导热体，且旧 ckpt/旧命令零漂移。但它**没有**让 `pass_all` / `pass_all_relative` 变绿：修好后暴露出的下一个瓶颈是鳍内薄层/轴向曲率的表达能力（pde 回退 + 鳍剖面过平），属于配方而非物理口径问题，需另开一项（见 §10.5）。

### 10.5 由此产生的后续项

1. **[高] 鳍内表达/采样**：`pde_relative_rms` 2.1（目标 0.1）。方向：鳍内 importance sampling（按 `|∂T/∂n|` 或分层加密）、`n_interior` 倾斜分配到鳍（当前按面积比例，鳍只占 28.6% 面积却承载全部薄层）、加宽/加深或更多 RFF map、训练步数 20k+。**不要再动量纲与门槛。**
2. **[中] `pde` 门槛也需要口径**：`pde_residual_rms` 在归一化坐标下与"目标场的陡度"强耦合（同一份误差，冷鳍场 26.7、正确物理场 210.6），跨口径不可比。建议与 robin 同样改用"相对特征通量"并**在文档里绑定 `ΔT` 与 `L0`**（本轮已给出公式，但门槛数值未动，仍待用户决定）。
3. **[中] SDK/GPU 档同类问题**（§8 backlog 5）：内联 Robin + 未归一化 mm 坐标 + SI `k/h`。
4. **[低] 制品策略（D3 待确认）**：本轮按"并列不替换"处理——`outputs_exp7_fix/` 与 `outputs_exp7_legacy/` 新增，`outputs_prod/`（exp7）保留为历史 FAIL 存档；若要把它降级/替换需用户决定。

### 10.6 复跑命令

```bash
cd ch03_heatsink

# 回归测试（含 1 mm FV 参考，约 2 s）
python3 tests/test_robin_scale.py --with-fv

# after（正确物理口径）
python3 heat_sink_train.py --output_dir outputs_exp7_fix \
  --steps 8000 --hidden 192 --depth 6 --lr 0.003 \
  --n_interior 6000 --n_bottom 600 --n_side 500 --n_robin 900 --n_base_top 300 \
  --w_pde 1 --w_bottom 30 --w_side 8 --w_robin 30 --w_base_top 12 \
  --activation silu --hard_bottom True --fourier_maps 64 --fourier_scale 6 \
  --seed 0 --robin_scale 0.03

# before（复现修复前方程）
python3 heat_sink_train.py --output_dir outputs_exp7_legacy  ...（同上）... --robin_scale 0

# 同点集复评（--seed 0 固定采样）。归档的两份 JSON 就是这两条命令的产物
python3 validator.py --checkpoint outputs_exp7_fix/heat_sink.pt    --seed 0 --out_dir outputs_exp7_fix
python3 validator.py --checkpoint outputs_exp7_legacy/heat_sink.pt --seed 0 --out_dir outputs_exp7_legacy

# 可选：把 before 的旧字段 robin_residual_rms 按"正确口径"重算（会把该字段从
# 14.33 改写成 1087.12，与归档文件不同；跨 ckpt 对比请直接用
# robin_residual_rms_reference / robin_relative_rms，不必跑这条）
python3 validator.py --checkpoint outputs_exp7_legacy/heat_sink.pt --robin_scale 0.03 --seed 0 --out_dir /tmp/rep_before
```

> 未装 Hydra 时走 argparse 分支；`--hard_bottom True` 才是开（`type=bool` 对 `"False"` 也返回 True）。

### 10.7（可选）2k 步 pilot

上表用到的 2k 数据由同一条命令 + `--steps 2000 --output_dir /tmp/pilot_fix` 得到（约 13 min，scratch run 未入库）：

```bash
python3 heat_sink_train.py --output_dir /tmp/pilot_fix \
  --steps 2000 --hidden 192 --depth 6 --lr 0.003 \
  --n_interior 6000 --n_bottom 600 --n_side 500 --n_robin 900 --n_base_top 300 \
  --w_pde 1 --w_bottom 30 --w_side 8 --w_robin 30 --w_base_top 12 \
  --activation silu --hard_bottom True --fourier_maps 64 --fourier_scale 6 \
  --seed 0 --robin_scale 0.03
```

结果：`T_fin_root|mid|tip = 72.22 / 46.06 / 20.84 ℃`、`robin_relative_rms = 0.0106`、`pde_relative_rms = 4.40`、`fin_tip_1d_from_root = 31.41 ℃`（按探针跨度 19 mm 重算；旧稿按 20 mm 名义鳍高算得 29.46，偏低 1.95 ℃）。与 1D 参考差 10.6 ℃ → **AC9 也不通过**，且该点未入库（scratch），只作"步数不是解药"的趋势证据（§10.3）。

### 10.8 验证节点复核（2026-11-24 第二轮）：把物理判据接进门

本轮验证复核发现 §10.1 的 `pass_all_relative` 只由残差 + 温度范围组成，**不含物理探针判据**——实测 600 步场 `robin_relative_rms = 0.0193`（过 0.1 门）而鳍顶 5.14 ℃，即残差门可被非物理解满足。已修：

| 变更 | 内容 | 影响 |
|:---|:---|:---|
| 物理判据进门槛 | 新增 `pass_tip_band` / `pass_fin_profile` / `pass_1d_consistency` → `pass_physical`，并入 `pass_all_relative` | 现有两份制品的 `pass_all_relative` 仍为 false（原本 pde 也未过），无数值回退 |
| 探针随几何 | 探针由 `geo.base_height/fin_height/中间鳍轴线` 推导（`fin_probe_points`），`fin_width` 取 `geo.fin_width`；输出 `fin_probe_points_mm` / `fin_probe_span_mm` | 默认几何点位不变（10.5/20/29.5）；`--fin_height 40` 不再静默读错点 |
| 1D 参考用探针跨度 | `fin_tip_1d_from_root` 传入 `fin_height − 2×0.5 mm`（19 mm），不再用名义鳍高 | 8k：36.37 → **38.78 ℃**；before：8.17 → **8.71 ℃**（修掉 ~2 ℃ 系统性偏低） |
| `pde` 口径写进文档与 `convention` | `pde_relative_rms` 的分母残差是**归一化坐标**的 `k·∂²T/∂ñ²`；SI 等价量 = `r̃·kΔT/L0²`（差 `1/L0²=1111×`） | 只改措辞与 `convention` 字符串，数值未动 |
| `L0_SI` 由坐标归一化常量派生 | `equations.COORD_SCALE_MM = 30.0`，`L0_SI = COORD_SCALE_MM/1000`；三个脚本的 `scale = 30.0` 改为 `COORD_SCALE_MM` | 值仍为 `0.03`（逐位相等），改归一化时不再静默错 |
| 回归测试 | 新增 `tests/test_validator_gates.py`（V1–V5：探针随几何、常数场被物理门拒、键序契约、1D 跨度、`L0` 不变式） | `python3 tests/test_validator_gates.py`，< 1 s |

旧 ckpt 零漂移仍成立：无 `robin_scale` 键的旧格式 ckpt 在新 validator 下 `pde_residual_rms / robin_residual_rms / T_min / T_max / fin_tip_1d_bound / pass_all` 与基线 commit 逐位相同（本节点实测，见验证报告）。
