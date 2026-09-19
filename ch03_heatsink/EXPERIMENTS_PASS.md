# ch03 散热片 PINN · 追 PASS 实验报告（固化存档）

> 状态：**收尾固化**（质量门槛保持 FAIL，结论与后续项已建档）
> 日期：2026-09-07 · 实验场：`outputs/_exp/`（训练 runner `outputs/_exp/train_hp.py`，不随教材走）

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
- **robin 平台的根因定位**：鳍根 0~1 mm 热边界层 + `equations.py: robin_boundary` 无量纲化疑似漏 `1/scale` 因子（对流强度被放大 ~33×），把解逼向"冷鳍伪解"。该伪解在物理真解上不可满足，故 robin 无法再降。
- 按用户决定：**量纲修复不改代码**（涉及教材方程核心文件），列为高优先后续项。

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

## 6. robin 平台定位：鳍根边界层 + 量纲假说

### 6.1 分层诊断（exp7 场，robin 边界 4000 点）

| 位置（距鳍根） | robin rms | 鳍面温度 |
|:---|:---|:---|
| 0~1 mm | **68.9** | ~17 ℃ |
| 1~2 mm | 9.8 | ~11 ℃ |
| 2~6 mm | 4.4 | ~7 ℃ |
| 6~12 mm | 1.6 | ~2.6 ℃ |
| >12 mm / 鳍顶面 | 0.31 / 0.18 | ~0.1 ℃ |

robin 残差几乎全部来自**鳍根 0~1 mm 一条热边界层**；鳍顶面已接近门槛（0.18）。同时诊断显示模型呈**"冷鳍伪解"**：鳍内整体 ≈0.1–4 ℃，而按几何同款 1D 鳍估算鳍顶应 ≈46 ℃（`fin_tip_1d_bound=61.7` 为绝热上限）——鳍没有真正导热。

### 6.2 量纲假说（后续修复项，未改代码）

`equations.robin_boundary` 在**归一化坐标** `ỹ=y/scale`（scale=30 mm）上求导：

- 物理法向导数 `∂T/∂n_phys = (1/scale)·∂T/∂ñ`，scale=0.03 m。
- 物理 Robin：`−k·∂T/∂n_phys = h(T−T∞)` ⇒ `−(k/scale)·∂T/∂ñ = h(T−T∞)`。
- 现代码算的是 `−k·∂T/∂ñ − h(T−T∞)`，导数项**缺 1/scale（≈1/33）** ⇒ 等效把对流强度放大约 33×。

后果：物理真解要求鳍的横/法向薄层梯度比代码表述小 33×，而代码表述下真解不可满足；MLP 的最小代价解是"把鳍整体冻到接近 T∞"，robin 残差由鳍根必须连高温底座的那一小段钉住 → 平台 ~10–18（该数值对采样敏感：同 exp7 制品复跑 rms 为 18.3 与 10.5 两值，400 点验证方差大）。

修复路线（已验证性很强）：`robin_boundary` 增加可选 `scale` 参数、默认保持现行为，训练/验证链传物理 scale，8k 步即可验证鳍是否恢复导热与 robin 是否真正下降。

## 7. 制品与代码状态

| 项 | 状态 |
|:---|:---|
| 最佳制品 | **`outputs_prod/` = exp7**（Fourier RFF 64 / hard_bottom / SiLU / 192×6 / 8000 步；复跑报告 pde rms 16.5、robin rms 10.5、T∈[0.016, 99.98]，`pass_all=false`） |
| `heat_sink_train.py` | 已扩展 `HeatSinkNet(hard_bottom, T_source, activation, fourier_maps, fourier_scale, seed)`，默认值保持原 tanh/无 Fourier 行为，向后兼容 |
| `validator.py` | `load_and_validate` 改为按 ckpt config 重建全部网络字段（含 Fourier），旧 ckpt 走默认值不受影响 |
| `inference/` | 默认导出链（无 Fourier ckpt）不受影响；`net.py` 复刻网络尚未同步 Fourier，fourier ckpt 导出列为后续项 |

## 8. 后续改进项（Backlog，按优先级）

1. **[高] 修复 robin_boundary 无量纲 scale 因子**（`equations.py` + 训练/验证调用点，8k 步验证"鳍恢复导热 + robin 可降"）。预期同时修正"鳍顶 0.1℃ vs 应 ≈46℃"的物理形态错误。
2. **[中] 若 Fourier 转为默认配方**：`inference/net.py` 与 `export.py` 同步 RFF 支持，保证制品可走生产导出链。
3. **[低] validator 残差统计稳定性**：robin/pde 采样固定 RNG 或增大 `n_bc`（复跑波动 18.3→10.5，门槛判定需稳定口径）。
4. **[信息] 物理形态核对**：把模型鳍顶温度与 1D 鳍解析/有限差分解对照（当前模型鳍顶 ≈0.1℃ 提示冷鳍伪解）。

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
