# CAE 单位与无量纲化

## 本书默认

| 章 | 典型约定 | 换回 SI |
|:---|:---|:---|
| ch02 1D 热 | $x,t,u$ 无量纲或归一化到 $[0,1]$ | 用特征长度 $L$、时间 $L^2/\alpha$ |
| ch03 散热片 | 几何 **mm**（只在采样层出现），$k$、$h$、$T$ 为 SI 一致单位 | 导数项用 $L_0=0.03\ \mathrm{m}$ 归到 SI，见下节「ch03 单位规则」 |
| ch04–ch05 Darcy | 渗透率/压力场多为 **归一化** 合成数据 | 对接客户 CFD 时需重标定 |
| ch07 Cd | 无量纲 $C_d$ | 与风洞/CFD 一致即可 |

## 签审建议

1. 训练前写清 **参考量**：$L_{ref}, T_{ref}, U_{ref}, \rho, \mu$。  
2. 代理模型 I/O 与 **求解器导出场** 用同一套归一化（存于 checkpoint `meta`）。  
3. 报告同时给 **无量纲误差** 与 **SI 误差**（如最高温度 ℃）。

## ch03 单位规则（本章唯一事实源）

本章的坑不是"忘写单位"，而是**两套坐标混用**：采样层用 mm，网络输入被除以 `scale = 30` 归一化，而方程的 `k`、`h` 是 SI。三者必须在进入方程前对齐。

| 量 | 值 | 出现位置 | 备注 |
|:---|:---|:---|:---|
| 坐标归一化常数 | `scale = 30.0` | `heat_sink_train.py` / `validator.py`，`x̃ = x_mm / 30` | 无量纲，只为网络输入 |
| 物理特征长度 | `L0_SI = 0.03`（米） | `equations.py`，`robin_boundary(scale=...)` | **同一段 30 mm，换成米**；由 `COORD_SCALE_MM/1000` 派生 —— 随坐标归一化变，**不随鳍高变** |
| 边界条件配置 | `robin_scale: 0.03` | `conf/config.yaml` / ckpt | `0` = legacy（修复前行为） |

**推导**：$\tilde n = n_{\mathrm{SI}}/L_0 \Rightarrow \partial T/\partial n_{\mathrm{SI}} = (1/L_0)\,\partial T/\partial\tilde n$，因此

$$r_{robin} = -\frac{k}{L_0}\,\frac{\partial T}{\partial\tilde n} - h\,(T-T_\infty),\qquad L_0 = 0.03\ \mathrm{m}$$

漏掉 $1/L_0$ 等价于把 $h$ 放大 $1/L_0 \approx 33$ 倍（$h_{\text{eff}} = 333\ \mathrm{W/(m^2\cdot K)}$），鳍被解成 ≈0 ℃ 的冷铁片。判据：$\varepsilon = hL_0/k = 0.3$；本章有限体积参考解在正确口径下给出鳍顶 31.3 ℃。

**可执行规则**（不要用"单位自洽"这种空话）：

1. 几何 mm **只允许**出现在采样层（`heat_sink_geometry.py`）与可视化里；进入网络一律 `/scale`，进入方程一律 `/1000`（或 `/L0` 换回无量纲）。
2. 任何对归一化坐标求导的项，乘 `1/L0`（`robin_boundary(scale=L0_SI)`）。
3. 残差门槛必须写清除数：本章 `GATE = 0.1` 指 $\tilde r = r/(h\Delta T)$（robin）、$r/(k\Delta T)$（pde），$0.1 \equiv 100\ \mathrm{W/m^2}$（robin，$\Delta T=100$ K）。**两项的 $r$ 不在同一坐标系**：robin 的 $r$ 用 $L_0$ 换回 SI（$\mathrm{W/m^2}$），pde 的 $r$ 是**对归一化坐标**的拉普拉斯 $k\,\partial^2T/\partial\tilde n^2$（SI 等价量 $=\tilde r\,k\Delta T/L_0^2$，差 $1/L_0^2=1111$ 倍）。读 `pde_relative_rms` 时按前者理解。
4. 新脚本复制本章公式时先跑 `python ch03_heatsink/tests/test_robin_scale.py`：C1 保证旧口径逐位不变，C3 把 `0.03` 与 `30` 的 1000× 差异钉死。

## 与 Icepak/CFD 对接

- 从求解器导出时记录 `meta.yaml`：`units`, `reference_pressure`, `turbulence_model`。  
- 反归一化公式写入 [VV_REPORT_TEMPLATE.md](VV_REPORT_TEMPLATE.md)。
