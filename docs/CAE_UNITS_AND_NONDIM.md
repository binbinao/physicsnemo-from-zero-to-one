# CAE 单位与无量纲化

## 本书默认

| 章 | 典型约定 | 换回 SI |
|:---|:---|:---|
| ch02 1D 热 | $x,t,u$ 无量纲或归一化到 $[0,1]$ | 用特征长度 $L$、时间 $L^2/\alpha$ |
| ch03 散热片 | 几何 **mm**，$T$ 与 $T_{source}$、$T_\infty$ 同单位（℃ 或 K 一致即可） | 导热系数 $k$ 与几何单位自洽 |
| ch04–ch05 Darcy | 渗透率/压力场多为 **归一化** 合成数据 | 对接客户 CFD 时需重标定 |
| ch07 Cd | 无量纲 $C_d$ | 与风洞/CFD 一致即可 |

## 签审建议

1. 训练前写清 **参考量**：$L_{ref}, T_{ref}, U_{ref}, \rho, \mu$。  
2. 代理模型 I/O 与 **求解器导出场** 用同一套归一化（存于 checkpoint `meta`）。  
3. 报告同时给 **无量纲误差** 与 **SI 误差**（如最高温度 ℃）。

## 与 Icepak/CFD 对接

- 从求解器导出时记录 `meta.yaml`：`units`, `reference_pressure`, `turbulence_model`。  
- 反归一化公式写入 [VV_REPORT_TEMPLATE.md](VV_REPORT_TEMPLATE.md)。
