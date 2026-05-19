# 参考解对照实验设计（C33）

## ch01 弹簧

- 参考：解析解 $x(t) = A\cos(\omega t)$  
- 指标：$L^2$ 相对误差、外推区间误差  

## ch02 1D 热

- 参考：Fourier 级数或高精度 FDM 在同一 $(x,t)$ 网格  
- 对比：线图中 PINN vs 参考  

## ch03 散热片

- 最低：运行 `validator.py` 残差报告  
- 推荐：Icepak/Flotherm **单工况** 导出 $T(x,y)$， subsample 做点wise MAE  
- 注意：本书代码为 **2D 稳态**，与 3D 瞬态封装对齐时需降阶说明  

## ch04 翼型 / Darcy

- **路径 A（Darcy）**：与数值解同网格的 FD 参考（`darcy_residual` 应 $\to 0$）  
- **路径 B（翼型）**：AirfRANS 子集；记录 **RANS + 湍流模型 + $Ma$, $\alpha$**  

## ch07 汽车 Cd

- toy：与 `generate_toy_cars.py` 公式对比（sanity）  
- 生产：与 CFD $C_d$ 散点图 + holdout 几何  

报告格式见 [VV_REPORT_TEMPLATE.md](VV_REPORT_TEMPLATE.md)。
