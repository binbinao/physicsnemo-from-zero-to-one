# CAE 守恒与残差检查清单

> PINN/FNO **不自动保证** 与 FEM/FDM 相同的离散守恒。签审前建议逐项勾选。

## PINN（ch01–ch03）

- [ ] 各损失项（PDE / BC）残差 RMS 是否均下降？  
- [ ] `ch03` 运行后查看 `outputs/validation_report.json`  
- [ ] 温度场是否满足 $T_\infty \leq T \leq T_{source}$（量级）？  
- [ ] 是否与 **参考解** 对比（解析解 / 粗网格 FEM / Icepak 抽检）？  
- [ ] 边界条件采样可视化（`sample_boundary` 散点图）？

## FNO / 数据驱动（ch04–ch06）

- [ ] train/val loss 趋势；外推工况 holdout  
- [ ] ch05：加大 `lambda_physics` 后 PDE 残差是否下降？  
- [ ] 全局积分量（阻力、总热流）与 CFD 对比，而非仅点wise MSE  

## 优化与部署（ch07）

- [ ] 优化结果是否在训练参数包络内？（`optimize.py` OOD 警告）  
- [ ] Top-K 是否已导出 [hifi_validation_queue.csv](../ch07_drivaernet_optim/outputs/hifi_queue.csv) 并 **回流 CFD**？  
- [ ] 复核后更新 [VV 报告](VV_REPORT_TEMPLATE.md)
