# 代理模型 V&V 报告（模板）

> 复制本页填写；ch07 训练后可从 checkpoint `meta` 取数据版本信息。

## 1. 项目信息

- 项目 / 日期 / 作者：  
- 代理类型：□ PINN  □ FNO  □ MLP  □ 其他  
- 代码版本：`git rev-parse HEAD` =  

## 2. 数据

| 项 | 值 |
|:---|:---|
| 来源 | □ toy  □ 客户 CFD  □ 公开数据集 |
| 样本数 train/val/test | |
| 求解器 / 湍流模型 | |
| 网格等级 / $y^+$ | |
| 数据版本 ID | |

## 3. 精度（须含 SI 与无量纲）

| 指标 | 代理 | 高保真参考 | 备注 |
|:---|:---|:---|:---|
| $C_d$ MAE | | | ch07 |
| $T_{max}$ 误差 (℃) | | | ch03 |
| 场 L2 相对误差 | | | ch04 |

## 4. 守恒 / 物理

- [ ] PDE 残差 RMS：  
- [ ] 全局力/热流平衡：  
- [ ] 见 [CAE_CONSERVATION_CHECKLIST.md](CAE_CONSERVATION_CHECKLIST.md)

## 5. 速度与 ROI

- 单次高保真：___ h  
- 代理推理：___ ms  
- 训练一次性：___ GPU·h  

## 6. 优化闭环（若适用）

- Top-K 候选数：  
- 已回流 CFD 复核：□ 是 □ 否  
- 复核后 $C_d$ 改善：___ %（[hifi_queue.csv](../ch07_drivaernet_optim/outputs/hifi_queue.csv)）

## 7. 结论

- [ ] 可用于 **方案筛选**  
- [ ] 可用于 **签审**（须签字部门：___）  
- 限制说明：  
