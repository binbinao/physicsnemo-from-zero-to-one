# toy → 真实 CFD 数据迁移检查表（C19）

- [ ] 设计变量定义与训练集 **一致**（命名、范围、单位）  
- [ ] 重新计算 `input_mean/std` 或归一化参数  
- [ ] 湍流/物理模型与训练数据 **一致**  
- [ ] holdout 几何未出现在训练中  
- [ ] 在 10–20 个 CFD 点上做点wise / $C_d$ 对比后再全量优化  
- [ ] 更新 checkpoint `meta.data_tier` = `production`  
- [ ] 填写 [VV_REPORT_TEMPLATE.md](VV_REPORT_TEMPLATE.md)
