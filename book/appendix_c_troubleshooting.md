# 附录 C · 常见踩坑 50 问

> **用途**：环境、训练、精度、工程预期类问题的快速检索。  
> 按主题分组；可用编辑器搜索关键词。

---

## C.1 环境与安装（1–12）

**1. `pip install nvidia-physicsnemo` 成功但 `import physicsnemo` 失败？**  
先确认 Python ≥3.10，且与 `pip` 同一解释器：`python -c "import physicsnemo"`。

**2. `physicsnemo.sym` 找不到？**  
需额外安装：`pip install nvidia-physicsnemo.sym`。

**3. CUDA available: False？**  
驱动未装、PyTorch 为 CPU 版、或云主机未选 GPU 规格。用 `nvidia-smi` 与 `torch.cuda.is_available()` 对照。

**4. CUDA 版本与 PyTorch 不匹配？**  
到 [pytorch.org](https://pytorch.org) 选对应 cu12x 的安装命令，或改用 NGC 容器（附录 B）。

**5. 该用哪个 Python 版本？**
本书以 **Python 3.10+（推荐 3.11）** 为准；`scripts/check_env.py` 低于 3.10 会直接报错。见 [ENVIRONMENT.md](../docs/ENVIRONMENT.md)。

**6. Windows 能跑吗？**  
第 1–2 章裸 PyTorch 一般可以；SDK 与分布式建议 **WSL2 + Ubuntu** 或 Linux 云主机。

**7. Mac M 系列能跑全书吗？**  
前 3 章可 CPU/MPS；第 4 章起建议云 GPU（附录 B）。

**8. `check_env` 全绿但跑 SDK 仍报错？**  
SDK 可能依赖 SymPy、Hydra 等；按报错 `pip install` 补齐，或对照该章 `README`。

**9. Hydra 报错 `Config not found`？**  
须在对应章节目录下运行，或指定 `--config-path` / `--config-name`（见该章 `conf/`）。

**10. Docker 里看不到 GPU？**  
安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)，`docker run` 加 `--gpus all`。

**11. 磁盘满了？**  
删除各章 `outputs/`、`data/*.pt`（可重新生成），清理 `~/.cache/pip`。

**12. 公司内网无法访问 GitHub？**  
镜像 clone 或离线拷贝仓库；PyTorch/PhysicsNeMo 需内网 PyPI 镜像或预下载 wheel。

---

## C.2 训练与 loss（13–28）

**13. PDE loss 一直停在 1e-1？**  
检查 `create_graph=True`、学习率、配点数量、损失权重（IC/BC 是否过小）。

**14. 三条 loss 有一条不降？**  
分别画分量曲线；常是 BC 权重不足或边界采样太少。

**15. loss 变成 NaN？**  
降低学习率；检查除零、exp 溢出；混合精度可先关闭 AMP 排查。

**16. 训练很慢？**  
减小 `n_pde`、网络宽度；确认是否误用 CPU；GPU 版用 `*_gpu.py`。

**17. 结果图是直线/常数？**  
网络未学到；查输入归一化、输出尺度、是否忘记 `model.train()`。

**18. MLP 很好、PINN 很差？**  
正常情况之一：PINN 超参更敏感；先对齐同一量纲再比。

**19. 复现不了书中 loss 数值？**  
随机种子、硬件、PyTorch 版本不同会导致差异；看**趋势**是否一致（见 C.4 工时问题）。

**20. 要多训多少 epoch？**  
以验证曲线平台期为准；本书 demo 多为几百–几千 epoch 量级。

**21. 配点越多越好吗？**  
收益递减；2D 问题可先从 2k–10k 内部点试起。

**22. 边界采样不均？**  
复杂几何需在尖角、接口处加密；Sym 版用约束密度参数。

**23. FNO 数据损失不降？**  
查数据形状 `(B,C,H,W)`、学习率、是否用错归一化。

**24. ch05 报 `No module named fno_model`？**  
须先 clone 完整仓库，且 **不要删除 ch04**；ch05 依赖 ch04 的 `fno_model.py`。

**25. `darcy_data.pt` 从哪来？**  
首次运行 ch04/ch05 训练脚本会自动生成于 `data/`。

**26. 多卡 DDP 卡住？**  
检查 `torchrun` 参数、`MASTER_ADDR`；防火墙关闭多机端口。

**27. Optuna HPO 太慢？**  
减少 `n_trials`、缩小搜索空间；先用单卡跑通再 HPO。

**28. ONNX 导出失败？**  
确认 `model.eval()`、固定输入 shape、opset 版本与部署端一致。

---

## C.3 PhysicsNeMo 与框架（29–38）

**29. sym 和 main 框架有什么区别？**  
sym：PINN、符号 PDE；main：FNO/AFNO、数据驱动训练。第 4 章起以 main 为主。

**30. 能否只用 sym 跑 FNO？**  
不推荐；FNO 在 `physicsnemo.models` 主框架中维护。

**31. Modulus 和 PhysicsNeMo 是什么关系？**  
Modulus 为旧名，现统一为 PhysicsNeMo v2。

**32. API 和网上旧教程不一致？**  
以本仓库 `*_sdk.py` 与 [官方文档](https://github.com/NVIDIA/physicsnemo) 为准；v2 有破坏性变更。

**33. 必须用 SymPy 吗？**  
裸 PyTorch 版不需要；SDK sym 版用 SymPy 定义 PDE。

**34. 几何 CSG 报错？**  
检查 primitive 是否相交、参数范围；见 ch03 `heat_sink_geometry.py`。

**35. Validator 与 Constraint 区别？**  
Constraint 参与训练损失；Validator 仅评估、不参与反传。

**36. 权重 `w_pde`, `w_ic` 怎么设？**  
先从同量级开始；BC/IC 常需更大权重（如 10×）。

**37. 能否用 DeepXDE 代替？**  
学习原理可以；本书代码与工业部署路径按 PhysicsNeMo 设计。

**38. 官方 example 和本书关系？**  
本书为「从零实现 + 三档对照」；官方 example 适合查 API 细节。

---

## C.4 精度、工时与预期（39–45）

### 我看到的工时对比可信吗？

**39. 书中「ANSYS 1 小时 vs PINN 0.1 秒」可信吗？**  
**对比的是不同环节**：传统侧是**单次全精度仿真**；AI 侧是**训练完成后的推理**。训练本身 often 需 **30 分钟–数小时**（与网格规模、数据量有关）。不要把「推理 0.1 秒」理解成「从零到出结果只要 0.1 秒」。

**40. 表中的 10–60 分钟从哪来？**  
行业经验范围，依赖网格、物理场、求解器设置；**非本书 benchmark 实测**。

**41. PINN 能替代 ANSYS 吗？**  
一般**不能全面替代**；适合代理模型、快速扫描、反问题。签审级结果仍常需传统求解器抽检。

**42. AI 结果和 CFD 差很多怎么办？**  
看相对误差分布、守恒量、边界层；必要时加物理损失、更多数据或网格对齐训练。

**43. 1000 工况 5 分钟怎么算出来的？**  
假设**模型已训练好**，1000 次前向推理 batch 化；不含训练与数据准备。

**44. 数字孪生 <100ms 现实吗？**  
小模型 + GPU/TensorRT 可达；需单独做推理优化与 SLA 测试。

**45. 如何向老板解释 ROI？**  
强调减少**等待与扫描次数**，而非宣称「求解器错误率更低」；列出训练一次性成本 vs 每次仿真节省人工时。

---

## C.5 数据、部署与杂项（46–50）

**46. 没有真实数据能学吗？**  
本书 7 章均可用合成数据；落地项目需逐步替换真实仿真/实验数据。

**47. 合成数据训的模型能上线吗？**  
需域适应与验证；合成数据用于**学流程**，不用于直接签审。

**48. FastAPI 部署要注意什么？**  
`model.eval()`、输入校验、并发与 GPU 锁；生产用 Triton 更合适（ch07 延伸阅读）。

**49. 书中图片显示不出来？**  
配图见 [`book/assets/`](../book/assets/README.md)，可用 `book/scripts/` 下脚本重新生成。

**50. 还有问题怎么办？**  
到 [GitHub Issues](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues) 提问，附上 `check_env` 输出、完整报错、章节与命令。

---

➡️ **附录 A**：[数学速查](appendix_a_math.md)  
➡️ **附录 B**：[云 GPU 与环境](appendix_b_cloud_gpu.md)  
➡️ **附录 D**：[PyTorch 最小集](appendix_d_pytorch_mini.md)

---

*附录 C · v1.0 · 更新：2026-05-15*
