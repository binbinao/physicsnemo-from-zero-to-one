# 附录 A · 数学与符号速查

> **用途**：读第 1–7 章时遇到公式、损失项、符号可在此查阅。  
> **前置**：高中物理 + 大一微积分即可；不要求泛函分析。

---

## A.1 全书常用符号

| 符号 | 含义 | 首次出现 |
|:---|:---|:---|
| $x, t$ | 空间、时间自变量 | 第 2 章 |
| $u(x,t)$ | 待求场（如温度） | 第 2 章 |
| $m, k$ | 质量、弹簧刚度 | 第 1 章 |
| $\alpha$ | 热扩散系数 | 第 2 章 |
| $\nabla$, $\nabla^2$ | 梯度、拉普拉斯 | 第 3 章 |
| $\mathcal{F}$ | 傅里叶变换 | 第 4 章 |
| $\mathcal{N}$ | 神经算子（输入函数→输出函数） | 第 4 章 |

---

## A.2 第 1 章：弹簧振子（ODE）

**方程**（无阻尼简谐振动）：

$$m \frac{d^2 x}{dt^2} + k x = 0$$

**解析解**（初值 $x(0)=x_0$, $\dot{x}(0)=v_0$）：

$$x(t) = x_0 \cos(\omega t) + \frac{v_0}{\omega}\sin(\omega t), \quad \omega = \sqrt{k/m}$$

**PINN 残差**（网络输出 $\hat{x}(t)$）：

$$r_{\text{ODE}}(t) = m \frac{d^2 \hat{x}}{dt^2} + k \hat{x}$$

损失：$\mathcal{L}_{\text{PDE}} = \mathbb{E}_t[r_{\text{ODE}}^2]$，再加 IC 损失约束 $t=0$ 时的位置与速度。

---

## A.3 第 2 章：1D 热传导（PDE）

**方程**：

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

**PINN 残差**：

$$r(x,t) = \frac{\partial \hat{u}}{\partial t} - \alpha \frac{\partial^2 \hat{u}}{\partial x^2}$$

**边界/初值**（本书默认）：

- IC：$u(x,0) = u_0(x)$
- BC：Dirichlet（给定边界温度）或 Neumann（给定热流）

**自动微分要点**：对 $\hat{u}(x,t)$ 求 $\partial \hat{u}/\partial t$ 时，输入须 `requires_grad=True`；求二阶导时第一次 `grad` 要 `create_graph=True`。

---

## A.4 第 3 章：2D 稳态热（椭圆型 PDE）

**方程**：

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0 \quad \text{（稳态，无源项）}$$

**多边界条件**：不同边界段可设不同 Dirichlet/Neumann；PhysicsNeMo-Sym 用 `Constraint` 分段表达。

---

## A.5 第 4–5 章：Darcy 流与 FNO

**Darcy 方程（简化）**：

$$-\nabla \cdot (a(x) \nabla u) = f$$

其中 $a(x)$ 为渗透率场，$u$ 为压力。第 5 章在数据损失外叠加 PDE 残差作为正则项。

**FNO 核心思想**（非严格证明）：

1. 在频域用可学习的线性层混合低模态；
2. 逆变换回物理空间；
3. 学的是「输入系数场 → 输出场」的算子，而非单点函数值。

---

## A.6 损失函数对照表

| 名称 | 公式（示意） | 章节 |
|:---|:---|:---|
| 数据损失 | $\| \hat{u} - u_{\text{data}} \|^2$ | 1（MLP）, 4–7 |
| PDE 损失 | $\| r_{\text{PDE}} \|^2$ | 1–3, 5 |
| IC 损失 | $\| \hat{u}(\cdot,0) - u_0 \|^2$ | 1–2 |
| BC 损失 | $\| \hat{u}|_{\partial\Omega} - g \|^2$ | 2–3 |
| 混合损失 | $\lambda_d \mathcal{L}_{\text{data}} + \lambda_p \mathcal{L}_{\text{physics}}$ | 5 |

---

## A.7 自动微分链式法则（PyTorch）

若 $\hat{u} = \text{NN}(x,t)$，则：

$$\frac{\partial \hat{u}}{\partial x} = \frac{\partial \hat{u}}{\partial \text{NN}} \cdot \frac{\partial \text{NN}}{\partial x}$$

`torch.autograd.grad(outputs, inputs, grad_outputs=..., create_graph=True)` 实现上述过程。**不要**对 numpy 数组手算导数再喂回网络。

---

## A.8 推荐阅读顺序

| 想补什么 | 资源 |
|:---|:---|
| 微积分复习 | 任意大一《高等数学》偏导数章节 |
| PINN 原理 | Raissi et al., *JCP* 2019（经典 PINN 论文） |
| FNO | Li et al., *ICLR* 2021 |
| 本书代码 | 各章 `chXX_*` 目录下 `*_raw.py` |

---

## A.9 万有逼近定理与 PINN 的理论基础

### 定理陈述（非严格形式）

> 设 $\sigma$ 为任意非常值、有界、单调递增的连续函数（如 sigmoid、tanh）。  
> 对任意连续函数 $f: [0,1]^d \to \mathbb{R}$ 及任意 $\varepsilon > 0$，  
> 存在宽度为 $N$ 的单隐层网络 $g(x) = \sum_{i=1}^{N} c_i \, \sigma(w_i \cdot x + b_i)$，  
> 使得 $\sup_{x \in [0,1]^d} |f(x) - g(x)| < \varepsilon$。

该定理最早由 Cybenko (1989) 针对 sigmoid 证明，Hornik et al. (1991) 推广至更广泛的激活函数类。

### 与 PINN 的联系

PINN 的核心假设：PDE 真解 $u^*$ 在紧集 $\overline{\Omega}$ 上连续（对大多数物理问题成立）。  
万有逼近定理保证：

$$\forall \varepsilon > 0, \quad \exists\, \theta^* \text{ 使得 } \|u_{\theta^*} - u^*\|_\infty < \varepsilon$$

即**存在**一组网络参数能以任意精度逼近 PDE 解。

### 重要警告

- **存在性 ≠ 可找性**：定理仅保证 $\theta^*$ 存在，梯度下降不一定能找到它。
- **宽度可能极大**：逼近高频函数时，$N$ 可能随 $\varepsilon^{-1}$ 指数增长。
- **深度有帮助**：深层网络可以用多项式级别的参数量逼近浅层网络需要指数级参数量才能表达的函数（深度分离定理）。

---

## A.10 PINN 误差分解

### 总误差分解

设 $u_{\theta}$ 为 PINN 训练得到的近似解，$u^*$ 为 PDE 真解，总误差可分解为三项：

$$\|u_{\theta} - u^*\| \leq \underbrace{\varepsilon_{\text{approx}}}_{\text{逼近误差}} + \underbrace{\varepsilon_{\text{optim}}}_{\text{优化误差}} + \underbrace{\varepsilon_{\text{gen}}}_{\text{泛化误差}}$$

### 各项含义与缓解策略

| 误差项 | 来源 | 缓解方法 |
|:---|:---|:---|
| $\varepsilon_{\text{approx}}$ | 网络容量有限，无法精确表达 $u^*$ | 增大网络宽度/深度；使用残差连接 |
| $\varepsilon_{\text{optim}}$ | 非凸损失面，优化器陷入局部极小 | 学习率调度；课程式训练；多次重启 |
| $\varepsilon_{\text{gen}}$ | 有限配点仅采样了 $\Omega$ 的子集 | 增大 $N_{\text{col}}$；使用自适应采样 |

### 关键洞察

$$\mathcal{L}_{\text{train}} \to 0 \quad \not\Rightarrow \quad \|u_\theta - u^*\| \to 0$$

训练损失趋于零仅意味着在**已采样配点**上残差为零。若 $\varepsilon_{\text{gen}}$ 未被控制（例如配点稀疏、解具有局部高梯度特征），网络可能在未采样区域偏离真解。这解释了为何某些 PINN 训练"收敛"后验证误差仍然很大。

### 实用判据

- 若训练损失低但验证误差高 → 主要是 $\varepsilon_{\text{gen}}$，需增加配点或改进采样策略
- 若训练损失下降停滞 → 主要是 $\varepsilon_{\text{optim}}$，需调整优化器或网络初始化
- 若增大网络后误差无明显改善 → 可能已进入 $\varepsilon_{\text{optim}}$ 主导区间

---

## A.11 配点法收敛率

### 随机配点（Monte Carlo）

设在 $d$ 维域 $\Omega$ 中随机均匀采样 $N$ 个配点，PDE 残差的 $L^2$ 近似误差为：

$$\|\mathcal{L}_N - \mathcal{L}\|_{L^2} = O(N^{-1/2})$$

- 收敛率 $N^{-1/2}$ **与维度 $d$ 无关**（Monte Carlo 的核心优势）
- 但收敛慢：误差减半需配点数增加 4 倍

### 准 Monte Carlo 方法

使用低差异序列（Sobol、Halton）替代伪随机数：

$$\text{误差} = O\left(\frac{(\log N)^d}{N}\right)$$

- 低维（$d \leq 6$）时显著优于纯随机
- 高维时对数因子 $(\log N)^d$ 可能很大，优势减弱
- PhysicsNeMo 中可通过自定义 `PointwiseConstraint` 使用 Sobol 序列

### 自适应残差采样（Residual-based Adaptive Refinement, RAR）

核心思想：在残差大的区域投放更多配点。

$$p(x) \propto |r(x)|^k, \quad k \geq 1$$

- 可达到比 $O(N^{-1/2})$ 更快的收敛
- 代价：每轮需重新评估残差分布
- 实践中 $k=1$ 或 $k=2$ 效果良好

### 参考文献

- Shin, Y., Darbon, J., & Karniadakis, G.E. (2020). "On the convergence of physics-informed neural networks." *Journal of Computational Physics*, 419, 109665.
- Lu, L. et al. (2021). "DeepXDE: A deep learning library for solving differential equations." *SIAM Review*, 63(1), 208-228.

---

## A.12 FNO 的算子逼近定理

### 算子逼近的理论起源

Chen & Chen (1995) 证明了算子版万有逼近定理：

> 设 $G: U \to V$ 为定义在 Banach 空间之间的连续非线性算子。  
> 则存在 branch-trunk 架构的神经网络 $G_\theta$，对 $U$ 的紧子集上任意 $\varepsilon > 0$：  
> $\sup_{u \in K} \|G(u) - G_\theta(u)\|_V < \varepsilon$

这是 DeepONet（Lu et al., 2021）的理论基础。

### FNO 的逼近机制

FNO 从不同角度逼近算子：

1. **频域线性混合**：在傅里叶空间中，卷积变为逐点乘法——FNO 学习的是 Green 函数在频域的表示
2. **模态截断**：仅保留前 $k_{\max}$ 个傅里叶模态，等价于对解的光滑性做先验假设
3. **分辨率无关性**：一旦训练完成，可在任意分辨率网格上评估（zero-shot super-resolution）

### 理论保证（Kovachki et al., 2021）

在适当光滑性条件下：

$$\|G - G_\theta\|_{L^2(U; H^s)} \leq C \cdot k_{\max}^{-r}$$

其中 $r$ 取决于真解的 Sobolev 正则性，$s$ 为目标空间的光滑阶。

### 关键限制

| 限制 | 说明 | 后果 |
|:---|:---|:---|
| 光滑性假设 | 需要解属于 $H^s$（$s$ 足够大） | 激波、间断解逼近困难 |
| 周期性偏好 | 标准 FFT 假设周期边界 | 非周期问题需 padding 或混合方法 |
| 模态截断 | 丢弃高频信息 | 无法捕捉亚网格尺度细节 |
| 计算复杂度 | $O(N \log N)$ per layer（$N$ 为网格点数） | 3D 高分辨率问题仍有挑战 |

### 参考文献

- Chen, T. & Chen, H. (1995). "Universal approximation to nonlinear operators." *IEEE Trans. Neural Networks*, 6(4), 911-917.
- Kovachki, N. et al. (2021). "Universal approximation of input-output maps by neural operators." *arXiv:2306.03809*.
- Li, Z. et al. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*.

---

## A.13 符号表（扩充）

下表汇总全书出现的核心符号，按「数学/物理含义/在 ML 框架中的对应」三维度整理。

### 网络与优化相关符号

| 符号 | 含义 | 首次出现 |
|:---|:---|:---|
| $\theta$ | 网络参数（weights + biases 的集合） | 第 1 章 |
| $\hat{u},\; u_\theta$ | 网络预测值（$\theta$ 参数化的近似解） | 第 1 章 |
| $u^*$ | PDE 真解 / 解析解 | 第 1 章 |
| $\lambda_{\text{pde}},\; \lambda_{\text{ic}},\; \lambda_{\text{bc}}$ | 复合损失中各项的权重系数 | 第 1 章 |
| $\mathcal{L}$ | 总损失函数 | 第 1 章 |
| $\eta$ | 学习率（learning rate） | 第 1 章 |
| $r(x,t)$ | PDE 残差（residual） | 第 1 章 |
| $\sigma(\cdot)$ | 激活函数（ReLU, tanh, SiLU 等） | 第 1 章 |

### 物理域与 PDE 相关符号

| 符号 | 含义 | 首次出现 |
|:---|:---|:---|
| $\Omega$ | 计算域（空间区域） | 第 2 章 |
| $\partial\Omega$ | 域边界 | 第 2 章 |
| $\alpha$ | 热扩散系数 $[\text{m}^2/\text{s}]$ | 第 2 章 |
| $N_{\text{col}}$ | 配点数量（collocation points） | 第 2 章 |
| $\nabla$ | 梯度算子 $(\partial/\partial x_1, \ldots, \partial/\partial x_d)$ | 第 3 章 |
| $\nabla^2,\; \Delta$ | 拉普拉斯算子 | 第 3 章 |
| $\Delta x,\; dx$ | 网格间距 / 配点间距 | 第 4 章 |

### FNO / 算子学习相关符号

| 符号 | 含义 | 首次出现 |
|:---|:---|:---|
| $k_{\max}$ | FNO 截断模态数（Fourier modes kept） | 第 4 章 |
| $a(x)$ | Darcy 渗透率场（输入函数） | 第 4 章 |
| $G$ | Green's function / 解算子 | 第 4 章 |
| $\mathcal{F},\; \mathcal{F}^{-1}$ | 傅里叶变换及其逆变换 | 第 4 章 |
| $\mathcal{N}$ | 神经算子映射 $a \mapsto u$ | 第 4 章 |
| $R_\phi$ | 频域可学习权重张量 | 第 4 章 |
| $P,\; Q$ | FNO 升维/降维投影矩阵 | 第 4 章 |

### 误差与收敛相关符号

| 符号 | 含义 | 首次出现 |
|:---|:---|:---|
| $\varepsilon_{\text{approx}}$ | 逼近误差（网络表达能力限制） | 附录 A.10 |
| $\varepsilon_{\text{optim}}$ | 优化误差（未找到全局最优） | 附录 A.10 |
| $\varepsilon_{\text{gen}}$ | 泛化误差（有限采样导致） | 附录 A.10 |
| $O(\cdot)$ | 大 O 渐近记号 | 附录 A.11 |
| $H^s$ | Sobolev 空间（$s$ 阶可微函数空间） | 附录 A.12 |

---

➡️ **附录 B**：[云 GPU 与环境](appendix_b_cloud_gpu.md)  
➡️ **附录 C**：[常见踩坑 50 问](appendix_c_troubleshooting.md)  
➡️ **附录 D**：[PyTorch 最小集](appendix_d_pytorch_mini.md)

---

*附录 A · v1.1 · 更新：2026-05-21*
