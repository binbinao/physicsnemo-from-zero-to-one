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

➡️ **附录 B**：[云 GPU 与环境](appendix_b_cloud_gpu.md)  
➡️ **附录 C**：[常见踩坑 50 问](appendix_c_troubleshooting.md)
