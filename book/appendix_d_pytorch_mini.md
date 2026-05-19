# 附录 D · PyTorch 30 分钟最小集

> **用途**：读第 1 章前，补齐「能看懂 PINN 代码」所需的最少 PyTorch。  
> **不是**完整 PyTorch 教程；深入请用 [官方 60 分钟入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)。

---

## D.1 本书里你会反复见到的 6 个概念

| PyTorch | 本书对应 |
|:---|:---|
| `torch.Tensor` | 网络输入/输出、物理场 $u(x,t)$ 在网格上的值 |
| `requires_grad=True` | PINN 要对坐标求导 → 需要梯度 |
| `nn.Module` | MLP、FNO、AFNO 都是子类 |
| `loss.backward()` | 反向传播，更新权重 |
| `optimizer.step()` | 每步训练 |
| `torch.autograd.grad` | PINN 的 PDE 残差（第 1–3 章核心） |

---

## D.2 10 行：张量与自动微分

```python
import torch

x = torch.linspace(0, 1, 5, requires_grad=True)
y = x ** 2
dy_dx = torch.autograd.grad(y, x, torch.ones_like(x), create_graph=True)[0]
print(dy_dx)  # 2*x
```

**要点**：`create_graph=True` 表示「还要对导数再求导」——第 2 章二阶 PDE 导数必须这样写。

---

## D.3 15 行：最小训练循环（对应 ch01 MLP）

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(100):
    x = torch.rand(32, 1)
    y_true = torch.sin(x)
    pred = model(x)
    loss = loss_fn(pred, y_true)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

对照仓库：`ch01_hello/mlp_spring.py`。

---

## D.4 PINN 与 MLP 的差别（1 分钟）

| | MLP（数据驱动） | PINN（物理驱动） |
|:---|:---|:---|
| 损失 | $\| \hat u - u_{\text{data}} \|^2$ | $\| \text{PDE残差} \|^2 + \text{IC/BC}$ |
| 数据 | 需要标签 | 可**零数据**，靠方程 |
| 代码 | 仅 `loss.backward()` | 多几次 `autograd.grad` 构造残差 |

对照：`ch01_hello/pinn_spring.py`。

---

## D.5 设备与 GPU（可选）

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x = x.to(device)
```

第 1–2 章 CPU 即可；第 4 章起建议 GPU（见 [附录 B](appendix_b_cloud_gpu.md)）。

---

## D.6 进一步学习（外链）

| 资源 | 适合 |
|:---|:---|
| [PyTorch 60min Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | 系统入门 |
| [autograd 机制](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | 理解 PINN |
| 本书 ch01 → ch02 | 边做边学 |

---

➡️ **附录 A**：[数学速查](appendix_a_math.md)  
➡️ **第 1 章**：[ch01.md](ch01.md)

---

*附录 D · v1.0 · 更新：2026-05-15*
