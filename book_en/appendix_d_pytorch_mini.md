# Appendix D · PyTorch 30-Minute Minimal Subset

> **Purpose**: Before reading Chapter 1, learn the minimum PyTorch needed to understand PINN code.  
> This is **not** a complete PyTorch tutorial; for more depth, see the [official 60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

---

## D.1 Six Concepts You'll See Repeatedly in This Book

| PyTorch | Corresponding Usage in This Book |
|:---|:---|
| `torch.Tensor` | Network inputs/outputs, values of the physical field $u(x,t)$ on the grid |
| `requires_grad=True` | PINNs need derivatives w.r.t. coordinates → gradients required |
| `nn.Module` | MLP, FNO, AFNO are all subclasses |
| `loss.backward()` | Backpropagation to update weights |
| `optimizer.step()` | One training step |
| `torch.autograd.grad` | PDE residual for PINNs (core of Chapters 1–3) |

---

## D.2 10 Lines: Tensors & Automatic Differentiation

```python
import torch

x = torch.linspace(0, 1, 5, requires_grad=True)
y = x ** 2
dy_dx = torch.autograd.grad(y, x, torch.ones_like(x), create_graph=True)[0]
print(dy_dx)  # 2*x
```

**Key point**: `create_graph=True` means "I need to differentiate the derivative again" — required for second-order PDE derivatives in Chapter 2.

---

## D.3 15 Lines: Minimal Training Loop (Corresponds to ch01 MLP)

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

Compare with the repo: `ch01_hello/mlp_spring.py`.

---

## D.4 The Difference Between PINN and MLP (1 Minute)

| | MLP (Data-Driven) | PINN (Physics-Driven) |
|:---|:---|:---|
| Loss | $\| \hat u - u_{\text{data}} \|^2$ | $\| \text{PDE residual} \|^2 + \text{IC/BC}$ |
| Data | Requires labels | Can be **zero-data**, relies on the equation |
| Code | Only `loss.backward()` | A few extra `autograd.grad` calls to construct the residual |

Compare with: `ch01_hello/pinn_spring.py`.

---

## D.5 Device & GPU (Optional)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x = x.to(device)
```

CPU is sufficient for Chapters 1–2; from Chapter 4 onwards, GPU is recommended (see [Appendix B](appendix_b_cloud_gpu.md)).

---

## D.6 Further Learning (External Links)

| Resource | Best For |
|:---|:---|
| [PyTorch 60min Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | Systematic introduction |
| [autograd mechanism](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | Understanding PINNs |
| This book: ch01 → ch02 | Learn by doing |

---

➡️ **Appendix A**: [Math Quick Reference](appendix_a_math.md)  
➡️ **Chapter 1**: [ch01.md](ch01.md)

---

*Appendix D · v1.0 · Updated: 2026-05-15*
