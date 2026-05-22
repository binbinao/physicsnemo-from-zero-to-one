# Appendix A · Mathematics & Notation Quick Reference

> **Purpose**: Look up formulas, loss terms, and notation encountered in Chapters 1–7.  
> **Prerequisites**: High-school physics + freshman calculus; functional analysis is not required.

---

## A.1 Notation Used Throughout the Book

| Symbol | Meaning | First Appearance |
|:---|:---|:---|
| $x, t$ | Space, time independent variables | Chapter 2 |
| $u(x,t)$ | Unknown field (e.g., temperature) | Chapter 2 |
| $m, k$ | Mass, spring stiffness | Chapter 1 |
| $\alpha$ | Thermal diffusivity | Chapter 2 |
| $\nabla$, $\nabla^2$ | Gradient, Laplacian | Chapter 3 |
| $\mathcal{F}$ | Fourier transform | Chapter 4 |
| $\mathcal{N}$ | Neural operator (input function → output function) | Chapter 4 |

---

## A.2 Chapter 1: Spring–Mass Oscillator (ODE)

**Equation** (undamped simple harmonic motion):

$$m \frac{d^2 x}{dt^2} + k x = 0$$

**Analytical solution** (initial conditions $x(0)=x_0$, $\dot{x}(0)=v_0$):

$$x(t) = x_0 \cos(\omega t) + \frac{v_0}{\omega}\sin(\omega t), \quad \omega = \sqrt{k/m}$$

**PINN residual** (network output $\hat{x}(t)$):

$$r_{\text{ODE}}(t) = m \frac{d^2 \hat{x}}{dt^2} + k \hat{x}$$

Loss: $\mathcal{L}_{\text{PDE}} = \mathbb{E}_t[r_{\text{ODE}}^2]$, plus an IC loss constraining position and velocity at $t=0$.

---

## A.3 Chapter 2: 1D Heat Conduction (PDE)

**Equation**:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

**PINN residual**:

$$r(x,t) = \frac{\partial \hat{u}}{\partial t} - \alpha \frac{\partial^2 \hat{u}}{\partial x^2}$$

**Boundary/initial conditions** (book defaults):

- IC: $u(x,0) = u_0(x)$
- BC: Dirichlet (prescribed boundary temperature) or Neumann (prescribed heat flux)

**Automatic differentiation notes**: When computing $\partial \hat{u}/\partial t$ for $\hat{u}(x,t)$, the inputs must have `requires_grad=True`; for second-order derivatives, the first `grad` call requires `create_graph=True`.

---

## A.4 Chapter 3: 2D Steady-State Heat (Elliptic PDE)

**Equation**:

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0 \quad \text{(steady state, no source term)}$$

**Multiple boundary conditions**: Different boundary segments can have different Dirichlet/Neumann conditions; PhysicsNeMo-Sym uses `Constraint` to express them segment by segment.

---

## A.5 Chapters 4–5: Darcy Flow & FNO

**Darcy equation (simplified)**:

$$-\nabla \cdot (a(x) \nabla u) = f$$

where $a(x)$ is the permeability field and $u$ is the pressure. Chapter 5 adds a PDE residual as a regularization term on top of the data loss.

**FNO core idea** (informal):

1. Mix low-frequency modes via learnable linear layers in the frequency domain;
2. Inverse-transform back to physical space;
3. The network learns an operator mapping "input coefficient field → output field," rather than pointwise function values.

---

## A.6 Loss Function Comparison Table

| Name | Formula (schematic) | Chapter |
|:---|:---|:---|
| Data loss | $\| \hat{u} - u_{\text{data}} \|^2$ | 1 (MLP), 4–7 |
| PDE loss | $\| r_{\text{PDE}} \|^2$ | 1–3, 5 |
| IC loss | $\| \hat{u}(\cdot,0) - u_0 \|^2$ | 1–2 |
| BC loss | $\| \hat{u}|_{\partial\Omega} - g \|^2$ | 2–3 |
| Hybrid loss | $\lambda_d \mathcal{L}_{\text{data}} + \lambda_p \mathcal{L}_{\text{physics}}$ | 5 |

---

## A.7 Chain Rule for Automatic Differentiation (PyTorch)

If $\hat{u} = \text{NN}(x,t)$, then:

$$\frac{\partial \hat{u}}{\partial x} = \frac{\partial \hat{u}}{\partial \text{NN}} \cdot \frac{\partial \text{NN}}{\partial x}$$

`torch.autograd.grad(outputs, inputs, grad_outputs=..., create_graph=True)` implements this process. **Do not** manually compute derivatives on numpy arrays and feed them back into the network.

---

## A.8 Recommended Reading Order

| What to review | Resource |
|:---|:---|
| Calculus refresher | Any freshman calculus textbook (partial derivatives chapter) |
| PINN fundamentals | Raissi et al., *JCP* 2019 (classic PINN paper) |
| FNO | Li et al., *ICLR* 2021 |
| Book code | `*_raw.py` files in each chapter's `chXX_*` directory |

---

## A.9 Universal Approximation Theorem & Theoretical Basis of PINNs

### Theorem Statement (Informal Form)

> Let $\sigma$ be any non-constant, bounded, monotonically increasing continuous function (e.g., sigmoid, tanh).  
> For any continuous function $f: [0,1]^d \to \mathbb{R}$ and any $\varepsilon > 0$,  
> there exists a single-hidden-layer network of width $N$: $g(x) = \sum_{i=1}^{N} c_i \, \sigma(w_i \cdot x + b_i)$,  
> such that $\sup_{x \in [0,1]^d} |f(x) - g(x)| < \varepsilon$.

This theorem was first proved by Cybenko (1989) for sigmoid activations and generalized by Hornik et al. (1991) to a broader class of activation functions.

### Connection to PINNs

The core assumption of PINNs: the true PDE solution $u^*$ is continuous on the compact set $\overline{\Omega}$ (which holds for most physical problems).  
The universal approximation theorem guarantees:

$$\forall \varepsilon > 0, \quad \exists\, \theta^* \text{ such that } \|u_{\theta^*} - u^*\|_\infty < \varepsilon$$

That is, there **exists** a set of network parameters that can approximate the PDE solution to arbitrary precision.

### Important Caveats

- **Existence ≠ Findability**: The theorem only guarantees that $\theta^*$ exists; gradient descent may not find it.
- **Width may be extremely large**: When approximating high-frequency functions, $N$ may grow exponentially with $\varepsilon^{-1}$.
- **Depth helps**: Deep networks can approximate with polynomially many parameters what shallow networks require exponentially many parameters to express (depth separation theorem).

---

## A.10 PINN Error Decomposition

### Total Error Decomposition

Let $u_{\theta}$ be the approximate solution obtained by PINN training, and $u^*$ the true PDE solution. The total error decomposes into three terms:

$$\|u_{\theta} - u^*\| \leq \underbrace{\varepsilon_{\text{approx}}}_{\text{approximation error}} + \underbrace{\varepsilon_{\text{optim}}}_{\text{optimization error}} + \underbrace{\varepsilon_{\text{gen}}}_{\text{generalization error}}$$

### Meaning and Mitigation Strategies

| Error Term | Source | Mitigation |
|:---|:---|:---|
| $\varepsilon_{\text{approx}}$ | Limited network capacity; cannot exactly represent $u^*$ | Increase network width/depth; use residual connections |
| $\varepsilon_{\text{optim}}$ | Non-convex loss landscape; optimizer stuck in local minima | Learning rate scheduling; curriculum training; multiple restarts |
| $\varepsilon_{\text{gen}}$ | Finite collocation points sample only a subset of $\Omega$ | Increase $N_{\text{col}}$; use adaptive sampling |

### Key Insight

$$\mathcal{L}_{\text{train}} \to 0 \quad \not\Rightarrow \quad \|u_\theta - u^*\| \to 0$$

Training loss approaching zero only means the residual is zero at the **sampled collocation points**. If $\varepsilon_{\text{gen}}$ is not controlled (e.g., sparse collocation points, solution with localized high-gradient features), the network may deviate from the true solution in unsampled regions. This explains why some PINN trainings that appear to "converge" still exhibit large validation errors.

### Practical Diagnostics

- If training loss is low but validation error is high → primarily $\varepsilon_{\text{gen}}$; add more collocation points or improve sampling strategy
- If training loss plateaus → primarily $\varepsilon_{\text{optim}}$; adjust optimizer or network initialization
- If increasing network size shows no significant improvement → likely in the $\varepsilon_{\text{optim}}$-dominated regime

---

## A.11 Collocation Method Convergence Rates

### Random Collocation (Monte Carlo)

Given $N$ uniformly random collocation points in a $d$-dimensional domain $\Omega$, the $L^2$ approximation error of the PDE residual is:

$$\|\mathcal{L}_N - \mathcal{L}\|_{L^2} = O(N^{-1/2})$$

- The convergence rate $N^{-1/2}$ is **independent of dimension $d$** (the core advantage of Monte Carlo)
- However, convergence is slow: halving the error requires 4× more collocation points

### Quasi-Monte Carlo Methods

Using low-discrepancy sequences (Sobol, Halton) instead of pseudo-random numbers:

$$\text{error} = O\left(\frac{(\log N)^d}{N}\right)$$

- Significantly better than pure random for low dimensions ($d \leq 6$)
- In high dimensions, the logarithmic factor $(\log N)^d$ can be large, reducing the advantage
- In PhysicsNeMo, Sobol sequences can be used via custom `PointwiseConstraint`

### Residual-based Adaptive Refinement (RAR)

Core idea: place more collocation points in regions with large residuals.

$$p(x) \propto |r(x)|^k, \quad k \geq 1$$

- Can achieve faster convergence than $O(N^{-1/2})$
- Cost: residual distribution must be re-evaluated each iteration
- In practice, $k=1$ or $k=2$ works well

### References

- Shin, Y., Darbon, J., & Karniadakis, G.E. (2020). "On the convergence of physics-informed neural networks." *Journal of Computational Physics*, 419, 109665.
- Lu, L. et al. (2021). "DeepXDE: A deep learning library for solving differential equations." *SIAM Review*, 63(1), 208-228.

---

## A.12 Operator Approximation Theorem for FNO

### Theoretical Origins of Operator Approximation

Chen & Chen (1995) proved the operator version of the universal approximation theorem:

> Let $G: U \to V$ be a continuous nonlinear operator between Banach spaces.  
> Then there exists a branch-trunk architecture neural network $G_\theta$ such that for any $\varepsilon > 0$ on a compact subset of $U$:  
> $\sup_{u \in K} \|G(u) - G_\theta(u)\|_V < \varepsilon$

This is the theoretical foundation of DeepONet (Lu et al., 2021).

### FNO's Approximation Mechanism

FNO approximates operators from a different angle:

1. **Frequency-domain linear mixing**: In Fourier space, convolution becomes pointwise multiplication — FNO learns a representation of the Green's function in the frequency domain
2. **Mode truncation**: Retaining only the first $k_{\max}$ Fourier modes is equivalent to imposing a smoothness prior on the solution
3. **Resolution invariance**: Once trained, the model can be evaluated on grids of any resolution (zero-shot super-resolution)

### Theoretical Guarantee (Kovachki et al., 2021)

Under appropriate smoothness conditions:

$$\|G - G_\theta\|_{L^2(U; H^s)} \leq C \cdot k_{\max}^{-r}$$

where $r$ depends on the Sobolev regularity of the true solution, and $s$ is the smoothness order of the target space.

### Key Limitations

| Limitation | Explanation | Consequence |
|:---|:---|:---|
| Smoothness assumption | Requires the solution to belong to $H^s$ ($s$ sufficiently large) | Difficult to approximate shocks and discontinuous solutions |
| Periodicity preference | Standard FFT assumes periodic boundaries | Non-periodic problems require padding or hybrid methods |
| Mode truncation | Discards high-frequency information | Cannot capture sub-grid scale details |
| Computational complexity | $O(N \log N)$ per layer ($N$ = number of grid points) | 3D high-resolution problems remain challenging |

### References

- Chen, T. & Chen, H. (1995). "Universal approximation to nonlinear operators." *IEEE Trans. Neural Networks*, 6(4), 911-917.
- Kovachki, N. et al. (2021). "Universal approximation of input-output maps by neural operators." *arXiv:2306.03809*.
- Li, Z. et al. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*.

---

## A.13 Extended Symbol Table

The table below summarizes all core symbols used throughout the book, organized along three dimensions: "mathematical / physical meaning / ML framework correspondence."

### Network and Optimization Symbols

| Symbol | Meaning | First Appearance |
|:---|:---|:---|
| $\theta$ | Network parameters (collection of weights + biases) | Chapter 1 |
| $\hat{u},\; u_\theta$ | Network prediction ($\theta$-parameterized approximate solution) | Chapter 1 |
| $u^*$ | True PDE solution / analytical solution | Chapter 1 |
| $\lambda_{\text{pde}},\; \lambda_{\text{ic}},\; \lambda_{\text{bc}}$ | Weight coefficients for each term in the composite loss | Chapter 1 |
| $\mathcal{L}$ | Total loss function | Chapter 1 |
| $\eta$ | Learning rate | Chapter 1 |
| $r(x,t)$ | PDE residual | Chapter 1 |
| $\sigma(\cdot)$ | Activation function (ReLU, tanh, SiLU, etc.) | Chapter 1 |

### Physical Domain and PDE Symbols

| Symbol | Meaning | First Appearance |
|:---|:---|:---|
| $\Omega$ | Computational domain (spatial region) | Chapter 2 |
| $\partial\Omega$ | Domain boundary | Chapter 2 |
| $\alpha$ | Thermal diffusivity $[\text{m}^2/\text{s}]$ | Chapter 2 |
| $N_{\text{col}}$ | Number of collocation points | Chapter 2 |
| $\nabla$ | Gradient operator $(\partial/\partial x_1, \ldots, \partial/\partial x_d)$ | Chapter 3 |
| $\nabla^2,\; \Delta$ | Laplacian operator | Chapter 3 |
| $\Delta x,\; dx$ | Grid spacing / collocation point spacing | Chapter 4 |

### FNO / Operator Learning Symbols

| Symbol | Meaning | First Appearance |
|:---|:---|:---|
| $k_{\max}$ | FNO truncation mode count (Fourier modes kept) | Chapter 4 |
| $a(x)$ | Darcy permeability field (input function) | Chapter 4 |
| $G$ | Green's function / solution operator | Chapter 4 |
| $\mathcal{F},\; \mathcal{F}^{-1}$ | Fourier transform and its inverse | Chapter 4 |
| $\mathcal{N}$ | Neural operator mapping $a \mapsto u$ | Chapter 4 |
| $R_\phi$ | Learnable weight tensor in frequency domain | Chapter 4 |
| $P,\; Q$ | FNO lifting/projection matrices | Chapter 4 |

### Error and Convergence Symbols

| Symbol | Meaning | First Appearance |
|:---|:---|:---|
| $\varepsilon_{\text{approx}}$ | Approximation error (limited network expressiveness) | Appendix A.10 |
| $\varepsilon_{\text{optim}}$ | Optimization error (failed to find global optimum) | Appendix A.10 |
| $\varepsilon_{\text{gen}}$ | Generalization error (caused by finite sampling) | Appendix A.10 |
| $O(\cdot)$ | Big-O asymptotic notation | Appendix A.11 |
| $H^s$ | Sobolev space (space of $s$-times differentiable functions) | Appendix A.12 |

---

➡️ **Appendix B**: [Cloud GPU & Environment](appendix_b_cloud_gpu.md)  
➡️ **Appendix C**: [50 Common Pitfalls](appendix_c_troubleshooting.md)  
➡️ **Appendix D**: [PyTorch Minimal Subset](appendix_d_pytorch_mini.md)

---

*Appendix A · v1.1 · Updated: 2026-05-21*
