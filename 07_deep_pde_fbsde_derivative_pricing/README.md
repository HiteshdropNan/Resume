# Deep FBSDE Solver: High-Dimensional Derivative Pricing via Neural Networks

> **Program:** Certificate in Quantitative Finance (CQF) & Computational Finance Research  
> **Student / Researcher:** Hitesh Nanwani  
> **Key Domains:** Deep Learning for PDEs • Forward-Backward SDEs • Feynman-Kac Formula • High-Dimensional Derivative Pricing

---

## Executive Summary

Traditional numerical methods for partial differential equations (PDEs)—such as finite-difference lattices and finite-element grids—suffer from the severe **"curse of dimensionality"**, where computational complexity scales exponentially with the number of underlying state variables $\mathcal{O}(M^D)$. For multi-asset basket options or portfolio risk measures ($D \ge 5$), standard grids become intractable.

This project implements **Forward-Backward Stochastic Neural Networks (FBSNNs)** based on the pioneering deep learning frameworks of Raissi (2018) and Han, Jentzen, and E (2018). By reformulating parabolic PDEs into decoupled systems of Forward-Backward Stochastic Differential Equations (FBSDEs), deep neural networks approximate the high-dimensional value function $u(t, x)$ and its gradient (delta hedging strategy) $\nabla_x u(t, x)$ across continuous state spaces without spatial grids.

---

## Mathematical Formulation

### 1. Parabolic PDEs and the Non-Linear Feynman-Kac Theorem
Consider a general non-linear second-order parabolic PDE on $t \in [0, T]$ and $x \in \mathbb{R}^D$:

$$\frac{\partial u}{\partial t} + \frac{1}{2} \text{Tr}\left( \sigma(t, x) \sigma(t, x)^\top \nabla_x^2 u \right) + \mu(t, x)^\top \nabla_x u + f\left(t, x, u, \sigma(t, x)^\top \nabla_x u\right) = 0$$

with terminal condition $u(T, x) = g(x)$.

By the non-linear Feynman-Kac theorem, the solution satisfies the coupled FBSDE system:
- **Forward SDE (Underlying State Diffusion):**
  $$dX_t = \mu(t, X_t) dt + \sigma(t, X_t) dW_t, \qquad X_0 = x_0$$
- **Backward SDE (Option Value & Hedging Gradient):**
  $$dY_t = -f(t, X_t, Y_t, Z_t) dt + Z_t^\top dW_t, \qquad Y_T = g(X_T)$$
where $Y_t = u(t, X_t)$ is the derivative value and $Z_t = \sigma(t, X_t)^\top \nabla_x u(t, X_t)$ represents the continuous delta-hedging vector.

### 2. Deep Neural Network Parameterization
- The value function $u(t, x; \theta)$ is parameterized by a deep neural network with weights $\theta$.
- The gradient vector $Z_t$ is computed **analytically using PyTorch automatic differentiation** (`torch.autograd.grad`):
  $$Z_t = \sigma(t, X_t)^\top \nabla_x u(t, X_t; \theta)$$
- The initial condition $Y_0 = u(0, X_0)$ is treated as a trainable scalar parameter.

### 3. Loss Function & Euler-Maruyama Discretization
Simulating $M$ paths over $N$ discrete time intervals $\Delta t = T/N$:

$$X_{n+1} = X_n + \mu(t_n, X_n) \Delta t + \sigma(t_n, X_n) \Delta W_n$$

$$Y_{n+1} = Y_n - f(t_n, X_n, Y_n, Z_n) \Delta t + Z_n^\top \Delta W_n$$

The network minimizes the mean squared terminal matching error:

$$\mathcal{L}(\theta, Y_0) = \frac{1}{M} \sum_{m=1}^M \left| Y_N^{(m)} - g\left(X_N^{(m)}\right) \right|^2$$

---

## Structure & Artifacts

```text
07_deep_pde_fbsde_derivative_pricing/
├── README.md                      # Mathematical foundations & architecture review
├── fbsde_solver.py                # Standalone production PyTorch FBSDE solver engine
└── deep_fbsde_pde_solver.ipynb    # Interactive notebook with convergence graphs
```

---

## Verification & Execution

To run the standalone verification script:

```bash
uv run python 07_deep_pde_fbsde_derivative_pricing/fbsde_solver.py
```

The script trains the FBSNN, computes the analytical Black-Scholes benchmark price, and verifies numerical convergence.
