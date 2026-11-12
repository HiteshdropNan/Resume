# Modern Portfolio Theory, Optimal Kelly Criterion & RMT Covariance Denoising

> **Program:** MSc in Financial Engineering (WorldQuant University) — MScFE 652 Portfolio Management  
> **Student / Quantitative Researcher:** Hitesh Nanwani  
> **Key Domains:** Markowitz MVO • Kelly Criterion Sizing • Random Matrix Theory (Marchenko-Pastur) • Hierarchical Risk Parity (HRP) • Covariance Detoning

---

## Executive Summary

This project develops an advanced institutional portfolio construction engine. It progresses from classical Harry Markowitz (1952) **Mean-Variance Optimization (MVO)** to continuous **Optimal Kelly Criterion** leverage modeling, and directly tackles the foundational problem of portfolio estimation error using cutting-edge quantitative techniques popularized by Marcos López de Prado: **Random Matrix Theory (RMT) Covariance Denoising, Covariance Detoning, and Hierarchical Risk Parity (HRP)**.

The framework is tested across a universe of 10 US equities (`TSLA`, `WMT`, `BAC`, `GS`, `LLY`, `MRK`, `GOOG`, `META`, `AAPL`, `XOM`) with strict **out-of-sample and TimeSeriesSplit cross-validation**.

---

## Methodological Framework

### 1. Classical Mean-Variance Optimization (Markowitz, 1952)
Computes the continuous **Efficient Frontier**, deriving the **Global Minimum Variance (GMV)** and **Tangency (Max Sharpe)** portfolios:

$$\max_{\mathbf{w}} \frac{\mathbf{w}^\top \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}} \quad \text{subject to} \quad \mathbf{w}^\top \mathbf{1} = 1, \quad w_i \ge 0$$

### 2. Optimal Kelly Criterion Allocation
The multi-asset Kelly Criterion maximizes the expected logarithmic growth rate of wealth:

$$g(\mathbf{w}) = \mathbb{E}[\ln(1 + \mathbf{w}^\top \mathbf{R})]$$

In continuous time with normal approximations, the unconstrained Kelly weight vector satisfies:

$$\mathbf{w}^* = \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu} - r_f \mathbf{1})$$

- **Risk Management Capping:** Evaluates realistic fractional Kelly allocations:
  - **Full-Kelly ($\mathbf{w}^*$):** Maximizes asymptotic capital growth but carries extreme volatility and large drawdown tails.
  - **Half-Kelly ($0.5 \mathbf{w}^*$):** Achieves $75\%$ of the growth rate with only $25\%$ of the portfolio variance—the preferred sizing of institutional prop desks.
  - **Double-Kelly ($2.0 \mathbf{w}^*$):** Leveraged aggressive benchmark.
- **Asset Weight Caps:** Enforces a $20\%$ maximum allocation cap per asset to prevent concentration risk.

---

## Advanced Covariance Cleaning & Machine Learning Improvements

Both MVO and Kelly optimization invert the empirical covariance matrix $\boldsymbol{\Sigma}^{-1}$. In sample dimensions where $N/T \approx \mathcal{O}(1)$, sample covariance matrices suffer severe estimation error ("noise multiplication"). This project implements three state-of-the-art solutions:

### A. Random Matrix Theory Denoising (Marchenko-Pastur Law)
Under the null hypothesis that empirical returns are purely random noise, the eigenvalues of $\boldsymbol{\Sigma}$ lie bounded between:

$$\lambda^{\pm} = \sigma^2 \left( 1 \pm \sqrt{\frac{N}{T}} \right)^2$$

- **Denoising Protocol:** Eigenvalues $\lambda \le \lambda^+$ are identified as noise and shrunk to their average trace, while eigenvalues $\lambda > \lambda^+$ are preserved as true economic signals.

### B. Covariance Detoning
The largest eigenvalue ($\lambda_1$) typically represents the broad market factor, dominating the covariance matrix. Detoning removes the market component:

$$\boldsymbol{\Sigma}_{\text{detoned}} = \boldsymbol{\Sigma} - \lambda_1 \mathbf{v}_1 \mathbf{v}_1^\top$$
allowing clustering algorithms to isolate pure sector and idiosyncratic risk relationships without market distortion.

### C. Hierarchical Risk Parity (HRP)
HRP (López de Prado) replaces matrix inversion altogether by combining graph theory and machine learning:
1. **Tree Clustering:** Calculates correlation distances $d_{i,j} = \sqrt{\frac{1}{2}(1 - \rho_{i,j})}$ and builds a hierarchical dendrogram.
2. **Quasi-Diagonalization:** Reorders the covariance matrix so similar assets sit adjacent.
3. **Recursive Bisection:** Allocates inverse-variance weights down the tree hierarchy, completely avoiding ill-conditioned matrix inversion.

---

## Structure & Artifacts

```text
05_portfolio_optimization_and_kelly/
├── README.md                                 # Complete theoretical & empirical documentation
├── kelly_rmt_optimizer.py                    # Standalone Kelly sizing, RMT denoising & HRP CLI engine
├── mean_variance_optimization.ipynb          # Step-by-step Markowitz MVO & Efficient Frontier
└── kelly_portfolio_rmt_denoising_hrp.ipynb   # Kelly portfolio, RMT Denoising, Detoning & HRP
```

---

## Verification & Execution

To run the standalone Kelly criterion, RMT denoising, and HRP optimization engine from the terminal:

```bash
uv run python 05_portfolio_optimization_and_kelly/kelly_rmt_optimizer.py
```

---

## Technologies Used

- **Language:** Python 3.12+ / Jupyter Notebook
- **Optimization:** `scipy.optimize`, `numpy` (linear algebra, eigendecomposition)
- **Machine Learning & Graph Theory:** `scipy.cluster.hierarchy`, `scikit-learn` (TimeSeriesSplit)
- **Data & Plotting:** `pandas`, `matplotlib`, `seaborn`, `yfinance`
