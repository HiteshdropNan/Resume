# Modern Portfolio Theory & Mean-Variance Optimization

> **Program:** MSc in Financial Engineering (WorldQuant University) — MScFE 652 Portfolio Management  
> **Student:** Hitesh Nanwani  
> **Key Domains:** Harry Markowitz MPT • Efficient Frontier • Sharpe Maximization • Out-of-Sample Verification

---

## Executive Summary

This project develops a quantitative portfolio construction engine based on Harry Markowitz’s (1952) **Mean-Variance Optimization (MVO)** framework. 

The strategy tests an investment universe of **10 blue-chip US equities** spanning technology, financial, healthcare, retail, and energy sectors (`TSLA`, `WMT`, `BAC`, `GS`, `LLY`, `MRK`, `GOOG`, `META`, `AAPL`, `XOM`). It generates the **Efficient Frontier**, derives the **Tangency (Max Sharpe) Portfolio** and **Global Minimum Variance (GMV) Portfolio**, and executes rigorous **out-of-sample forward testing**.

---

## Mathematical Formulation

### Mean-Variance Optimization (Markowitz, 1952)
For an asset universe with expected return vector $\boldsymbol{\mu} \in \mathbb{R}^N$ and covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$, the portfolio return $\mu_p$ and variance $\sigma_p^2$ for weight vector $\mathbf{w}$ are:

$$\mu_p = \mathbf{w}^\top \boldsymbol{\mu}, \qquad \sigma_p^2 = \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}$$

### 1. Global Minimum Variance (GMV) Portfolio
$$\min_{\mathbf{w}} \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^\top \mathbf{1} = 1, \quad w_i \ge 0$$

### 2. Tangency (Maximum Sharpe) Portfolio
$$\max_{\mathbf{w}} \frac{\mathbf{w}^\top \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}} \quad \text{subject to} \quad \mathbf{w}^\top \mathbf{1} = 1, \quad w_i \ge 0$$
where $r_f = 5.0\%$ p.a., reflecting benchmark interest rates during the sample period.

---

## In-Sample vs. Out-of-Sample Protocol

To guard against the classic "Markowitz curse" (error maximization in unconstrained historical covariance estimates):
- **In-Sample Period:** September 1, 2023 – September 30, 2025 (Parameter estimation & frontier derivation).
- **Out-of-Sample Period:** October 1, 2025 – December 31, 2025 (Forward validation of realized risk, return, and Sharpe decay).

---

## Notebooks

```text
04_portfolio_optimization/
├── README.md                          # Methodology, theoretical justification & findings
├── mean_variance_optimization.ipynb   # Complete step-by-step MVO & frontier generation
└── portfolio_management_gwp2.ipynb    # Extended asset allocation & factor portfolio models
```

---

## Technologies Used

- **Language:** Python 3.12+ / Jupyter Notebook
- **Optimization:** `scipy.optimize` (Sequential Least Squares Programming - SLSQP)
- **Data & Linear Algebra:** `numpy`, `pandas`, `yfinance`
- **Plotting:** `matplotlib`, `seaborn`
