# Multivariate VECM & Johansen Cointegration Trading

> **Program:** Certificate in Quantitative Finance (CQF) — Time Series & Statistical Arbitrage  
> **Student / Researcher:** Hitesh Nanwani  
> **Key Domains:** Cointegration • Johansen Maximum Likelihood Test • Vector Error Correction Models (VECM) • Pairs Trading

---

## Executive Summary

While simple two-asset pairs trading relies on Engle-Granger two-step regression, bivariate OLS suffers from directional dependency bias (regressing $Y$ on $X$ produces a different hedge ratio than $X$ on $Y$) and fails when three or more assets share multiple stationary equilibrium relationships.

This project implements the **Johansen Maximum Likelihood procedure** and **Vector Error Correction Models (VECM)** to systematically identify and exploit multivariate cointegrating spaces. The framework models both the **long-run equilibrium vectors ($\boldsymbol{\beta}$)** and the **short-run speed-of-adjustment parameters ($\boldsymbol{\alpha}$)** governing mean reversion across related equities.

---

## Methodological Framework

### 1. Vector Autoregression (VAR) to VECM Representation
Let $\mathbf{Y}_t \in \mathbb{R}^k$ be a vector of non-stationary $I(1)$ financial time series. A $\text{VAR}(p)$ process:

$$\mathbf{Y}_t = \boldsymbol{\Phi}_1 \mathbf{Y}_{t-1} + \dots + \boldsymbol{\Phi}_p \mathbf{Y}_{t-p} + \mathbf{c} + \boldsymbol{\epsilon}_t$$

can be transformed into the **Vector Error Correction Model (VECM)**:

$$\Delta \mathbf{Y}_t = \boldsymbol{\Pi} \mathbf{Y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol{\Gamma}_i \Delta \mathbf{Y}_{t-i} + \mathbf{c} + \boldsymbol{\epsilon}_t$$

where:
$$\boldsymbol{\Pi} = \sum_{j=1}^p \boldsymbol{\Phi}_j - \mathbf{I}, \qquad \boldsymbol{\Gamma}_i = -\sum_{j=i+1}^p \boldsymbol{\Phi}_j$$

### 2. The Johansen Cointegration Rank Test
The matrix $\boldsymbol{\Pi}$ has dimension $k \times k$ and rank $r$:
- If $r = 0$: No cointegration exists (series are independent unit-root processes).
- If $r = k$: All series are stationary in levels $I(0)$.
- If $0 < r < k$: There exist $r$ stationary linear combinations (cointegrating vectors).

The matrix factors into:
$$\boldsymbol{\Pi} = \boldsymbol{\alpha} \boldsymbol{\beta}^\top$$
- $\boldsymbol{\beta} \in \mathbb{R}^{k \times r}$: Cointegrating vectors defining the stationary spreads $z_t = \boldsymbol{\beta}^\top \mathbf{Y}_{t-1}$.
- $\boldsymbol{\alpha} \in \mathbb{R}^{k \times r}$: Speed-of-adjustment coefficients measuring how rapidly deviations from equilibrium are eliminated.

### 3. Trace and Maximum Eigenvalue Test Statistics
Based on ordered squared canonical correlations (eigenvalues) $\hat{\lambda}_1 > \hat{\lambda}_2 > \dots > \hat{\lambda}_k$:

- **Trace Statistic:**
  $$\lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^k \ln(1 - \hat{\lambda}_i)$$
  Tests $H_0: \text{rank}(\boldsymbol{\Pi}) \le r$ against $H_1: \text{rank}(\boldsymbol{\Pi}) > r$.

- **Maximum Eigenvalue Statistic:**
  $$\lambda_{\text{max}}(r, r+1) = -T \ln(1 - \hat{\lambda}_{r+1})$$
  Tests $H_0: \text{rank}(\boldsymbol{\Pi}) = r$ against $H_1: \text{rank}(\boldsymbol{\Pi}) = r + 1$.

---

## Structure & Artifacts

```text
09_vecm_cointegration_trading/
├── README.md                          # Econometric foundations & test specification
├── vecm_cointegration_engine.py       # Standalone Johansen test & VECM arbitrage backtest
├── vecm_johansen_cointegration.ipynb  # Interactive notebook with spread dynamics
└── data/
    └── prices_coint_auto.xlsx         # Historical automotive & index dataset (Ford, GM, S&P 500)
```

---

## Verification & Execution

To run the standalone verification script:

```bash
uv run python 09_vecm_cointegration_trading/vecm_cointegration_engine.py
```
