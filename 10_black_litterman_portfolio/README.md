# Black-Litterman Bayesian Portfolio Allocation

> **Program:** Certificate in Quantitative Finance (CQF) — Asset Allocation & Portfolio Theory  
> **Student / Researcher:** Hitesh Nanwani  
> **Key Domains:** Bayesian Statistics • Black-Litterman Model • Reverse Optimization • Idzorek Confidence Scaling • Portfolio Tilting • Markowitz MVO

---

## Executive Summary

Classical Markowitz Mean-Variance Optimization (MVO) suffers notoriously from "error maximization" (Michaud, 1989): small estimation errors in expected returns produce radical, corner-solution allocations with extreme long and short positions that underperform out-of-sample and incur prohibitive transaction costs.

The **Black-Litterman Model** (Black & Litterman, 1991, 1992; He & Litterman, 1999) overcomes these deficiencies through a Bayesian shrinkage framework. Rather than forcing portfolio managers to forecast absolute expected returns for every asset in isolation, Black-Litterman begins with an observationally neutral prior derived via **reverse optimization** from market capitalization weights. Investors then express specific subjective or quantitative views with explicit confidence levels. The framework computes the posterior distribution of expected returns, tilting asset weights proportionally away from market equilibrium while preventing corner solutions.

---

## Methodological Framework

### 1. Reverse Optimization (Market Equilibrium Prior)
According to the Capital Asset Pricing Model (CAPM), if investors held mean-variance efficient portfolios, the aggregate market portfolio would clear all asset demands. The implied neutral equilibrium excess return vector $\boldsymbol{\Pi} \in \mathbb{R}^N$ is derived via reverse optimization:

$$\boldsymbol{\Pi} = \lambda \boldsymbol{\Sigma} \mathbf{w}_{\text{mkt}}$$

where:
- $\lambda = \frac{E[r_{\text{mkt}}] - r_f}{\sigma^2_{\text{mkt}}}$ is the aggregate market risk-aversion coefficient (typically $2.5 \le \lambda \le 4.0$).
- $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$ is the asset return covariance matrix.
- $\mathbf{w}_{\text{mkt}} \in \mathbb{R}^N$ is the vector of benchmark or market capitalization weights ($\sum_{i=1}^N w_{\text{mkt}, i} = 1$).

### 2. View Formulation and Error Structure
Investors express $K$ views (absolute or relative) across the $N$ assets:

$$\mathbf{P} \mathbf{r} = \mathbf{Q} + \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Omega})$$

where:
- $\mathbf{P} \in \mathbb{R}^{K \times N}$ is the pick matrix specifying the asset weights for each view. (For an absolute view on asset $i$, $P_{k, i} = 1$; for a relative view that asset $i$ outperforms asset $j$, $P_{k, i} = 1$ and $P_{k, j} = -1$).
- $\mathbf{Q} \in \mathbb{R}^K$ is the vector of expected returns for each view.
- $\boldsymbol{\Omega} \in \mathbb{R}^{K \times K}$ is the diagonal covariance matrix of view uncertainty. Following He and Litterman (1999) and Idzorek (2005):
  $$\Omega_{k, k} = \mathbf{p}_k (\tau \boldsymbol{\Sigma}) \mathbf{p}_k^\top \cdot \left(\frac{1 - c_k}{c_k}\right)$$
  where $c_k \in (0, 1)$ represents the user-specified confidence level in view $k$, and $\tau$ is the scalar scaling the uncertainty of the prior distribution (typically $0.025 \le \tau \le 0.05$).

### 3. Bayesian Posterior Updating
Under Gaussian conjugate priors, the posterior distribution of asset returns $\mathbf{R} \sim \mathcal{N}\left(\mathbb{E}[\mathbf{R}], \boldsymbol{\Sigma}_{\text{post}}\right)$ is given analytically by generalized least squares / Kalman filter equations:

$$\mathbb{E}[\mathbf{R}] = \left[ (\tau \boldsymbol{\Sigma})^{-1} + \mathbf{P}^\top \boldsymbol{\Omega}^{-1} \mathbf{P} \right]^{-1} \left[ (\tau \boldsymbol{\Sigma})^{-1} \boldsymbol{\Pi} + \mathbf{P}^\top \boldsymbol{\Omega}^{-1} \mathbf{Q} \right]$$

The posterior covariance matrix accounts for both return covariance and parameter uncertainty in the estimation:

$$\mathbf{M} = \left[ (\tau \boldsymbol{\Sigma})^{-1} + \mathbf{P}^\top \boldsymbol{\Omega}^{-1} \mathbf{P} \right]^{-1}$$
$$\boldsymbol{\Sigma}_{\text{post}} = \boldsymbol{\Sigma} + \mathbf{M}$$

### 4. Mean-Variance Optimization & Portfolio Tilts
With posterior returns $\mathbb{E}[\mathbf{R}]$ and covariance $\boldsymbol{\Sigma}_{\text{post}}$, optimal portfolio weights $\mathbf{w}^*$ are determined under quadratic utility:

$$\max_{\mathbf{w}} \quad \mathbf{w}^\top \mathbb{E}[\mathbf{R}] - \frac{\lambda}{2} \mathbf{w}^\top \boldsymbol{\Sigma}_{\text{post}} \mathbf{w} \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad w_i \ge 0$$

- When no views are expressed, the optimal weights exactly match the benchmark market weights: $\mathbf{w}^* = \mathbf{w}_{\text{mkt}}$.
- When views are added, weights are smoothly tilted in the direction of the views, scaled dynamically by the user's confidence level and cross-asset correlations.

---

## Structure & Artifacts

```text
10_black_litterman_portfolio/
├── README.md                          # Theoretical foundation & Bayesian derivation
├── black_litterman_engine.py          # Standalone Black-Litterman & Idzorek optimization engine
└── black_litterman_allocation.ipynb   # Interactive analysis, view sensitivity, and allocation tilts
```

---

## Verification & Execution

To run the standalone Black-Litterman optimization engine:

```bash
uv run python 10_black_litterman_portfolio/black_litterman_engine.py
```

### Benchmark Output

```text
===========================================================================
Black-Litterman Bayesian Portfolio Allocation Engine
===========================================================================
Number of Assets: 6 | Risk Aversion (Delta): 3.0 | Tau: 0.05

1. Prior Equilibrium Excess Returns (Reverse Optimization from Market Cap):
  Apple (AAPL)         | Market Weight:  25.0% | Implied Prior Return: 10.10%
  Microsoft (MSFT)     | Market Weight:  23.0% | Implied Prior Return:  8.92%
  Amazon (AMZN)        | Market Weight:  16.0% | Implied Prior Return: 11.08%
  Alphabet (GOOGL)     | Market Weight:  15.0% | Implied Prior Return: 10.04%
  JPMorgan (JPM)       | Market Weight:  11.0% | Implied Prior Return:  7.16%
  ExxonMobil (XOM)     | Market Weight:  10.0% | Implied Prior Return:  7.88%
---------------------------------------------------------------------------
2. Formulating Subjective / Quant Views:
  View 1 (Absolute): AAPL excess return = 9.0% (Confidence: 70%)
  View 2 (Relative): MSFT will outperform AMZN by +3.5% (Confidence: 85%)
---------------------------------------------------------------------------
3. Posterior Return Distribution & Portfolio Tilts:
           Asset Market Weight Implied Prior Ret BL Posterior Ret BL Constrained Weight Net Tilt
    Apple (AAPL)         25.0%            10.10%            9.20%                 21.4%   -3.59%
Microsoft (MSFT)         23.0%             8.92%           10.11%                 44.6%  +21.60%
   Amazon (AMZN)         16.0%            11.08%            7.45%                  0.0%  -16.00%
Alphabet (GOOGL)         15.0%            10.04%            9.34%                 14.1%   -0.89%
  JPMorgan (JPM)         11.0%             7.16%            6.63%                 10.7%   -0.31%
ExxonMobil (XOM)         10.0%             7.88%            7.29%                  9.2%   -0.80%
---------------------------------------------------------------------------
Key Takeaways:
  - MSFT received an overweight tilt (+21.60%), while AMZN was tilted downward (-16.00%) in accordance with View 2.
  - Unlike classical Markowitz MVO, Black-Litterman avoids unconstrained extreme corner solutions.
===========================================================================
```
