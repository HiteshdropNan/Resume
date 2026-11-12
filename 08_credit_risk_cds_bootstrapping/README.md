# Credit Risk & CDS Hazard Rate Bootstrapping

> **Program:** Certificate in Quantitative Finance (CQF) — Credit Risk & Derivatives Modeling  
> **Student / Researcher:** Hitesh Nanwani  
> **Key Domains:** Credit Default Swaps (CDS) • Hazard Rate Bootstrapping • ISDA Standard Model • Gaussian Copula Multi-Name Default

---

## Executive Summary

Credit Default Swaps (CDS) represent the standard benchmark for single-name credit risk. Rather than assuming constant historical default probabilities, quantitative credit desks bootstrap **implied continuous survival probabilities $S(t)$** and **piecewise hazard rates $\lambda(t)$** directly from liquid market par spreads.

This project implements the **ISDA Standard Model** for credit bootstrapping, validated against the classical Peña CQF reference framework. It models:
1. Exact replication of the Premium Leg and Protection Leg equivalence.
2. Iterative bootstrapping of the piecewise constant hazard rate term structure.
3. Multi-name default correlation modeling using a **Gaussian Copula with Cholesky decomposition** for First-to-Default (FTD) and Collateralized Debt Obligation (CDO) tranche loss estimation.

---

## Mathematical Formulation

### 1. Par CDS Valuation: Premium vs. Protection Leg
For a CDS contract with maturity $T_N$, par spread $s_N$, and payment frequency $\Delta t$:

- **Premium Leg (Fee Payer):**
  $$\text{Premium Leg}(s_N) = s_N \sum_{i=1}^N \Delta t_i D(0, t_i) S(t_i)$$
- **Protection Leg (Contingent Payer upon default):**
  $$\text{Protection Leg} = (1 - R) \sum_{i=1}^N D(0, \bar{t}_i) [S(t_{i-1}) - S(t_i)]$$
where $R$ is the expected recovery rate ($R = 50\%$ in sovereign/IG corporate benchmarks), $D(0, t)$ is the risk-free discount factor, and $S(t) = \mathbb{P}(\tau > t)$ is the survival probability.

### 2. Bootstrapping Equation
Equating $\text{Premium Leg} = \text{Protection Leg}$ yields the recursive formula for $S(t_N)$ given prior survival points $\{S(t_0), \dots, S(t_{N-1})\}$:

$$S(t_N) = \frac{(1 - R) D(0, t_N) S(t_{N-1}) + \sum_{i=1}^{N-1} \left[ (1 - R) D(0, t_i) [S(t_{i-1}) - S(t_i)] - s_N \Delta t_i D(0, t_i) S(t_i) \right]}{(1 - R) D(0, t_N) + s_N \Delta t_N D(0, t_N)}$$

For Year 1 ($N = 1$):
$$S(t_1) = \frac{1 - R}{(1 - R) + s_1 \Delta t_1}$$

### 3. Piecewise Hazard Rates
The piecewise constant hazard rate $\lambda_i$ over interval $[t_{i-1}, t_i]$ satisfies:
$$S(t_i) = S(t_{i-1}) e^{-\lambda_i \Delta t_i} \implies \lambda_i = -\frac{\ln\left(\frac{S(t_i)}{S(t_{i-1})}\right)}{\Delta t_i}$$

### 4. Multi-Name Correlated Default Simulation (Gaussian Copula)
To simulate correlated default times for a basket of $K$ reference entities:
1. Decompose the asset correlation matrix $\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\top$ via Cholesky decomposition.
2. Generate correlated standard normals $\mathbf{Z} = \mathbf{L} \boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
3. Transform to uniform marginals via standard normal CDF: $U_k = \Phi(Z_k)$.
4. Invert each entity's survival curve to determine default time $\tau_k = S_k^{-1}(U_k)$.

---

## Benchmark Validation

Validated directly against the CQF Peña reference dataset:

| Maturity | Par Spread | Discount Factor | Implied Survival $S(t)$ | Reference | Absolute Diff | Piecewise $\lambda_i$ (bps) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Year** | 29 bps | 0.9803 | **0.994233** | 0.994233 | $4.5 \times 10^{-7}$ | 57.8 bps |
| **2 Year** | 39 bps | 0.9514 | **0.984505** | 0.984505 | $4.3 \times 10^{-7}$ | 98.3 bps |
| **3 Year** | 46 bps | 0.9159 | **0.972636** | 0.972636 | $3.4 \times 10^{-7}$ | 121.3 bps |
| **4 Year** | 52 bps | 0.8756 | **0.958824** | 0.958824 | $4.1 \times 10^{-7}$ | 143.0 bps |
| **5 Year** | 57 bps | 0.8328 | **0.943693** | 0.943693 | $3.3 \times 10^{-7}$ | 159.1 bps |

---

## Structure & Artifacts

```text
08_credit_risk_cds_bootstrapping/
├── README.md                           # Mathematical documentation & benchmark tables
├── cds_bootstrapping_engine.py         # Standalone calibration and copula simulation engine
└── cds_hazard_rate_bootstrapping.ipynb # Interactive notebook with term structures & basket defaults
```

---

## Verification & Execution

To run the standalone verification script:

```bash
uv run python 08_credit_risk_cds_bootstrapping/cds_bootstrapping_engine.py
```
