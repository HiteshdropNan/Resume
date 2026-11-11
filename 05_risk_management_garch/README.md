# Quantitative Risk Management: Probabilistic Graphical Models & DCC-GARCH

> **Program:** MSc in Financial Engineering (WorldQuant University) — MScFE 660 Risk Management  
> **Student:** Hitesh Nanwani  
> **Key Domains:** Volatility Modeling • DCC-GARCH • Probabilistic Graphical Models • Value-at-Risk (VaR) • Extreme Value Cascades

---

## Executive Summary

This project implements econometric and probabilistic frameworks to quantify systemic market risk, volatility clustering, and price contagion. Drawing on University College London (UCL) research (*Alvi, 2018: Application of Probabilistic Graphical Models in Forecasting Crude Oil Price*), this study models energy market risk transmission using **Dynamic Conditional Correlation GARCH (DCC-GARCH)** and **Bayesian Belief Networks**.

The framework measures tail-risk exposure under non-Gaussian, fat-tailed empirical distributions via **Value-at-Risk (VaR)** and **Expected Shortfall (CVaR)**.

---

## Methodological Framework

### 1. Univariate Volatility: GARCH(1,1)
Asset returns $r_t = \mu + \epsilon_t$, where innovations $\epsilon_t = \sigma_t z_t$ with standardized residual $z_t \sim \text{i.i.d.}(0, 1)$:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

where $\omega > 0, \alpha \ge 0, \beta \ge 0$, and covariance stationarity requires $\alpha + \beta < 1$.

### 2. Time-Varying Correlations: DCC-GARCH (Engle, 2002)
The conditional covariance matrix $\mathbf{H}_t = \mathbf{D}_t \mathbf{R}_t \mathbf{D}_t$, where $\mathbf{D}_t = \text{diag}(\sigma_{1,t}, \dots, \sigma_{N,t})$ and the dynamic correlation matrix $\mathbf{R}_t$ evolves according to:

$$\mathbf{Q}_t = (1 - a - b)\bar{\mathbf{Q}} + a (\mathbf{z}_{t-1} \mathbf{z}_{t-1}^\top) + b \mathbf{Q}_{t-1}$$

$$\mathbf{R}_t = \text{diag}(\mathbf{Q}_t)^{-1/2} \mathbf{Q}_t \, \text{diag}(\mathbf{Q}_t)^{-1/2}$$

where $\bar{\mathbf{Q}}$ is the unconditional covariance matrix of standardized residuals $\mathbf{z}_t$.

### 3. Tail-Risk Metrics: VaR & Expected Shortfall (CVaR)
For confidence level $\alpha \in (0, 1)$:

$$\text{VaR}_\alpha(L) = \inf \{ l \in \mathbb{R} : P(L > l) \le 1 - \alpha \}$$

$$\text{ES}_\alpha(L) = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha(L)]$$

### 4. Probabilistic Graphical Models (Bayesian Networks)
Models directed acyclic dependency graphs (DAGs) across macroeconomic indicators, supply-demand shocks, and prompt-month commodity futures to identify risk cascade paths and conditional probability distributions.

---

## Structure & Artifacts

```text
05_risk_management_garch/
├── README.md                            # Methodology & econometric review
├── crude_oil_graphical_models_risk.ipynb # Complete graphical models & volatility notebook
├── figs/                                # Visual regime shifts & DCC correlation plots
│   └── fig03_regime_dcc.png
└── tables/                              # Econometric summaries & GARCH parameter outputs
    ├── table01_spectral_summary.csv
    ├── table03_beta_AR_by_quantile.csv
    ├── table04_garch.csv
    └── table07_cascade_links.csv
```

---

## Technologies Used

- **Language:** Python 3.12+ / Jupyter Notebook
- **Econometrics & Time Series:** `arch`, `statsmodels`, `scipy`
- **Data & Linear Algebra:** `pandas`, `numpy`
- **Plotting:** `matplotlib`, `seaborn`
