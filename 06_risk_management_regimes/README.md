# Quantitative Risk Management: DCC-GARCH, Graphical Models & Regime Detection

> **Program:** MSc in Financial Engineering (WorldQuant University) — MScFE 660 Risk Management & Econometrics  
> **Student / Quantitative Researcher:** Hitesh Nanwani  
> **Key Domains:** Volatility Clustering • DCC-GARCH • Bayesian Networks • Value-at-Risk (VaR) • Markov Regime Switching

---

## Executive Summary

This project implements an econometric and machine learning framework to model systemic financial risk, non-linear volatility clustering, and price contagion across major commodity energy markets. 

The research spans two core investigations:
1. **Systemic Risk & Transmission in Crude Oil Markets:** Based on UCL research (*Alvi, 2018: Application of Probabilistic Graphical Models in Forecasting Crude Oil Price*), modeling time-varying volatility cascades via **Dynamic Conditional Correlation GARCH (DCC-GARCH)** and **Bayesian Belief Networks (Directed Acyclic Graphs)**.
2. **Non-Linear Regime Shift Detection in Natural Gas (`UNG`):** Employing **Markov-Switching Autoregressive Models (MS-AR)** to detect latent transitions between calm (low-volatility) and crisis (high-volatility) market states in prompt-month commodity futures.

---

## Methodological Framework

### 1. Dynamic Conditional Correlation GARCH (DCC-GARCH)
To capture time-varying cross-asset correlation without the curse of dimensionality, DCC-GARCH decomposes the conditional covariance matrix:

$$\mathbf{H}_t = \mathbf{D}_t \mathbf{R}_t \mathbf{D}_t$$

where $\mathbf{D}_t = \text{diag}(\sigma_{1,t}, \dots, \sigma_{N,t})$ represents individual univariate GARCH(1,1) volatilities:

$$\sigma_{i,t}^2 = \omega_i + \alpha_i \epsilon_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2$$

The dynamic pseudo-correlation matrix $\mathbf{Q}_t$ evolves according to:

$$\mathbf{Q}_t = (1 - a - b)\bar{\mathbf{Q}} + a (\mathbf{z}_{t-1} \mathbf{z}_{t-1}^\top) + b \mathbf{Q}_{t-1}$$

$$\mathbf{R}_t = \text{diag}(\mathbf{Q}_t)^{-1/2} \mathbf{Q}_t \, \text{diag}(\mathbf{Q}_t)^{-1/2}$$

### 2. Markov Regime-Switching Models (Natural Gas `UNG`)
Commodity futures exhibit pronounced structural breaks during supply disruptions and weather extremes. The regime $S_t \in \{1, 2\}$ follows a discrete first-order Markov chain with transition probability matrix:

$$P = \begin{pmatrix} p_{11} & p_{12} \\ p_{21} & p_{22} \end{pmatrix}, \qquad p_{ij} = \mathbb{P}(S_t = j \mid S_{t-1} = i)$$

The return process switches conditional mean and volatility parameters across states:

$$r_t \mid S_t \sim \mathcal{N}(\mu_{S_t}, \sigma_{S_t}^2)$$
The model yields smoothed posterior probabilities $\mathbb{P}(S_t = \text{Turbulent} \mid \mathcal{F}_T)$ for dynamic risk hedging.

### 3. Non-Gaussian Tail Risk: VaR & Expected Shortfall (CVaR)
Computes empirical and parametric tail loss distributions under fat-tailed Student's $t$ and skewed distributions:

$$\text{VaR}_\alpha = \inf \{ l : \mathbb{P}(L > l) \le 1 - \alpha \}$$

$$\text{ES}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha]$$

### 4. Probabilistic Graphical Models (Bayesian DAGs)
Constructs Bayesian Networks capturing conditional dependency structures among macroeconomic indicators, refinery utilization rates, inventory drawdowns, and front-month contracts to map systemic contagion propagation.

---

## Structure & Artifacts

```text
06_risk_management_regimes/
├── README.md                            # Comprehensive econometric documentation
├── crude_oil_graphical_models_risk.ipynb # Bayesian networks & DCC-GARCH modeling (Crude Oil)
├── natural_gas_regime_detection.ipynb    # Markov regime-switching model on UNG Natural Gas
├── figs/                                # DCC correlation plots, cascade graphs & regime charts
│   ├── fig01_spectral_sri.png
│   ├── fig02_quantreg_betas.png
│   ├── fig03_regime_dcc.png
│   ├── fig04_var_irf.png
│   ├── fig05_bootstrap.png
│   ├── fig06_har_2020.png
│   ├── fig07_crisis_windows.png
│   ├── fig08_roc.png
│   └── fig09_cascade.html
└── tables/                              # GARCH outputs, Granger tests, and FEVD matrices
    ├── table00_data_audit.csv through table17_improvements.csv
```

---

## Technologies Used

- **Language:** Python 3.12+ / Jupyter Notebook
- **Econometrics & Time Series:** `arch`, `statsmodels` (`MarkovRegression`, `VAR`), `scipy`
- **Linear Algebra & Data:** `numpy`, `pandas`
- **Visualization:** `matplotlib`, `seaborn`, interactive HTML (`plotly`)
