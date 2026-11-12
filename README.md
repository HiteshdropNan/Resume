# Hitesh Nanwani | Quantitative Finance & Financial Engineering Portfolio

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CQF](https://img.shields.io/badge/Certificate-CQF%20(Fitch%20Learning)-darkblue)](https://www.cqf.com/)
[![WQU MScFE](https://img.shields.io/badge/MSc-Financial%20Engineering%20(WQU)-green)](https://www.wqu.edu/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hitesh-nanwani)

Welcome to my quantitative finance and financial engineering research portfolio. This repository compiles graduate-level research projects, algorithmic trading engines, stochastic calculus models, statistical machine learning pipelines, and portfolio optimization systems developed during my **MSc in Financial Engineering at WorldQuant University**, the **Certificate in Quantitative Finance (CQF)** at Fitch Learning, and ongoing research as a **Quantitative Research Consultant at WorldQuant Brain** (where I have developed and deployed 400+ statistical arbitrage alphas).

---

## 🎯 Flagship Quantitative Projects Matrix

| # | Pillar | Domain | Key Mathematical Models & Frameworks | Primary Tech Stack | Status / Artifacts |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **01** | **[Algorithmic Trading & Genetic Algorithms](./01_algorithmic_trading_cqf)** | Systematic Trading & Alpha Generation | Cointegration, Statistical Arbitrage, Pairs Trading, Evolutionary Parameter Optimization (`deap`) | Python, `deap`, Alpaca API, `statsmodels` | [Strategy Code & Engine](./01_algorithmic_trading_cqf) |
| **02** | **[Stochastic Volatility & Derivatives](./02_stochastic_volatility_derivatives)** | Derivatives & Stochastic Calculus | Heston (1993) SDEs, Fourier Inversion, Smile Calibration, Binomial/Trinomial Trees, Dynamic Delta Hedging | Python, `scipy`, `numpy`, Jupyter | [Interactive Notebooks](./02_stochastic_volatility_derivatives) |
| **03** | **[Deep Learning for Asset Allocation](./03_deep_learning_asset_allocation)** | Quant ML & Time Series | Recurrent Neural Networks (LSTM), Sequence-to-Sequence Return Forecasting, Dynamic SAA Optimization | PyTorch, Keras, `scikit-learn` | [LSTM Allocation Notebook](./03_deep_learning_asset_allocation) |
| **04** | **[Machine Learning & Factor Models](./04_machine_learning_factor_models)** | Statistical Learning & Factor Investing | LASSO ($L_1$) Sparse Feature Selection, PCA Eigenportfolios, k-Means Risk Clustering | Python, `scikit-learn`, `pandas`, `numpy` | [Factor Analysis Notebooks](./04_machine_learning_factor_models) |
| **05** | **[Portfolio Optimization & Kelly Sizing](./05_portfolio_optimization_and_kelly)** | Asset Management & Allocation | Markowitz MVO, Optimal Kelly Criterion, Random Matrix Theory (Marchenko-Pastur Denoising), HRP, Detoning | Python, `scipy.optimize`, `numpy`, RMT | [Kelly & MVO Notebooks](./05_portfolio_optimization_and_kelly) |
| **06** | **[Risk Management & Regime Shifts](./06_risk_management_regimes)** | Quantitative Risk & Econometrics | DCC-GARCH Volatility, Value-at-Risk (VaR), Expected Shortfall, Bayesian DAGs, Natural Gas Markov Switching | Python, `arch`, `statsmodels`, SciPy | [Risk & Regime Notebooks](./06_risk_management_regimes) |

---

## 🔬 Project Deep Dives

### 1. [Algorithmic Trading & Genetic Algorithm Strategy Optimization](./01_algorithmic_trading_cqf)
* **Program:** CQF Final Project | Fitch Learning
* **Focus:** End-to-end systematic trading platform combining cointegrated statistical arbitrage (pairs trading) with trend-following strategies.
* **Core Innovation:** Employs genetic algorithm evolutionary computation (`deap`) across high-dimensional parameter spaces (lookback periods, entry/exit z-score triggers, trailing stops) to maximize risk-adjusted performance (Annualized Sharpe Ratio & APR).
* **Live Execution:** Fully operational live trading execution client built on the Alpaca Markets API.

$$\text{Spread}_t = P_t^{\text{Stock}} - \beta P_t^{\text{ETF}}, \qquad z_t = \frac{\text{Spread}_t - \mu}{\sigma}, \qquad \text{Sharpe} = \frac{\sqrt{252} \cdot (\mu_R - r_f)}{\sigma_R}$$

---

### 2. [Stochastic Volatility Modeling & Numerical Derivative Pricing](./02_stochastic_volatility_derivatives)
* **Program:** MScFE 620 Derivative Pricing | WorldQuant University
* **Focus:** Continuous-time stochastic volatility and discrete-time lattice valuation.
* **Core Models:**
  - **Heston (1993) SDE System:**
    $$dS(t) = r S(t) dt + \sqrt{\nu(t)} S(t) dW_1(t)$$
    $$d\nu(t) = \kappa (\theta - \nu(t)) dt + \sigma_\nu \sqrt{\nu(t)} dW_2(t), \quad \mathbb{E}[dW_1 dW_2] = \rho dt$$
  - Analyzed volatility smile skewness under varying correlation regimes ($\rho = -0.30$ vs. $\rho = -0.70$) and verified the Feller condition ($2\kappa\theta > \sigma_\nu^2$).
  - Evaluated numerical Greeks ($\Delta$, $\Gamma$, Vega) and executed dynamic delta hedging simulations under discrete rebalancing friction.

---

### 3. [Deep Learning in Finance: Multi-Asset LSTM Allocation](./03_deep_learning_asset_allocation)
* **Program:** MScFE 642 Deep Learning for Finance | WorldQuant University
* **Focus:** Multi-Asset Long Short-Term Memory (LSTM) recurrent neural network architectures for dynamic strategic asset allocation.
* **Key Results:** Captured non-linear temporal dependencies and cross-sector spillovers across equities, outperforming static historical mean and ARIMA/GARCH econometric baselines.

---

### 4. [Statistical Machine Learning & Factor Investing](./04_machine_learning_factor_models)
* **Program:** MScFE 632 Machine Learning in Finance | WorldQuant University
* **Focus:** High-dimensional factor selection, dimensionality reduction, and unsupervised risk clustering.
* **Core Methodologies:**
  - **LASSO Regularization ($L_1$ Shrinkage):** Enforcing sparsity to isolate statistically robust predictors and eliminate collinear factor noise.
  - **PCA Eigenportfolios:** Decomposing multi-asset return covariance into orthogonal risk components ($\lambda_1$ market factor vs. sector rotation).
  - **k-Means Clustering:** Discovering empirical risk groupings directly from return dynamics rather than relying on static sector classifications.

---

### 5. [Portfolio Optimization, Kelly Criterion & RMT Covariance Denoising](./05_portfolio_optimization_and_kelly)
* **Program:** MScFE 652 Portfolio Management | WorldQuant University
* **Focus:** Advanced institutional portfolio construction across 10 US equities (`TSLA`, `WMT`, `BAC`, `GS`, `LLY`, `MRK`, `GOOG`, `META`, `AAPL`, `XOM`).
* **Cutting-Edge Quantitative Improvements:**
  - **Optimal Kelly Criterion:** Evaluated Full-Kelly, Half-Kelly (preferred institutional volatility reduction), and Double-Kelly leverage models.
  - **Random Matrix Theory (RMT) Denoising:** Filtered covariance eigenvalues bounded by the Marchenko-Pastur distribution $\lambda^\pm = \sigma^2(1 \pm \sqrt{N/T})^2$ to separate true economic signal from noise.
  - **Covariance Detoning & HRP:** Stripped out dominant market eigenvalues and implemented Hierarchical Risk Parity (López de Prado) tree-clustering to eliminate matrix inversion instability.

---

### 6. [Quantitative Risk Management: Graphical Models, DCC-GARCH & Regimes](./06_risk_management_regimes)
* **Program:** MScFE 660 Risk Management & Econometrics | WorldQuant University
* **Focus:** Modeling systemic market risk, non-linear volatility clustering, and price contagion across energy commodities.
* **Key Models:**
  - **DCC-GARCH & Bayesian Networks:** Implemented Dynamic Conditional Correlation GARCH and Bayesian Directed Acyclic Graphs (DAGs) on Crude Oil markets based on UCL dissertation research.
  - **Markov Regime-Switching Models (Natural Gas `UNG`):** Modeled non-linear transition dynamics between calm and turbulent volatility regimes in commodity futures.
  - **Non-Gaussian Tail Risk:** Computed Value-at-Risk (VaR) and Expected Shortfall (CVaR) under Student's $t$ and skewed empirical distributions.

---

## 🛠️ Technical & Quantitative Competencies

* **Quantitative Methods:** Statistical Arbitrage, Alpha Generation, Market Microstructure (Order Book Dynamics, Imbalance, Decay), Stochastic Calculus (Heston, Ito's Lemma), Random Matrix Theory (Marchenko-Pastur Denoising), Hierarchical Risk Parity (HRP), Kelly Criterion, Monte Carlo Simulation, Value-at-Risk (VaR, CVaR), DCC-GARCH.
* **Programming Languages:** Python (NumPy, SciPy, pandas, statsmodels, PyTorch, scikit-learn, DEAP, arch), Modern C++ (C++20, templates, multithreading, latency considerations), SQL / PostgreSQL.
* **Tools & Frameworks:** Jupyter, Docker, Git, Linux, Alpaca API, Backtrader.

---

## 🚀 Quickstart & Reproduction

To clone and explore the notebooks and strategies locally:

```bash
# 1. Clone the repository
git clone https://github.com/HiteshdropNan/Resume.git
cd Resume

# 2. Set up virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# 3. Launch JupyterLab / Notebook to inspect research
uv run jupyter lab
```

---

## 📬 Contact & Links

* **Author:** Hitesh Nanwani
* **LinkedIn:** [linkedin.com/in/hitesh-nanwani](https://www.linkedin.com/in/hitesh-nanwani)
* **Repository:** [github.com/HiteshdropNan/Resume](https://github.com/HiteshdropNan/Resume)
