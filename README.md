# Hitesh Nanwani | Quantitative Finance & Financial Engineering Portfolio

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CQF](https://img.shields.io/badge/Certificate-CQF%20(Fitch%20Learning)-darkblue)](https://www.cqf.com/)
[![WQU MScFE](https://img.shields.io/badge/MSc-Financial%20Engineering%20(WQU)-green)](https://www.wqu.edu/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hitesh-nanwani)

Welcome to my quantitative finance and financial engineering research portfolio. This repository compiles graduate-level research projects, algorithmic trading engines, stochastic calculus models, deep learning architectures, and portfolio optimization systems developed during my **MSc in Financial Engineering at WorldQuant University**, the **Certificate in Quantitative Finance (CQF)** at Fitch Learning, and ongoing research as a **Quantitative Research Consultant at WorldQuant Brain** (where I have developed and deployed 400+ statistical arbitrage alphas).

---

## 🎯 Flagship Quantitative Projects Matrix

| Project | Domain | Key Mathematical Models & Frameworks | Primary Tech Stack | Status / Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **[01. Algorithmic Trading & Genetic Algorithms](./01_algorithmic_trading_cqf)** | Systematic Trading & Alpha Generation | Cointegration, Statistical Arbitrage, Pairs Trading, Evolutionary Parameter Optimization | Python, `deap`, Alpaca API, `statsmodels` | [Strategy Code & Engine](./01_algorithmic_trading_cqf) |
| **[02. Stochastic Volatility & Derivatives](./02_stochastic_volatility_derivatives)** | Derivatives & Stochastic Calculus | Heston (1993) SDEs, Fourier Inversion, Smile Calibration, Binomial/Trinomial Trees, Dynamic Delta Hedging | Python, `scipy`, `numpy`, Jupyter | [Interactive Notebooks](./02_stochastic_volatility_derivatives) |
| **[03. Deep Learning for Asset Allocation](./03_deep_learning_asset_allocation)** | Quant ML & Time Series | Recurrent Neural Networks (LSTM), Sequence-to-Sequence Return Forecasting, Dynamic SAA Optimization | PyTorch, Keras, `scikit-learn` | [LSTM Allocation Notebook](./03_deep_learning_asset_allocation) |
| **[04. Portfolio Optimization & MVO](./04_portfolio_optimization)** | Asset Management & Risk Parity | Markowitz (1952) Mean-Variance Optimization, Efficient Frontier, Tangency / GMV, Out-of-Sample Verification | Python, `scipy.optimize`, `pandas` | [MVO Jupyter Notebook](./04_portfolio_optimization) |
| **[05. Risk Management & DCC-GARCH](./05_risk_management_garch)** | Quantitative Risk & Econometrics | Probabilistic Graphical Models (Bayesian DAGs), DCC-GARCH Volatility, Value-at-Risk (VaR), Expected Shortfall | Python, `arch`, `statsmodels`, SciPy | [Risk Modeling Notebook](./05_risk_management_garch) |

---

## 🔬 Project Deep Dives

### 1. [Algorithmic Trading & Genetic Algorithm Strategy Optimization](./01_algorithmic_trading_cqf)
* **Program:** CQF Final Project | Fitch Learning
* **Focus:** End-to-end systematic trading system combining cointegrated statistical arbitrage (pairs trading) with trend-following strategies.
* **Core Innovation:** Employs genetic algorithm evolutionary computation (`deap`) across high-dimensional parameter spaces (lookback periods, entry/exit z-score triggers, trailing stops) to maximize risk-adjusted performance (Annualized Sharpe Ratio & APR).
* **Live Integration:** Fully operational client built on Alpaca Markets API.

$$\text{Spread}_t = P_t^{\text{Stock}} - \beta P_t^{\text{ETF}}, \qquad z_t = \frac{\text{Spread}_t - \mu}{\sigma}, \qquad \text{Sharpe} = \frac{\sqrt{252} \cdot (\mu_R - r_f)}{\sigma_R}$$

---

### 2. [Stochastic Volatility Modeling & Numerical Derivative Pricing](./02_stochastic_volatility_derivatives)
* **Program:** MScFE 620 Derivative Pricing | WorldQuant University
* **Focus:** Advanced continuous-time and discrete-time contingent claim valuation.
* **Core Models:**
  - **Heston (1993) SDE System:**
    $$dS(t) = r S(t) dt + \sqrt{\nu(t)} S(t) dW_1(t)$$
    $$d\nu(t) = \kappa (\theta - \nu(t)) dt + \sigma_\nu \sqrt{\nu(t)} dW_2(t), \quad \mathbb{E}[dW_1 dW_2] = \rho dt$$
  - Analyzed volatility smile skewness under varying correlation regimes ($\rho = -0.30$ vs. $\rho = -0.70$) and verified the Feller condition ($2\kappa\theta > \sigma_\nu^2$).
  - Evaluated numerical Greeks ($\Delta$, $\Gamma$) and executed dynamic delta hedging simulations with discrete rebalancing friction.

---

### 3. [Deep Learning in Finance: Multi-Asset LSTM Allocation](./03_deep_learning_asset_allocation)
* **Program:** MScFE 642 Deep Learning for Finance | WorldQuant University
* **Focus:** Multi-Asset Long Short-Term Memory (LSTM) recurrent neural network architectures for dynamic strategic asset allocation.
* **Key Results:** Captured non-linear temporal dependencies and cross-sector spillovers across equities, outperforming static historical mean and ARIMA/GARCH baselines.

---

### 4. [Modern Portfolio Theory & Mean-Variance Optimization](./04_portfolio_optimization)
* **Program:** MScFE 652 Portfolio Management | WorldQuant University
* **Focus:** Full empirical implementation of Harry Markowitz’s (1952) MVO framework across 10 US equities (`TSLA`, `WMT`, `BAC`, `GS`, `LLY`, `MRK`, `GOOG`, `META`, `AAPL`, `XOM`).
* **Protocol:** Derived the Efficient Frontier, Global Minimum Variance (GMV) Portfolio, and Tangency Portfolio in-sample (Sep 2023 – Sep 2025), followed by out-of-sample forward verification (Oct 2025 – Dec 2025) to quantify parameter estimation risk.

---

### 5. [Quantitative Risk Management: Graphical Models & DCC-GARCH](./05_risk_management_garch)
* **Program:** MScFE 660 Risk Management | WorldQuant University
* **Focus:** Modeling systemic market risk, contagion, and volatility clustering.
* **Methodology:** Implemented Dynamic Conditional Correlation GARCH (DCC-GARCH) and Bayesian Belief Networks based on UCL dissertation research on crude oil risk. Computed non-Gaussian tail risk via Value-at-Risk (VaR) and Expected Shortfall (CVaR).

---

## 🛠️ Technical & Quantitative Competencies

* **Quantitative Methods:** Statistical Arbitrage, Alpha Generation, Market Microstructure (Order Book Dynamics, Imbalance, Decay), Stochastic Calculus (Heston, Ito's Lemma), Monte Carlo Simulation, Value-at-Risk (VaR, CVaR), DCC-GARCH.
* **Programming Languages:** Python (NumPy, SciPy, pandas, statsmodels, PyTorch, scikit-learn, DEAP), Modern C++ (C++20, templates, multithreading, latency considerations), SQL / PostgreSQL.
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
