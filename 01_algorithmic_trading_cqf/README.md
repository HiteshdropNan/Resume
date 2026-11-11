# Algorithmic Trading & Genetic Algorithm Strategy Optimization

> **Program:** Certificate in Quantitative Finance (CQF) Final Project  
> **Author:** Hitesh Nanwani  
> **Key Domains:** Statistical Arbitrage • Cointegration • Genetic Algorithms • Live Order Execution

---

## Executive Summary

This project implements an end-to-end systematic trading platform combining **statistical arbitrage (mean-reversion pair trading)** and **trend-following strategies**, backed by **Genetic Algorithm (GA)** heuristic parameter optimization and live execution via the **Alpaca Markets API**.

Rather than relying on fixed or arbitrary parameters, the strategy dynamically searches high-dimensional parameter spaces (lookback windows, z-score entry/exit thresholds, stop-loss percentages) using evolutionary computation to maximize risk-adjusted performance (Annualized Sharpe Ratio & Annualized Percentage Return).

---

## Strategy Architectures

### 1. Statistical Arbitrage / Mean Reversion
- **Cointegration & Spread Modeling:** Computes rolling hedge ratios ($\beta$) between cointegrated ETF-equity pairs using ordinary least squares (OLS) regression:
  $$\text{Spread}_t = P_t^{\text{Stock}} - \beta P_t^{\text{ETF}}$$
- **Normalized Z-Score Tracking:** Standardizes the spread to identify mean-reverting deviations:
  $$z_t = \frac{\text{Spread}_t - \mu_{\text{lookback}}}{\sigma_{\text{lookback}}}$$
- **Execution Rules:** Long/short entry triggered when $|z_t| > z_{\text{entry}}$, with positions closed at $|z_t| \le z_{\text{exit}}$ or when trailing stop losses are violated.

### 2. Trend-Following Strategy
- Multi-timeframe moving average convergence/divergence with dynamic volatility filtering to capture sustained momentum regimes while minimizing whipsaw trades during sideways consolidation.

---

## Genetic Algorithm Optimization (`deap`)

Parameters are optimized using the **Distributed Evolutionary Algorithms in Python (`deap`)** library:
- **Chromosome Representation:** `[lookback_window, z_entry, z_exit, stop_loss_pct]`
- **Fitness Function:** Annualized Sharpe Ratio with transaction fee & slippage penalties:
  $$\text{Fitness} = \frac{\sqrt{252} \cdot \mathbb{E}[R - R_f]}{\sigma_R} - \text{Penalty}(\text{Drawdown}_{\text{max}} > 0.20)$$
- **Evolutionary Operators:** Two-point crossover (`cxUniform`), Gaussian mutation (`mutGaussian`), and tournament selection (`selTournament`).

---

## Mathematical Formulations

$$\text{Total Return} = \frac{P_{\text{end}} - P_{\text{start}}}{P_{\text{start}}}$$

$$\text{APR} = \left(1 + R_{\text{total}}\right)^{\frac{252}{n}} - 1$$

$$\text{Annualized Sharpe Ratio} = \frac{\sqrt{252} \cdot (\mu_R - r_f)}{\sigma_R}$$

$$\text{Max Drawdown} = \max_{t} \left(\frac{\max_{\tau \le t} P_\tau - P_t}{\max_{\tau \le t} P_\tau}\right)$$

---

## Repository Structure

```text
01_algorithmic_trading_cqf/
├── README.md                     # Project documentation & mathematical overview
├── cqf_algorithmic_trading.py    # Backtesting engine & genetic algorithm optimizer
├── mean_reversion_strategy.py    # Alpaca live trading implementation (Mean Reversion)
└── trend_strategy.py             # Alpaca live trading implementation (Trend Following)
```

---

## Technologies Used

- **Language:** Python 3.12+
- **Optimization:** DEAP (Distributed Evolutionary Algorithms in Python)
- **Broker / Market Data:** Alpaca Trade API (`alpaca-py`)
- **Econometrics & ML:** `scikit-learn` (Linear Regression), `statsmodels`
- **Data & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `yfinance`
