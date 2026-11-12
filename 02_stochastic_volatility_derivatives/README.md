# Stochastic Volatility Modeling & Numerical Derivative Pricing

> **Program:** MSc in Financial Engineering (WorldQuant University) — MScFE 620 Derivative Pricing  
> **Student / Modeler:** Hitesh Nanwani  
> **Key Domains:** Stochastic Calculus • Heston (1993) Model • Numerical Greeks • Dynamic Delta Hedging

---

## Executive Summary

This project implements advanced continuous-time and discrete-time derivative pricing engines. It explores the transition from constant-volatility Black-Scholes frameworks to **stochastic volatility dynamics** (Heston Model) and numerical lattice methods (Binomial & Trinomial trees) for European and American contingent claims.

Special focus is placed on **calibrating the volatility smile/surface**, measuring **numerical Greek sensitivities ($\Delta$, $\Gamma$, Vega)**, and testing **dynamic delta hedging performance** under transaction costs and discrete rebalancing intervals.

---

## Methodological Framework

### 1. Heston (1993) Stochastic Volatility Model
The asset price $S(t)$ and variance $\nu(t)$ satisfy the coupled system of Stochastic Differential Equations (SDEs):

$$dS(t) = r S(t) dt + \sqrt{\nu(t)} S(t) dW_1(t)$$

$$d\nu(t) = \kappa (\theta - \nu(t)) dt + \sigma_\nu \sqrt{\nu(t)} dW_2(t)$$

with correlated Brownian motions:
$$\mathbb{E}[dW_1(t) dW_2(t)] = \rho dt$$

- **Mean Reversion Rate ($\kappa$):** Speed at which variance reverts to its long-term mean $\theta$.
- **Vol-of-Vol ($\sigma_\nu$):** Controls the kurtosis (fat tails) of the asset return distribution.
- **Correlation ($\rho$):** Controls the skewness of the implied volatility surface (the leverage effect, typically negative for equities $\rho < 0$).
- **Feller Condition:** Verified to guarantee that variance stays strictly positive:
  $$2\kappa\theta > \sigma_\nu^2$$

### 2. Numerical Greeks via Central Finite Differences
Greeks are evaluated numerically across varying correlation regimes ($\rho = -0.30$ vs. $\rho = -0.70$):

$$\Delta = \frac{\partial V}{\partial S} \approx \frac{V(S + \Delta S) - V(S - \Delta S)}{2\Delta S}$$

$$\Gamma = \frac{\partial^2 V}{\partial S^2} \approx \frac{V(S + \Delta S) - 2V(S) + V(S - \Delta S)}{(\Delta S)^2}$$

### 3. Numerical Trees & Dynamic Delta Hedging
- **Binomial & Trinomial Lattices:** Backward induction pricing of American options with early exercise premiums.
- **Dynamic Hedging Simulation:** Rebalancing tracking portfolios to neutralize directional risk:
  $$\Pi_t = V_t - \Delta_t S_t$$
  Demonstrates how discrete rebalancing time steps ($\Delta t$) introduce tracking error (gamma risk).

---

## Structure & Artifacts

```text
02_stochastic_volatility_derivatives/
├── README.md                              # Mathematical background & implementation notes
├── heston_pricing_engine.py               # Standalone Heston SDE, Fourier pricer & hedging CLI engine
├── heston_stochastic_volatility.ipynb     # Heston SDE simulation, smile calibration & Greeks (Hitesh)
└── derivative_pricing_trees_hedging.ipynb # Binomial/Trinomial trees & dynamic hedging engine
```

---

## Verification & Execution

To run the standalone Heston stochastic volatility engine from the terminal:

```bash
uv run python 02_stochastic_volatility_derivatives/heston_pricing_engine.py
```

---

## Technologies Used

- **Language:** Python 3.12+ / Jupyter Notebook
- **Scientific Computing:** `numpy`, `scipy` (numerical integration, optimization), `pandas`
- **Visualization:** `matplotlib`, `seaborn`
