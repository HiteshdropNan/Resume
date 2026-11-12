# Deep Learning for Finance: Multi-Asset LSTM for Strategic Allocation

> **Program:** MSc in Financial Engineering (WorldQuant University) — MScFE 642 Deep Learning for Finance  
> **Student:** Hitesh Nanwani  
> **Key Domains:** Recurrent Neural Networks • LSTM • Strategic Asset Allocation • Time Series Forecasting

---

## Executive Summary

This project implements deep recurrent neural network architectures—specifically **Long Short-Term Memory (LSTM)** networks—to model complex, non-linear dependencies in multi-asset financial return sequences. The forecasted return vectors directly feed a dynamic **Strategic Asset Allocation (SAA)** optimizer.

Unlike static econometric baselines (e.g., ARIMA or rolling historical means), multi-asset LSTMs capture temporal memory, regime shifts, and cross-asset spillover effects across diverse equity sectors.

---

## Model Architecture

### LSTM Cell Mathematics
LSTMs mitigate vanishing and exploding gradients over long financial time horizons via three gating mechanisms:

$$\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad &&\text{(Forget Gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad &&\text{(Input Gate)} \\
\tilde{C}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad &&\text{(Candidate Cell State)} \\
C_t &= f_t * C_{t-1} + i_t * \tilde{C}_t \quad &&\text{(Updated Cell State)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad &&\text{(Output Gate)} \\
h_t &= o_t * \tanh(C_t) \quad &&\text{(Hidden State Vector)}
\end{aligned}$$

### End-to-End Pipeline
1. **Feature Engineering:** Computes stationary log returns, rolling volatility, volume momentum, and cross-correlation features.
2. **Sequential Windowing:** Constructs 3D tensors `(samples, timesteps, features)` with walk-forward expanding-window splits to prevent lookahead bias.
3. **Regularization:** Employs spatial Dropout ($p = 0.2$), $L_2$ kernel regularization, and Early Stopping based on validation loss.
4. **Dynamic Allocation Engine:** Maps predicted returns into dynamic portfolio weights subject to no-shorting and leverage constraints.

---

## Structure & Artifacts

```text
03_deep_learning_asset_allocation/
├── README.md                                 # Architectural overview & performance summary
├── multi_asset_lstm_engine.py                # Standalone PyTorch Multi-Asset LSTM & Dynamic SAA CLI engine
├── multi_asset_lstm_allocation.ipynb         # Multi-Asset LSTM training & strategic allocation
└── deep_learning_single_asset_baseline.ipynb # Single-asset deep learning benchmark (AAPL)
```

---

## Verification & Execution

To train the PyTorch Multi-Asset LSTM and run the out-of-sample backtest from the terminal:

```bash
uv run python 03_deep_learning_asset_allocation/multi_asset_lstm_engine.py
```

---

## Technologies Used

- **Frameworks:** PyTorch / TensorFlow, Keras
- **Machine Learning:** `scikit-learn` (MinMaxScaler, metrics)
- **Time Series & Data:** `pandas`, `numpy`, `yfinance`
- **Visualization:** `matplotlib`, `plotly`
