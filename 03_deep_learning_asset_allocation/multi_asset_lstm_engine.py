"""
Multi-Asset LSTM Return Forecasting & Dynamic Asset Allocation Engine
======================================================================
MScFE 642: Deep Learning for Finance | WorldQuant University
Lead Researcher: Hitesh Nanwani

Implements:
1. PyTorch Multi-Asset Recurrent Neural Network (LSTM) architecture.
2. Sliding-window sequence creation (L=20 trading days lookback across N=5 asset classes).
3. Supervised forward 1-day return forecasting with MSE loss & Adam optimization.
4. Dynamic Tactical Asset Allocation strategy driven by model return forecasts.
5. Out-of-sample backtest vs. benchmark 1/N Equal-Weight portfolio (Sharpe, Volatility, Max DD).
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class AllocationConfig:
    assets: tuple = ("SPY", "GLD", "TLT", "EEM", "VNQ")
    lookback_window: int = 20      # L = 20 trading days (~1 month)
    forecast_horizon: int = 1      # 1-day forward return
    train_split: float = 0.70      # 70% train, 30% test
    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.10
    learning_rate: float = 0.003
    batch_size: int = 32
    epochs: int = 25
    seed: int = 42


class MultiAssetLSTM(nn.Module):
    """Deep Recurrent Neural Network for joint multivariate asset return prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take final time step representation
        preds = self.fc(last_hidden)
        return preds


class DeepAssetAllocator:
    """End-to-end training and out-of-sample asset allocation simulator."""

    def __init__(self, config: AllocationConfig = AllocationConfig()):
        self.cfg = config
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.device = torch.device("cpu")
        self.model = MultiAssetLSTM(
            input_dim=len(self.cfg.assets),
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout
        ).to(self.device)

    def generate_synthetic_multi_asset_data(self, n_days: int = 1000) -> pd.DataFrame:
        """Generate realistic correlated multi-asset returns for SPY, GLD, TLT, EEM, VNQ."""
        k = len(self.cfg.assets)
        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

        # Correlation matrix representing cross-asset dynamics
        # SPY (Equities), GLD (Gold), TLT (Treasuries), EEM (Emerging Markets), VNQ (Real Estate)
        corr = np.array([
            [1.00,  0.05, -0.35,  0.75,  0.65],  # SPY
            [0.05,  1.00,  0.20,  0.15,  0.10],  # GLD
            [-0.35, 0.20,  1.00, -0.25, -0.15],  # TLT
            [0.75,  0.15, -0.25,  1.00,  0.55],  # EEM
            [0.65,  0.10, -0.15,  0.55,  1.00]   # VNQ
        ])
        vols = np.array([0.012, 0.009, 0.008, 0.015, 0.013])
        cov = np.outer(vols, vols) * corr

        # Mean returns (annualized ~ 5% to 10%)
        means = np.array([0.00035, 0.00020, 0.00015, 0.00030, 0.00025])

        raw_returns = np.random.multivariate_normal(means, cov, size=n_days)
        return pd.DataFrame(raw_returns, index=dates, columns=self.cfg.assets)

    def prepare_sliding_windows(self, returns_df: pd.DataFrame):
        """Transform return series into rolling 3D tensors: (N_samples, lookback, N_assets)."""
        data = returns_df.values
        T, K = data.shape
        L = self.cfg.lookback_window

        X_list, y_list = [], []
        for i in range(T - L):
            X_list.append(data[i : i + L])
            y_list.append(data[i + L])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        split_idx = int(len(X) * self.cfg.train_split)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        test_dates = returns_df.index[L + split_idx :]

        return X_train, y_train, X_test, y_test, test_dates

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> list:
        """Execute mini-batch training loop using Adam optimizer and MSE loss."""
        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()

        loss_history = []
        self.model.train()
        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            epoch_loss /= len(dataset)
            loss_history.append(epoch_loss)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [Epoch {epoch+1:2d}/{self.cfg.epochs}] Training MSE Loss: {epoch_loss:.6f}")

        return loss_history

    def backtest_out_of_sample(self, X_test: np.ndarray, y_test: np.ndarray, test_dates: pd.DatetimeIndex) -> dict:
        """Simulate dynamic Tactical Asset Allocation vs. Equal-Weight benchmark."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.from_numpy(X_test).to(self.device)).cpu().numpy()

        n_samples, n_assets = preds.shape

        # Dynamic allocation rule: Softmax temperature weighting on positive return forecasts
        # Shift predictions so minimum value is scaled
        scaled_preds = preds * 100.0  # Scale for temperature
        exp_preds = np.exp(np.clip(scaled_preds, -5.0, 5.0))
        dynamic_weights = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

        # Equal-weight benchmark (1/N)
        eq_weights = np.ones((n_samples, n_assets)) / n_assets

        # Daily strategy portfolio returns
        strat_returns = np.sum(dynamic_weights * y_test, axis=1)
        bm_returns = np.sum(eq_weights * y_test, axis=1)

        # Performance analytics
        def compute_metrics(rets):
            mean_ret = np.mean(rets) * 252
            vol = np.std(rets) * np.sqrt(252) + 1e-9
            sharpe = (mean_ret - 0.04) / vol  # Assuming 4% risk-free rate
            cum_ret = np.cumprod(1 + rets) - 1
            cum_series = np.cumprod(1 + rets)
            peak = np.maximum.accumulate(cum_series)
            max_dd = np.max((peak - cum_series) / peak)
            return {
                "annualized_return": float(mean_ret),
                "annualized_vol": float(vol),
                "sharpe_ratio": float(sharpe),
                "total_return": float(cum_ret[-1]),
                "max_drawdown": float(max_dd)
            }

        return {
            "dynamic_lstm": compute_metrics(strat_returns),
            "benchmark_equal_weight": compute_metrics(bm_returns),
            "test_sample_size": n_samples,
            "weights_mean": np.mean(dynamic_weights, axis=0)
        }


def main():
    print("=" * 75)
    print("Multi-Asset LSTM Return Forecasting & Dynamic Asset Allocation Engine")
    print("=" * 75)
    config = AllocationConfig()
    allocator = DeepAssetAllocator(config)

    print(f"Asset Universe: {list(config.assets)} | Lookback: {config.lookback_window} days")
    print(f"Architecture  : 2-Layer LSTM (Hidden Dim: {config.hidden_dim}) + Linear Projection Head")
    print("-" * 75)

    print("1. Loading & Preprocessing Time Series Data:")
    df_returns = allocator.generate_synthetic_multi_asset_data(n_days=1000)
    X_train, y_train, X_test, y_test, test_dates = allocator.prepare_sliding_windows(df_returns)
    print(f"  Training Samples: {len(X_train)} | Out-of-Sample Test Windows: {len(X_test)}")
    print("-" * 75)

    print("2. Training Deep LSTM Network (PyTorch):")
    allocator.train_model(X_train, y_train)
    print("-" * 75)

    print("3. Out-of-Sample Dynamic Asset Allocation Backtest:")
    results = allocator.backtest_out_of_sample(X_test, y_test, test_dates)

    strat = results["dynamic_lstm"]
    bm = results["benchmark_equal_weight"]

    print(f"  {'Metric':<28} | {'Dynamic LSTM Strategy':<22} | {'1/N Equal Weight':<20}")
    print("  " + "-" * 72)
    print(f"  {'Annualized Return':<28} | {strat['annualized_return']*100:>19.2f}% | {bm['annualized_return']*100:>17.2f}%")
    print(f"  {'Annualized Volatility':<28} | {strat['annualized_vol']*100:>19.2f}% | {bm['annualized_vol']*100:>17.2f}%")
    print(f"  {'Sharpe Ratio (Rf=4.0%)':<28} | {strat['sharpe_ratio']:>20.2f}  | {bm['sharpe_ratio']:>18.2f} ")
    print(f"  {'Cumulative Total Return':<28} | {strat['total_return']*100:>19.2f}% | {bm['total_return']*100:>17.2f}%")
    print(f"  {'Maximum Drawdown':<28} | {strat['max_drawdown']*100:>19.2f}% | {bm['max_drawdown']*100:>17.2f}%")
    print("-" * 75)

    print("4. Average Out-of-Sample Dynamic Portfolio Weights:")
    for asset, w in zip(config.assets, results["weights_mean"]):
        print(f"  {asset:<5} : {w*100:5.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
