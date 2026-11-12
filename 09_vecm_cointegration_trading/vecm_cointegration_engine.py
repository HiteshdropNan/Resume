#!/usr/bin/env python
"""
Multivariate VECM & Johansen Cointegration Statistical Arbitrage Engine.
Implements the Johansen Maximum Likelihood procedure, Vector Error Correction
Models (VECM), and mean-reverting spread trading strategies across co-integrated equities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


class VECMCointegrationEngine:
    """
    Multivariate cointegration analysis and statistical arbitrage framework.
    """
    def __init__(self, data_path=None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "data", "prices_coint_auto.xlsx")
        self.data_path = data_path
        self.data = self._load_data()

    def _load_data(self):
        """Loads and cleans equity time series."""
        df = pd.read_excel(self.data_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        return df.dropna()

    def check_stationarity(self):
        """Tests individual series for I(1) unit roots via Augmented Dickey-Fuller (ADF)."""
        adf_results = {}
        for col in self.data.columns:
            stat, pval, _, _, crit, _ = adfuller(self.data[col].dropna())
            stat_d1, pval_d1, _, _, _, _ = adfuller(self.data[col].diff().dropna())
            adf_results[col] = {
                "Level ADF Stat": stat,
                "Level p-value": pval,
                "Diff ADF Stat": stat_d1,
                "Diff p-value": pval_d1,
                "Is I(1)": pval > 0.05 and pval_d1 < 0.05
            }
        return pd.DataFrame(adf_results).T

    def run_johansen(self, assets=None, det_order=0, k_ar_diff=1):
        """
        Executes Johansen cointegration rank test (Trace & Max-Eigenvalue tests).
        det_order: 0 (linear trend in data, constant in cointegrating relation)
        """
        if assets is None:
            assets = ["Ford", "General Motors"]
        sub_df = self.data[assets]
        
        # statsmodels coint_johansen
        jh = coint_johansen(sub_df, det_order=det_order, k_ar_diff=k_ar_diff)

        # Trace test table
        trace_table = pd.DataFrame({
            "Hypothesis": [f"r <= {i}" for i in range(len(assets))],
            "Trace Stat": jh.lr1,
            "90% Crit": jh.cvt[:, 0],
            "95% Crit": jh.cvt[:, 1],
            "99% Crit": jh.cvt[:, 2],
            "Cointegrated (95%)": jh.lr1 > jh.cvt[:, 1]
        })

        # Max eigenvalue table
        max_eig_table = pd.DataFrame({
            "Hypothesis": [f"r = {i}" for i in range(len(assets))],
            "Max Eig Stat": jh.lr2,
            "90% Crit": jh.cvm[:, 0],
            "95% Crit": jh.cvm[:, 1],
            "99% Crit": jh.cvm[:, 2],
            "Cointegrated (95%)": jh.lr2 > jh.cvm[:, 1]
        })

        # Cointegrating vector beta (first eigenvector normalized by first asset, real part)
        evec_real = np.real(jh.evec[:, 0])
        beta = evec_real / evec_real[0]

        return {
            "trace_table": trace_table,
            "max_eig_table": max_eig_table,
            "eigenvalues": np.real(jh.eig),
            "eigenvectors": np.real(jh.evec),
            "beta": beta,
            "assets": assets
        }

    def fit_vecm(self, assets=None, k_ar_diff=1, coint_rank=1):
        """Fits full Vector Error Correction Model (VECM)."""
        if assets is None:
            assets = ["Ford", "General Motors"]
        sub_df = self.data[assets]
        model = VECM(sub_df, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic="co")
        res = model.fit()
        return res

    def backtest_statistical_arbitrage(self, assets=None, entry_z=1.5, exit_z=0.2):
        """Backtests mean-reverting long-short spread portfolio."""
        jh_res = self.run_johansen(assets)
        assets = jh_res["assets"]
        beta = jh_res["beta"]

        sub_df = self.data[assets]
        spread = sub_df.values @ beta

        # Z-score of spread
        roll_mean = pd.Series(spread, index=sub_df.index).rolling(window=60).mean()
        roll_std = pd.Series(spread, index=sub_df.index).rolling(window=60).std()
        z_score = (pd.Series(spread, index=sub_df.index) - roll_mean) / roll_std

        # Position logic
        pos = pd.Series(0, index=sub_df.index)
        pos[z_score > entry_z] = -1.0  # Short spread
        pos[z_score < -entry_z] = 1.0  # Long spread
        pos[z_score.abs() < exit_z] = 0.0
        pos = pos.ffill().fillna(0)

        # Asset returns
        ret = sub_df.pct_change()
        strat_ret = pos.shift(1) * (ret[assets[0]] - beta[1] * ret[assets[1]])
        strat_ret = strat_ret.dropna()

        # Metrics
        ann_factor = 252
        total_ret = (1 + strat_ret).prod() - 1.0
        sharpe = np.sqrt(ann_factor) * strat_ret.mean() / (strat_ret.std() + 1e-8)
        cum = (1 + strat_ret).cumprod()
        dd = (cum.cummax() - cum) / cum.cummax()
        max_dd = dd.max()

        return {
            "spread": pd.Series(spread, index=sub_df.index),
            "z_score": z_score,
            "beta": beta,
            "strategy_returns": strat_ret,
            "total_return": total_ret,
            "annualized_sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": (strat_ret > 0).mean()
        }


def main():
    print("=" * 75)
    print("Multivariate VECM & Johansen Cointegration Trading Engine")
    print("=" * 75)

    engine = VECMCointegrationEngine()
    print(f"Loaded {len(engine.data)} trading days across assets: {list(engine.data.columns)}\n")

    print("1. Augmented Dickey-Fuller (ADF) Unit Root Tests:")
    adf_df = engine.check_stationarity()
    print(adf_df[["Level p-value", "Diff p-value", "Is I(1)"]].to_string())

    print("-" * 75)
    print("2. Johansen Cointegration Test (Ford vs. General Motors):")
    jh = engine.run_johansen(["Ford", "General Motors"])
    print("\nTrace Statistic Test:")
    print(jh["trace_table"].to_string(index=False))

    print("\nMax Eigenvalue Test:")
    print(jh["max_eig_table"].to_string(index=False))

    beta = jh["beta"]
    print(f"\nNormalized Cointegrating Vector (Beta):")
    print(f"  Spread = 1.000 * {jh['assets'][0]} + ({beta[1]:.4f}) * {jh['assets'][1]}")

    print("-" * 75)
    print("3. Vector Error Correction Model (VECM) Parameters:")
    vecm_res = engine.fit_vecm(["Ford", "General Motors"])
    print(f"  Speed of Adjustment (Alpha):\n{vecm_res.alpha}")

    print("-" * 75)
    print("4. Statistical Arbitrage Spread Backtest Performance:")
    bt = engine.backtest_statistical_arbitrage(["Ford", "General Motors"])
    print(f"  Total Return:        {bt['total_return']*100:.2f}%")
    print(f"  Annualized Sharpe:   {bt['annualized_sharpe']:.2f}")
    print(f"  Max Drawdown:        {bt['max_drawdown']*100:.2f}%")
    print(f"  Trade Win Rate:      {bt['win_rate']*100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
