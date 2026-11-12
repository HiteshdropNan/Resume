"""
Optimal Kelly Criterion, RMT Covariance Denoising & HRP Engine
==============================================================
MScFE 652: Portfolio Management | WorldQuant University
Lead Researcher: Hitesh Nanwani

Implements:
1. Optimal Kelly Criterion: Full-Kelly vs. Half-Kelly (institutional drawdown reduction).
2. Marchenko-Pastur Random Matrix Theory (RMT) spectral covariance denoising.
3. Covariance Detoning (stripping dominant market mode).
4. Hierarchical Risk Parity (HRP / López de Prado 2016) recursive bisection allocation.
5. Out-of-sample risk-adjusted performance comparison.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


@dataclass
class PortfolioConfig:
    assets: tuple = ("AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "BAC", "XOM", "LLY", "MRK", "WMT")
    risk_free_rate: float = 0.04
    n_days: int = 1250          # 5 years of daily data
    seed: int = 42


class KellyRMTOptimizer:
    """Institutional portfolio construction engine integrating Kelly sizing, RMT filtering, and HRP."""

    def __init__(self, config: PortfolioConfig = PortfolioConfig()):
        self.cfg = config
        np.random.seed(self.cfg.seed)
        self.returns_df = self._generate_market_returns()

    def _generate_market_returns(self) -> pd.DataFrame:
        """Simulate realistic equity return series with factor structure and noise."""
        k = len(self.cfg.assets)
        t = self.cfg.n_days
        dates = pd.date_range("2019-01-01", periods=t, freq="B")

        # Market beta + sector factors + idiosyncratic noise
        market_factor = np.random.normal(0.0004, 0.011, t)
        tech_factor = np.random.normal(0.0002, 0.008, t)
        defensive_factor = np.random.normal(0.0001, 0.006, t)

        betas_mkt = np.array([1.2, 1.15, 1.1, 1.25, 1.05, 1.1, 0.85, 0.70, 0.65, 0.60])
        betas_tech = np.array([0.8, 0.85, 0.75, 0.70, -0.1, -0.1, -0.2, 0.0, 0.0, 0.1])
        betas_def = np.array([-0.2, -0.15, -0.1, -0.2, 0.1, 0.1, 0.2, 0.8, 0.85, 0.75])

        returns = np.zeros((t, k))
        for i in range(k):
            idio = np.random.normal(0, 0.012, t)
            returns[:, i] = (
                betas_mkt[i] * market_factor +
                betas_tech[i] * tech_factor +
                betas_def[i] * defensive_factor +
                idio
            )

        return pd.DataFrame(returns, index=dates, columns=self.cfg.assets)

    def denoise_covariance_rmt(self, cov_matrix: np.ndarray, t_obs: int, n_assets: int) -> np.ndarray:
        """Filter covariance eigenvalues using the Marchenko-Pastur theoretical bounds."""
        q = t_obs / n_assets
        # Correlation matrix decomposition
        vols = np.sqrt(np.diag(cov_matrix))
        corr = cov_matrix / np.outer(vols, vols)

        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Marchenko-Pastur theoretical maximum eigenvalue for white noise
        sigma_sq = 1.0  # Normalized variance of standardized returns
        lambda_plus = sigma_sq * (1.0 + np.sqrt(1.0 / q)) ** 2

        # Replace eigenvalues within noise band [0, lambda_plus] with their average
        noise_mask = eigenvalues <= lambda_plus
        if np.any(noise_mask):
            mean_noise = np.mean(eigenvalues[noise_mask])
            eigenvalues_denoised = eigenvalues.copy()
            eigenvalues_denoised[noise_mask] = mean_noise
        else:
            eigenvalues_denoised = eigenvalues.copy()

        # Reconstruct correlation matrix and rescale diagonal to 1.0
        corr_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T
        diag_sqrt = np.sqrt(np.diag(corr_denoised))
        corr_denoised = corr_denoised / np.outer(diag_sqrt, diag_sqrt)

        # Reconstruct covariance matrix
        cov_denoised = corr_denoised * np.outer(vols, vols)
        return cov_denoised

    def detone_covariance(self, cov_matrix: np.ndarray, n_market_modes: int = 1) -> np.ndarray:
        """Strip dominant market eigenvalue(s) from covariance to isolate idiosyncratic alpha risks."""
        vols = np.sqrt(np.diag(cov_matrix))
        corr = cov_matrix / np.outer(vols, vols)

        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Zero out the dominant market mode(s)
        eigenvalues_detoned = eigenvalues.copy()
        eigenvalues_detoned[:n_market_modes] = 0.0

        corr_detoned = eigenvectors @ np.diag(eigenvalues_detoned) @ eigenvectors.T
        diag_sqrt = np.sqrt(np.maximum(np.diag(corr_detoned), 1e-8))
        corr_detoned = corr_detoned / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(corr_detoned, 1.0)

        cov_detoned = corr_detoned * np.outer(vols, vols)
        return cov_detoned

    def compute_kelly_weights(self, mu: np.ndarray, cov: np.ndarray, fraction: float = 1.0) -> np.ndarray:
        """Compute Optimal Kelly sizing weights w* = (1 / gamma) * inv(Sigma) * (mu - rf)."""
        k = len(mu)
        excess_mu = mu - (self.cfg.risk_free_rate / 252)
        inv_cov = np.linalg.pinv(cov)
        unconstrained_w = fraction * (inv_cov @ excess_mu)

        # Long-only budget-constrained quadratic formulation
        def obj(w):
            return 0.5 * (w @ cov @ w) - (1.0 / fraction) * (w @ excess_mu)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 0.40) for _ in range(k)]
        w0 = np.ones(k) / k

        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        return res.x if res.success else w0

    def compute_hrp_weights(self, cov: np.ndarray) -> np.ndarray:
        """Compute Hierarchical Risk Parity (HRP) weights via tree clustering & recursive bisection."""
        vols = np.sqrt(np.diag(cov))
        corr = cov / np.outer(vols, vols)
        dist = np.sqrt(0.5 * (1.0 - np.clip(corr, -1.0, 1.0)))

        # Hierarchical tree clustering
        dist_condensed = squareform(dist, checks=False)
        link = linkage(dist_condensed, method="single")

        # Quasi-diagonalization order
        def get_quasi_diag(link_mat):
            link_mat = link_mat.astype(int)
            sort_ix = pd.Series([link_mat[-1, 0], link_mat[-1, 1]])
            num_items = link_mat[-1, 3]
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= num_items]
                i = df0.index
                j = df0.values - num_items
                sort_ix[i] = link_mat[j, 0]
                df0 = pd.Series(link_mat[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df0]).sort_index()
                sort_ix.index = range(sort_ix.shape[0])
            return sort_ix.tolist()

        sorted_order = get_quasi_diag(link)
        sorted_cov = cov[np.ix_(sorted_order, sorted_order)]

        # Recursive bisection
        weights = pd.Series(1.0, index=sorted_order)
        clusters = [sorted_order]
        while len(clusters) > 0:
            clusters = [c[start:end] for c in clusters for start, end in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
            for i in range(0, len(clusters), 2):
                c0 = clusters[i]
                c1 = clusters[i + 1]
                cov_0 = cov[np.ix_(c0, c0)]
                cov_1 = cov[np.ix_(c1, c1)]
                # Inverse variance cluster weights
                w0_inv = 1.0 / np.diag(cov_0)
                w0_norm = w0_inv / np.sum(w0_inv)
                var_0 = w0_norm @ cov_0 @ w0_norm

                w1_inv = 1.0 / np.diag(cov_1)
                w1_norm = w1_inv / np.sum(w1_inv)
                var_1 = w1_norm @ cov_1 @ w1_norm

                alpha = 1.0 - var_0 / (var_0 + var_1)
                weights[c0] *= alpha
                weights[c1] *= (1.0 - alpha)

        return weights.sort_index().values

    def run_benchmark_comparison(self) -> dict:
        """Run and compare Markowitz MVO, Full-Kelly, Half-Kelly, and HRP under raw vs RMT denoised covariance."""
        returns = self.returns_df.values
        t_obs, n_assets = returns.shape
        mu = np.mean(returns, axis=0) * 252
        cov_raw = np.cov(returns, rowvar=False) * 252

        # 1. Denoised Covariance via Marchenko-Pastur
        cov_denoised = self.denoise_covariance_rmt(cov_raw, t_obs=t_obs, n_assets=n_assets)

        # 2. Portfolio Allocations
        w_eq = np.ones(n_assets) / n_assets
        w_full_raw = self.compute_kelly_weights(mu, cov_raw, fraction=1.0)
        w_half_raw = self.compute_kelly_weights(mu, cov_raw, fraction=2.0)
        w_half_rmt = self.compute_kelly_weights(mu, cov_denoised, fraction=2.0)
        w_hrp = self.compute_hrp_weights(cov_raw)

        strategies = {
            "Equal Weight (1/N)": w_eq,
            "Full Kelly (Raw Cov)": w_full_raw,
            "Half Kelly (Raw Cov)": w_half_raw,
            "Half Kelly (RMT Denoised)": w_half_rmt,
            "Hierarchical Risk Parity": w_hrp
        }

        analytics = {}
        for name, w in strategies.items():
            port_ret = np.sum(w * mu)
            port_vol = np.sqrt(w @ cov_raw @ w)
            sharpe = (port_ret - self.cfg.risk_free_rate) / port_vol
            analytics[name] = {
                "annual_return": float(port_ret),
                "annual_vol": float(port_vol),
                "sharpe_ratio": float(sharpe),
                "weights": w
            }

        return analytics


def main():
    print("=" * 80)
    print("Optimal Kelly Criterion, RMT Covariance Denoising & HRP Engine")
    print("=" * 80)
    config = PortfolioConfig()
    engine = KellyRMTOptimizer(config)

    print(f"Asset Universe: {list(config.assets)} (N={len(config.assets)})")
    print(f"Sample History: {config.n_days} trading days | Risk-Free Rate: {config.risk_free_rate*100:.1f}%")
    print("-" * 80)

    # 1. RMT Eigenvalue Analysis
    t_obs, n_assets = engine.returns_df.shape
    q_ratio = t_obs / n_assets
    lambda_plus = (1.0 + np.sqrt(1.0 / q_ratio)) ** 2
    print("1. Marchenko-Pastur Random Matrix Theory (RMT) Spectrum:")
    print(f"  Sample Observations (T)         : {t_obs}")
    print(f"  Matrix Aspect Ratio (Q = T/N)   : {q_ratio:.1f}")
    print(f"  Theoretical Upper Bound (λ+)    : {lambda_plus:.4f}")
    print("  Eigenvalues above λ+ represent genuine economic factor signals.")
    print("-" * 80)

    # 2. Strategy Comparison
    print("2. Portfolio Construction & Risk-Adjusted Analytics:")
    analytics = engine.run_benchmark_comparison()

    print(f"  {'Strategy':<26} | {'Ann. Return':<12} | {'Ann. Vol':<10} | {'Sharpe':<8}")
    print("  " + "-" * 62)
    for name, m in analytics.items():
        print(f"  {name:<26} | {m['annual_return']*100:>10.2f}% | {m['annual_vol']*100:>8.2f}% | {m['sharpe_ratio']:>6.2f}")
    print("-" * 80)

    # 3. Allocations Breakdown
    print("3. Portfolio Allocations Breakdown (% Weights):")
    df_weights = pd.DataFrame(
        {name: m["weights"] * 100 for name, m in analytics.items()},
        index=config.assets
    )
    print(df_weights.round(1).to_string())
    print("=" * 80)


if __name__ == "__main__":
    main()
