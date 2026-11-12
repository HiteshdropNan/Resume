#!/usr/bin/env python
"""
Black-Litterman Bayesian Portfolio Allocation Engine.
Implements the Black-Litterman model (He & Litterman, 1999; Idzorek, 2005)
combining market equilibrium prior returns (CAPM reverse optimization)
with subjective/quantitative investor views to generate smooth portfolio tilts.
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco


class BlackLittermanModel:
    """
    Black-Litterman asset allocation framework.
    """
    def __init__(self, asset_names, cov_matrix, market_weights, risk_aversion=3.0, tau=0.05):
        self.assets = list(asset_names)
        self.N = len(asset_names)
        self.cov = np.array(cov_matrix, dtype=float)
        self.w_mkt = np.array(market_weights, dtype=float)
        self.w_mkt = self.w_mkt / np.sum(self.w_mkt)  # Normalize
        self.delta = risk_aversion  # Risk aversion parameter lambda
        self.tau = tau  # Scaling factor for uncertainty in prior
        
        # 1. Reverse optimization: Implied market equilibrium excess returns
        self.pi = self.delta * (self.cov @ self.w_mkt)

    def set_views(self, P, Q, Omega=None, confidences=None):
        """
        Sets investor views:
          P: (K x N) pick matrix of view asset weights
          Q: (K,) vector of expected excess returns for the views
          Omega: (K x K) diagonal covariance matrix of view uncertainties
          confidences: Optional list of confidence percentages (0 to 1) per view
        """
        self.P = np.array(P, dtype=float)
        self.Q = np.array(Q, dtype=float).reshape(-1, 1)
        K = self.P.shape[0]

        if Omega is None:
            # Default Idzorek / He & Litterman formulation: Omega = diag(P * (tau * Cov) * P^T)
            tau_cov_P = self.P @ (self.tau * self.cov) @ self.P.T
            self.Omega = np.diag(np.diag(tau_cov_P))
            
            # If explicit confidences provided, scale variance inversely
            if confidences is not None:
                for k, conf in enumerate(confidences):
                    if conf > 0 and conf < 1:
                        # Lower confidence = higher variance/uncertainty
                        scale = (1.0 - conf) / conf
                        self.Omega[k, k] *= scale
        else:
            self.Omega = np.array(Omega, dtype=float)

    def compute_posterior(self):
        """
        Computes Black-Litterman posterior combined return distribution:
          E[R] = [ (tau * Sigma)^-1 + P^T Omega^-1 P ]^-1 [ (tau * Sigma)^-1 Pi + P^T Omega^-1 Q ]
        """
        tau_cov = self.tau * self.cov
        inv_tau_cov = np.linalg.inv(tau_cov)
        inv_Omega = np.linalg.inv(self.Omega)

        # Posterior precision matrix M^-1
        M_inv = inv_tau_cov + (self.P.T @ inv_Omega @ self.P)
        M = np.linalg.inv(M_inv)

        # Posterior expected returns
        term1 = inv_tau_cov @ self.pi.reshape(-1, 1)
        term2 = self.P.T @ inv_Omega @ self.Q
        er_post = (M @ (term1 + term2)).flatten()

        # Posterior covariance matrix
        cov_post = self.cov + M

        # Optimal unconstrained weights: w* = (1 / delta) * cov_post^-1 * er_post
        w_unconstrained = np.linalg.inv(self.delta * cov_post) @ er_post
        w_unconstrained = w_unconstrained / np.sum(w_unconstrained)

        return {
            "implied_prior_returns": self.pi,
            "posterior_returns": er_post,
            "posterior_cov": cov_post,
            "unconstrained_weights": w_unconstrained,
            "tilt": w_unconstrained - self.w_mkt
        }

    def compute_constrained_weights(self, er_post, allow_short=False):
        """Computes optimal weights under no-shorting and leverage constraints."""
        N = self.N
        def obj(w):
            port_ret = w @ er_post
            port_var = w @ self.cov @ w
            return -(port_ret - 0.5 * self.delta * port_var)

        bounds = None if allow_short else [(0.0, 1.0) for _ in range(N)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        init_guess = np.ones(N) / N

        res = sco.minimize(obj, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
        return res.x


def main():
    print("=" * 75)
    print("Black-Litterman Bayesian Portfolio Allocation Engine")
    print("=" * 75)

    # Asset Universe
    assets = ["Apple (AAPL)", "Microsoft (MSFT)", "Amazon (AMZN)", "Alphabet (GOOGL)", "JPMorgan (JPM)", "ExxonMobil (XOM)"]
    w_mkt = [0.25, 0.23, 0.16, 0.15, 0.11, 0.10]
    
    # Stylized covariance matrix (annualized volatility ~20-30%, correlation ~0.4-0.6)
    vols = np.array([0.24, 0.22, 0.28, 0.26, 0.20, 0.22])
    corr = np.full((6, 6), 0.45)
    np.fill_diagonal(corr, 1.0)
    cov = np.diag(vols) @ corr @ np.diag(vols)

    bl = BlackLittermanModel(assets, cov, w_mkt, risk_aversion=3.0, tau=0.05)
    print(f"Number of Assets: {len(assets)} | Risk Aversion (Delta): {bl.delta} | Tau: {bl.tau}\n")

    print("1. Prior Equilibrium Excess Returns (Reverse Optimization from Market Cap):")
    for a, w, pi in zip(assets, w_mkt, bl.pi):
        print(f"  {a:20s} | Market Weight: {w*100:5.1f}% | Implied Prior Return: {pi*100:5.2f}%")

    print("-" * 75)
    print("2. Formulating Subjective / Quant Views:")
    # View 1: Apple will have an annualized excess return of 9.0% (Absolute view)
    # View 2: Microsoft will outperform Amazon by 3.5% (Relative view)
    P = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # View 1: AAPL
        [0.0, 1.0, -1.0, 0.0, 0.0, 0.0]   # View 2: MSFT vs AMZN
    ]
    Q = [0.090, 0.035]
    confidences = [0.70, 0.85]  # 70% confidence in AAPL, 85% confidence in MSFT > AMZN
    bl.set_views(P, Q, confidences=confidences)

    print("  View 1 (Absolute): AAPL excess return = 9.0% (Confidence: 70%)")
    print("  View 2 (Relative): MSFT will outperform AMZN by +3.5% (Confidence: 85%)")

    print("-" * 75)
    print("3. Posterior Return Distribution & Portfolio Tilts:")
    post = bl.compute_posterior()
    w_constrained = bl.compute_constrained_weights(post["posterior_returns"], allow_short=False)

    df_results = pd.DataFrame({
        "Asset": assets,
        "Market Weight": [f"{w*100:.1f}%" for w in w_mkt],
        "Implied Prior Ret": [f"{r*100:.2f}%" for r in post["implied_prior_returns"]],
        "BL Posterior Ret": [f"{r*100:.2f}%" for r in post["posterior_returns"]],
        "BL Constrained Weight": [f"{w*100:.1f}%" for w in w_constrained],
        "Net Tilt": [f"{(w - wm)*100:+.2f}%" for w, wm in zip(w_constrained, w_mkt)]
    })
    print(df_results.to_string(index=False))

    print("-" * 75)
    print("Key Takeaways:")
    print("  - Notice MSFT received an overweight tilt, while AMZN was tilted downward in accordance with View 2.")
    print("  - Unlike classical Markowitz MVO, Black-Litterman avoids extreme 0% or 100% corner solutions.")
    print("=" * 75)


if __name__ == "__main__":
    main()
