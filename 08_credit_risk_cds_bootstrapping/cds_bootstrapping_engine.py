#!/usr/bin/env python
"""
Credit Risk & CDS Hazard Rate Bootstrapping Engine.
Implements the ISDA Standard Model for extracting default probabilities
and piecewise-constant hazard rate term structures from market CDS par spreads.
Includes Gaussian Copula simulation for correlated multi-name credit baskets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CDSBootstrapper:
    """
    Bootstraps survival probabilities S(t) and piecewise hazard rates lambda(t)
    from market Credit Default Swap (CDS) par spreads.
    """
    def __init__(self, recovery_rate=0.5):
        self.R = recovery_rate  # Recovery rate (typically 40% for senior unsecured, 50% for sovereign)
        self.loss_given_default = 1.0 - recovery_rate

    def bootstrap(self, maturities, spreads_bps, discount_factors):
        """
        Calculates implied survival probabilities S(t) iteratively:
          Premium Leg = Protection Leg
        
        Parameters:
          maturities: array-like of maturities in years [1, 2, 3, ...]
          spreads_bps: array-like of CDS par spreads in basis points [29, 39, ...]
          discount_factors: array-like of risk-free discount factors D(0, t)

        Returns:
          DataFrame containing bootstrapped survival probabilities,
          cumulative default probabilities, and piecewise hazard rates.
        """
        maturities = np.array(maturities, dtype=float)
        spreads = np.array(spreads_bps, dtype=float) * 1e-4  # Convert bps to decimal
        dfs = np.array(discount_factors, dtype=float)
        N = len(maturities)

        S = np.zeros(N + 1)
        S[0] = 1.0  # At t=0, survival probability is 100%

        dt = np.diff(np.insert(maturities, 0, 0.0))

        # Year 1 bootstrap
        S[1] = (self.loss_given_default * dfs[0]) / (
            self.loss_given_default * dfs[0] + spreads[0] * dt[0] * dfs[0]
        )

        # Iterative bootstrap for Year 2 to N
        for n in range(1, N):
            s_n = spreads[n]
            # Accumulate prior protection and premium legs
            accrued = 0.0
            for i in range(n):
                prot_i = self.loss_given_default * dfs[i] * (S[i] - S[i + 1])
                prem_i = s_n * dt[i] * dfs[i] * S[i + 1]
                accrued += (prot_i - prem_i)

            denom = (self.loss_given_default * dfs[n]) + (s_n * dt[n] * dfs[n])
            numer = (self.loss_given_default * dfs[n] * S[n]) + accrued
            S[n + 1] = numer / denom

        # Calculate piecewise constant hazard rates: lambda_i = -ln(S_i / S_{i-1}) / dt_i
        hazard_rates = np.zeros(N)
        for i in range(N):
            hazard_rates[i] = -np.log(S[i + 1] / S[i]) / dt[i]

        cum_default_probs = 1.0 - S[1:]

        results = pd.DataFrame({
            "Maturity (Years)": maturities,
            "Spread (bps)": spreads_bps,
            "Discount Factor": dfs,
            "Survival Probability": S[1:],
            "Cumulative Default Prob": cum_default_probs,
            "Piecewise Hazard Rate": hazard_rates,
            "Hazard Rate (bps)": hazard_rates * 10000
        })

        return results, S

    def price_cds(self, spread_bps, maturity_idx, S, dfs, dt=1.0):
        """Computes the mark-to-market NPV of a CDS position given survival curve."""
        s = spread_bps * 1e-4
        prem_leg = sum(s * dt * dfs[i] * S[i + 1] for i in range(maturity_idx))
        prot_leg = sum(self.loss_given_default * dfs[i] * (S[i] - S[i + 1]) for i in range(maturity_idx))
        return prot_leg - prem_leg

    def simulate_copula_basket_defaults(self, hazard_rates, maturities, corr_matrix, n_sims=50000):
        """
        Simulates default times for a correlated multi-name credit basket
        using a Gaussian Copula with Cholesky decomposition.
        """
        n_names = corr_matrix.shape[0]
        L = np.linalg.cholesky(corr_matrix)
        
        # Correlated standard normal variates
        Z = np.random.randn(n_sims, n_names)
        Z_corr = Z @ L.T

        # Uniform marginals via probability integral transform
        from scipy.stats import norm
        U = norm.cdf(Z_corr)

        # Invert survival curve to obtain default times tau: S(tau) = U
        # For piecewise flat hazard rates:
        default_times = np.zeros_like(U)
        cum_dt = np.array(maturities)
        
        # Simplified exponential inversion using average hazard rate
        avg_hazard = np.mean(hazard_rates)
        default_times = -np.log(U) / avg_hazard

        # 1-year and 5-year default correlation
        def_1y = (default_times <= 1.0).astype(float)
        def_5y = (default_times <= 5.0).astype(float)

        return {
            "default_times": default_times,
            "1y_default_rate": np.mean(def_1y, axis=0),
            "5y_default_rate": np.mean(def_5y, axis=0),
            "empirical_1y_corr": np.corrcoef(def_1y.T),
            "first_to_default_5y_prob": np.mean(np.min(default_times, axis=1) <= 5.0)
        }


def main():
    print("=" * 75)
    print("Credit Risk & CDS Hazard Rate Bootstrapping Engine")
    print("=" * 75)

    # Benchmark CQF Peña CDS Curve
    maturities = [1, 2, 3, 4, 5]
    spreads_bps = [29.0, 39.0, 46.0, 52.0, 57.0]
    dfs = [0.9803, 0.9514, 0.9159, 0.8756, 0.8328]
    R = 0.5  # 50% recovery rate

    bootstrapper = CDSBootstrapper(recovery_rate=R)
    results, S = bootstrapper.bootstrap(maturities, spreads_bps, dfs)

    print(f"Recovery Rate: {R * 100:.1f}%\n")
    print("Bootstrapped Term Structure:")
    print(results.to_string(index=False, justify="right"))

    print("-" * 75)
    print("Benchmark Validation vs. CQF Peña Reference Table:")
    ref_survival = [0.994233, 0.984505, 0.972636, 0.958824, 0.943693]
    for i, (calc, ref) in enumerate(zip(results["Survival Probability"], ref_survival), 1):
        err = abs(calc - ref)
        print(f"  Year {i}: Implied S(t) = {calc:.6f} | Reference = {ref:.6f} | Abs Diff = {err:.8f}")

    print("-" * 75)
    print("Correlated Multi-Name Credit Basket (Gaussian Copula Simulation):")
    n_names = 4
    rho = 0.40  # 40% asset correlation
    corr_matrix = np.full((n_names, n_names), rho)
    np.fill_diagonal(corr_matrix, 1.0)

    sim = bootstrapper.simulate_copula_basket_defaults(
        results["Piecewise Hazard Rate"].values, maturities, corr_matrix, n_sims=20000
    )

    print(f"  Basket Size: {n_names} Reference Entities | Asset Correlation: {rho*100:.0f}%")
    print(f"  Simulated 5-Year Marginal Default Rate: {np.mean(sim['5y_default_rate'])*100:.2f}%")
    print(f"  First-to-Default (FTD) 5-Year Probability: {sim['first_to_default_5y_prob']*100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
