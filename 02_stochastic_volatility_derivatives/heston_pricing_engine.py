"""
Heston (1993) Stochastic Volatility & Dynamic Delta Hedging Engine
==================================================================
MScFE 620: Derivative Pricing | WorldQuant University
Researcher: Hitesh Nanwani

Implements:
1. Continuous-time Heston SDE Euler-Maruyama discretization with full truncation.
2. Semi-analytical Fourier inversion pricer using characteristic functions.
3. Verification of the Feller condition (2*kappa*theta > sigma_v^2).
4. Numerical Greeks (Delta, Gamma, Vega).
5. Dynamic delta hedging simulation tracking discrete rebalancing error.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


@dataclass
class HestonParameters:
    S0: float = 80.0          # Initial spot price
    K: float = 80.0           # Strike price (ATM)
    r: float = 0.055          # Risk-free rate (5.5%)
    T: float = 1.0            # Time to maturity (1 year)
    v0: float = 0.04          # Initial variance (20% volatility)
    kappa: float = 1.5        # Mean reversion speed
    theta: float = 0.04       # Long-term variance
    sigma_v: float = 0.30     # Volatility of variance
    rho: float = -0.30        # Asset-variance correlation (volatility skew)


class HestonPricer:
    """Semi-analytical and Monte Carlo pricing engine for the Heston (1993) model."""

    def __init__(self, params: HestonParameters):
        self.p = params

    def check_feller_condition(self) -> dict:
        """Verify the Feller condition 2*kappa*theta > sigma_v^2 to ensure variance remains strictly positive."""
        lhs = 2 * self.p.kappa * self.p.theta
        rhs = self.p.sigma_v ** 2
        satisfied = lhs > rhs
        ratio = lhs / rhs
        return {
            "2_kappa_theta": lhs,
            "sigma_v_squared": rhs,
            "satisfied": satisfied,
            "feller_ratio": ratio
        }

    def _char_func(self, phi: complex, j: int) -> complex:
        """Characteristic function of log-price under Heston model (j=1, 2)."""
        p = self.p
        u = 0.5 if j == 1 else -0.5
        b = p.kappa - p.rho * p.sigma_v if j == 1 else p.kappa
        a = p.kappa * p.theta
        x0 = np.log(p.S0)

        # Auxiliary variables
        d = np.sqrt((p.rho * p.sigma_v * phi * 1j - b)**2 - (p.sigma_v**2) * (2 * u * phi * 1j - phi**2))
        g = (b - p.rho * p.sigma_v * phi * 1j - d) / (b - p.rho * p.sigma_v * phi * 1j + d)

        # Continuous complex logarithm representation
        C = (p.r * phi * 1j * p.T +
             (a / (p.sigma_v**2)) * ((b - p.rho * p.sigma_v * phi * 1j - d) * p.T -
                                     2 * np.log((1 - g * np.exp(-d * p.T)) / (1 - g))))
        D = ((b - p.rho * p.sigma_v * phi * 1j - d) / (p.sigma_v**2)) * (
            (1 - np.exp(-d * p.T)) / (1 - g * np.exp(-d * p.T))
        )
        return np.exp(C + D * p.v0 + 1j * phi * x0)

    def analytical_call_price(self) -> float:
        """Compute exact European call price via Fourier inversion of characteristic functions."""
        p = self.p
        k_log = np.log(p.K)

        def integrand(phi, j):
            cf = self._char_func(phi, j)
            val = (np.exp(-1j * phi * k_log) * cf / (1j * phi)).real
            return val

        # Numerical integration for P1 and P2
        p1_int, _ = quad(integrand, 0, 100, args=(1,), limit=100)
        p2_int, _ = quad(integrand, 0, 100, args=(2,), limit=100)

        P1 = 0.5 + (1 / np.pi) * p1_int
        P2 = 0.5 + (1 / np.pi) * p2_int

        call_price = p.S0 * P1 - p.K * np.exp(-p.r * p.T) * P2
        return float(call_price)

    def monte_carlo_simulation(self, n_paths: int = 25000, n_steps: int = 252, seed: int = 42) -> dict:
        """Euler-Maruyama Monte Carlo simulation with full reflection/truncation scheme."""
        p = self.p
        np.random.seed(seed)
        dt = p.T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Correlated Brownian motion increments
        Z1 = np.random.normal(0, 1, (n_paths, n_steps))
        Z_ind = np.random.normal(0, 1, (n_paths, n_steps))
        Z2 = p.rho * Z1 + np.sqrt(1 - p.rho**2) * Z_ind

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = p.S0
        v[:, 0] = p.v0

        for t in range(n_steps):
            v_curr = np.maximum(v[:, t], 0.0)  # Truncation
            v_sqrt = np.sqrt(v_curr)

            # Log-spot step (exact martingale)
            S[:, t + 1] = S[:, t] * np.exp((p.r - 0.5 * v_curr) * dt + v_sqrt * sqrt_dt * Z1[:, t])

            # Variance step (Euler-Maruyama with reflection)
            v[:, t + 1] = np.maximum(
                0.0,
                v[:, t] + p.kappa * (p.theta - v_curr) * dt + p.sigma_v * v_sqrt * sqrt_dt * Z2[:, t]
            )

        payoffs = np.maximum(S[:, -1] - p.K, 0.0)
        mc_price = float(np.exp(-p.r * p.T) * np.mean(payoffs))
        mc_std = float(np.std(payoffs) / np.sqrt(n_paths))

        return {
            "mc_price": mc_price,
            "mc_stderr": mc_std,
            "S_terminal": S[:, -1],
            "S_paths": S,
            "v_paths": v
        }

    def compute_numerical_greeks(self, h_S: float = 0.5, h_v: float = 0.005) -> dict:
        """Compute numerical Delta, Gamma, and Vega using central finite differences on the analytical pricer."""
        orig_S0 = self.p.S0
        orig_v0 = self.p.v0

        # Delta & Gamma via S0 shifts
        self.p.S0 = orig_S0 + h_S
        p_up = self.analytical_call_price()
        self.p.S0 = orig_S0 - h_S
        p_down = self.analytical_call_price()
        self.p.S0 = orig_S0
        p_base = self.analytical_call_price()

        delta = (p_up - p_down) / (2 * h_S)
        gamma = (p_up - 2 * p_base + p_down) / (h_S ** 2)

        # Vega via v0 shifts (dC / d_sigma = 2 * sqrt(v0) * dC / dv0)
        self.p.v0 = orig_v0 + h_v
        p_v_up = self.analytical_call_price()
        self.p.v0 = orig_v0 - h_v
        p_v_down = self.analytical_call_price()
        self.p.v0 = orig_v0

        vega_variance = (p_v_up - p_v_down) / (2 * h_v)
        vega_vol = vega_variance * 2 * np.sqrt(orig_v0) * 0.01  # Per 1% vol change

        return {
            "price": p_base,
            "delta": delta,
            "gamma": gamma,
            "vega_vol_1pct": vega_vol
        }

    def dynamic_delta_hedging_simulation(self, n_paths: int = 1000, n_steps: int = 52, seed: int = 42) -> dict:
        """Simulate dynamic delta-hedging error of a short call position over weekly rebalancing periods."""
        p = self.p
        np.random.seed(seed)
        dt = p.T / n_steps

        mc_res = self.monte_carlo_simulation(n_paths=n_paths, n_steps=n_steps, seed=seed)
        S_paths = mc_res["S_paths"]
        v_paths = mc_res["v_paths"]

        # Track PnL across paths
        hedging_errors = np.zeros(n_paths)

        # Initial call price
        initial_call = self.analytical_call_price()

        for i in range(n_paths):
            cash = initial_call
            stock_holding = 0.0

            for t in range(n_steps):
                t_remaining = max(p.T - t * dt, 1e-4)
                # Black-Scholes implied proxy delta using current local stochastic vol
                sigma_curr = np.sqrt(max(v_paths[i, t], 1e-4))
                d1 = (np.log(S_paths[i, t] / p.K) + (p.r + 0.5 * sigma_curr**2) * t_remaining) / (sigma_curr * np.sqrt(t_remaining))
                target_delta = norm.cdf(d1)

                # Rebalance shares
                share_diff = target_delta - stock_holding
                cash = cash * np.exp(p.r * dt) - share_diff * S_paths[i, t]
                stock_holding = target_delta

            # Terminal payoff obligations
            terminal_payoff = max(S_paths[i, -1] - p.K, 0.0)
            portfolio_val = stock_holding * S_paths[i, -1] + cash
            hedging_errors[i] = portfolio_val - terminal_payoff

        return {
            "mean_hedging_error": float(np.mean(hedging_errors)),
            "std_hedging_error": float(np.std(hedging_errors)),
            "relative_tracking_error": float(np.std(hedging_errors) / initial_call)
        }


def main():
    print("=" * 75)
    print("Heston (1993) Continuous-Time Stochastic Volatility Engine")
    print("=" * 75)

    params = HestonParameters(
        S0=80.0, K=80.0, r=0.055, T=1.0,
        v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.30, rho=-0.30
    )
    pricer = HestonPricer(params)

    # 1. Feller Condition Check
    feller = pricer.check_feller_condition()
    print("1. Feller Condition Verification:")
    print(f"  2 * kappa * theta  = {feller['2_kappa_theta']:.4f}")
    print(f"  sigma_v ^ 2        = {feller['sigma_v_squared']:.4f}")
    print(f"  Feller Ratio       = {feller['feller_ratio']:.2f}")
    print(f"  Condition Status   = {'SATISFIED (Variance process stays positive)' if feller['satisfied'] else 'VIOLATED'}")
    print("-" * 75)

    # 2. Analytical vs Monte Carlo Pricing
    call_exact = pricer.analytical_call_price()
    mc_res = pricer.monte_carlo_simulation(n_paths=25000, n_steps=100)
    call_mc = mc_res["mc_price"]
    mc_err = mc_res["mc_stderr"]

    print("2. Call Option Valuation (ATM K=80, T=1Y):")
    print(f"  Fourier Semi-Analytical Price : ${call_exact:.4f}")
    print(f"  Euler-Maruyama MC Price       : ${call_mc:.4f} +/- ${mc_err:.4f}")
    diff_pct = abs(call_mc - call_exact) / call_exact * 100
    print(f"  Absolute Discrepancy          : {diff_pct:.2f}%")
    print("-" * 75)

    # 3. Numerical Greeks
    greeks = pricer.compute_numerical_greeks()
    print("3. Numerical Greeks (Central Finite Differences):")
    print(f"  Delta (dC/dS)                 : {greeks['delta']:.4f}")
    print(f"  Gamma (d2C/dS2)               : {greeks['gamma']:.4f}")
    print(f"  Vega (per 1% Volatility Move) : ${greeks['vega_vol_1pct']:.4f}")
    print("-" * 75)

    # 4. Dynamic Delta Hedging Simulation
    hedge_res = pricer.dynamic_delta_hedging_simulation(n_paths=1000, n_steps=52)
    print("4. Dynamic Delta-Hedging Simulation (Weekly Rebalancing, 1,000 Paths):")
    print(f"  Mean Tracking PnL             : ${hedge_res['mean_hedging_error']:.4f}")
    print(f"  Tracking Error (Std Dev)      : ${hedge_res['std_hedging_error']:.4f}")
    print(f"  Relative Hedging Slippage     : {hedge_res['relative_tracking_error']*100:.2f}% of Option Premium")
    print("=" * 75)


if __name__ == "__main__":
    main()
