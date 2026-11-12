"""
Markov Regime-Switching & Tail Risk Management Engine
=====================================================
MScFE 660: Risk Management & Econometrics | WorldQuant University
Lead Researcher: Hitesh Nanwani

Implements:
1. Hamilton (1989) Two-State Markov Regime-Switching model with switching variance.
2. Maximum likelihood estimation of state transition probability matrix P.
3. Unconditional regime duration calculation: E[Duration_i] = 1 / (1 - p_ii).
4. Parametric and Historical Value-at-Risk (VaR at 95% and 99% confidence).
5. Expected Shortfall (Conditional VaR / CVaR) conditioned on market volatility regimes.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


@dataclass
class RegimeConfig:
    n_days: int = 1500          # ~6 years of daily commodity trading data
    p00: float = 0.96           # Probability of staying in calm regime (persistence)
    p11: float = 0.88           # Probability of staying in turbulent regime
    vol_calm: float = 0.012     # Daily vol in calm regime (~19% annualized)
    vol_crisis: float = 0.038   # Daily vol in crisis regime (~60% annualized)
    seed: int = 42


class RegimeSwitchingEngine:
    """Quantitative risk engine modeling non-linear regime shifts and extreme tail risks."""

    def __init__(self, config: RegimeConfig = RegimeConfig()):
        self.cfg = config
        np.random.seed(self.cfg.seed)
        self.returns_df = self._simulate_commodity_regime_data()

    def _simulate_commodity_regime_data(self) -> pd.DataFrame:
        """Simulate realistic Natural Gas / Crude Oil return series governed by Markov state transitions."""
        n = self.cfg.n_days
        dates = pd.date_range("2018-01-01", periods=n, freq="B")

        states = np.zeros(n, dtype=int)
        returns = np.zeros(n)

        # Transition probabilities
        p00, p11 = self.cfg.p00, self.cfg.p11
        curr_state = 0

        for t in range(n):
            if curr_state == 0:
                curr_state = 0 if np.random.rand() < p00 else 1
            else:
                curr_state = 1 if np.random.rand() < p11 else 0

            states[t] = curr_state
            if curr_state == 0:
                returns[t] = np.random.normal(0.0003, self.cfg.vol_calm)
            else:
                returns[t] = np.random.normal(-0.0012, self.cfg.vol_crisis)

        return pd.DataFrame({"Return": returns, "True_State": states}, index=dates)

    def fit_markov_model(self) -> dict:
        """Estimate the two-state Markov switching model with regime-dependent mean and variance."""
        returns = self.returns_df["Return"]

        # Markov model with switching variance
        model = MarkovRegression(
            returns,
            k_regimes=2,
            trend="c",
            switching_variance=True
        )
        res = model.fit(disp=False)

        # Extract parameters
        mu_0, mu_1 = res.params["const[0]"], res.params["const[1]"]
        sigma2_0, sigma2_1 = res.params["sigma2[0]"], res.params["sigma2[1]"]

        # Ensure regime 0 is labeled as the low-volatility (calm) regime
        if sigma2_0 > sigma2_1:
            mu_0, mu_1 = mu_1, mu_0
            sigma2_0, sigma2_1 = sigma2_1, sigma2_0
            p00 = res.regime_transition[1, 1, 0]
            p11 = res.regime_transition[0, 0, 0]
        else:
            p00 = res.regime_transition[0, 0, 0]
            p11 = res.regime_transition[1, 1, 0]

        vol_0_ann = np.sqrt(sigma2_0) * np.sqrt(252)
        vol_1_ann = np.sqrt(sigma2_1) * np.sqrt(252)

        duration_0 = 1.0 / (1.0 - p00)
        duration_1 = 1.0 / (1.0 - p11)

        return {
            "mu_calm_daily": float(mu_0),
            "mu_crisis_daily": float(mu_1),
            "vol_calm_ann": float(vol_0_ann),
            "vol_crisis_ann": float(vol_1_ann),
            "p00": float(p00),
            "p11": float(p11),
            "p01": float(1.0 - p00),
            "p10": float(1.0 - p11),
            "expected_duration_calm_days": float(duration_0),
            "expected_duration_crisis_days": float(duration_1),
            "smoothed_probabilities": res.smoothed_marginal_probabilities
        }

    def compute_regime_tail_risk(self, alpha: float = 0.95) -> dict:
        """Compute Parametric & Empirical Value-at-Risk (VaR) and Expected Shortfall (CVaR) per regime."""
        df = self.returns_df
        calm_rets = df[df["True_State"] == 0]["Return"].values
        crisis_rets = df[df["True_State"] == 1]["Return"].values
        all_rets = df["Return"].values

        def eval_tail(rets):
            z_score = norm.ppf(alpha)
            mu = np.mean(rets)
            sigma = np.std(rets)

            # Parametric Gaussian VaR & CVaR
            param_var = -(mu - z_score * sigma)
            param_cvar = -(mu - sigma * (norm.pdf(z_score) / (1.0 - alpha)))

            # Empirical / Historical Simulation VaR & CVaR
            hist_var = -np.percentile(rets, (1.0 - alpha) * 100)
            tail_losses = -rets[-rets >= hist_var]
            hist_cvar = np.mean(tail_losses) if len(tail_losses) > 0 else hist_var

            return {
                "param_var": float(param_var),
                "param_cvar": float(param_cvar),
                "hist_var": float(hist_var),
                "hist_cvar": float(hist_cvar)
            }

        return {
            "unconditional": eval_tail(all_rets),
            "calm_regime": eval_tail(calm_rets),
            "crisis_regime": eval_tail(crisis_rets)
        }


def main():
    print("=" * 75)
    print("Markov Regime-Switching & Commodity Tail Risk Engine")
    print("=" * 75)
    config = RegimeConfig()
    engine = RegimeSwitchingEngine(config)

    print(f"Sample History: {config.n_days} trading days (Simulating Energy / Natural Gas dynamics)")
    print(f"Model: Two-State Hamilton (1989) Markov Switching with Dynamic Variance")
    print("-" * 75)

    # 1. Estimation
    print("1. Maximum Likelihood Estimation of Markov States:")
    fit = engine.fit_markov_model()
    print(f"  Calm Regime (State 0)   : Annualized Volatility = {fit['vol_calm_ann']*100:5.2f}% | Mean = {fit['mu_calm_daily']*252*100:+.2f}%")
    print(f"  Crisis Regime (State 1) : Annualized Volatility = {fit['vol_crisis_ann']*100:5.2f}% | Mean = {fit['mu_crisis_daily']*252*100:+.2f}%")
    print("-" * 75)

    # 2. Transition Dynamics & Persistence
    print("2. Transition Probability Matrix & Expected State Durations:")
    print("             To State 0 (Calm)   To State 1 (Crisis)")
    print(f"  From State 0 (Calm)    : {fit['p00']*100:6.2f}%             {fit['p01']*100:6.2f}%")
    print(f"  From State 1 (Crisis)  : {fit['p10']*100:6.2f}%             {fit['p11']*100:6.2f}%")
    print(f"\n  Expected Calm Regime Duration   : {fit['expected_duration_calm_days']:.1f} trading days (~{fit['expected_duration_calm_days']/21:.1f} months)")
    print(f"  Expected Crisis Regime Duration : {fit['expected_duration_crisis_days']:.1f} trading days (~{fit['expected_duration_crisis_days']/21:.1f} months)")
    print("-" * 75)

    # 3. Tail Risk & Expected Shortfall
    for conf_lvl in (0.95, 0.99):
        risk = engine.compute_regime_tail_risk(alpha=conf_lvl)
        print(f"3. Tail Risk Profile ({conf_lvl*100:.0f}% Confidence Level, 1-Day Horizon):")
        print(f"  {'Market State':<22} | {'Parametric VaR':<15} | {'Expected Shortfall (CVaR)':<24}")
        print("  " + "-" * 67)
        for state_name, label in [("unconditional", "Unconditional (All)"),
                                  ("calm_regime", "Calm Market State"),
                                  ("crisis_regime", "Turbulent Crisis State")]:
            m = risk[state_name]
            print(f"  {label:<22} | {m['param_var']*100:>13.2f}% | {m['param_cvar']*100:>22.2f}%")
        print("-" * 75)

    print("Key Takeaways:")
    print("  - Notice that during the crisis regime, 1-day 99% VaR explodes by >3x.")
    print("  - Static risk models that ignore regime shifts severely underestimate tail losses.")
    print("=" * 75)


if __name__ == "__main__":
    main()
