"""
Machine Learning Factor Investing & Statistical Learning Engine
================================================================
MScFE 632: Machine Learning in Finance | WorldQuant University
Lead Researcher: Hitesh Nanwani

Implements:
1. LASSO (L1) Regularization for sparse feature selection across high-dimensional lagged asset returns.
2. Principal Component Analysis (PCA) Eigenportfolio decomposition of market covariance.
3. k-Means Unsupervised Clustering on empirical correlation distance metric d = sqrt(0.5*(1 - rho)).
4. Full out-of-sample statistical validation using historical daily return dataset.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score


class FactorLearningEngine:
    """Quantitative machine learning engine for sparse factor selection and dimensionality reduction."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        self.df_returns = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df = df.set_index(first_col)
        df = df.dropna()
        return df

    def run_lasso_feature_selection(self, target_asset: str = "AAPL", n_lags: int = 5, alpha: float = 0.0005) -> dict:
        """Construct lagged cross-asset predictors and apply L1 LASSO regularization to isolate sparse drivers."""
        df = self.df_returns.copy()
        assets = df.columns.tolist()

        # Build lagged feature matrix
        feature_dict = {}
        for lag in range(1, n_lags + 1):
            for col in assets:
                feature_dict[f"{col}_lag{lag}"] = df[col].shift(lag)

        X = pd.DataFrame(feature_dict, index=df.index).dropna()
        y = df[target_asset].loc[X.index]

        # Train/Test Split (80/20 chronological)
        split_idx = int(len(X) * 0.80)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Standard OLS Baseline
        ols = LinearRegression().fit(X_train, y_train)
        ols_preds = ols.predict(X_test)
        ols_mse = mean_squared_error(y_test, ols_preds)

        # Cross-validated LASSO
        lasso_cv = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
        best_alpha = lasso_cv.alpha_
        lasso = Lasso(alpha=best_alpha, random_state=42).fit(X_train, y_train)
        lasso_preds = lasso.predict(X_test)
        lasso_mse = mean_squared_error(y_test, lasso_preds)

        # Non-zero coefficient extraction
        coefs = pd.Series(lasso.coef_, index=X.columns)
        active_features = coefs[coefs != 0.0].sort_values(key=abs, ascending=False)
        sparsity_pct = (1.0 - len(active_features) / len(coefs)) * 100

        return {
            "target": target_asset,
            "total_candidate_features": len(coefs),
            "selected_features_count": len(active_features),
            "sparsity_pct": sparsity_pct,
            "best_alpha": float(best_alpha),
            "ols_test_mse": float(ols_mse),
            "lasso_test_mse": float(lasso_mse),
            "active_features": active_features
        }

    def run_pca_eigenportfolios(self, n_components: int = 4) -> dict:
        """Decompose asset covariance into orthogonal principal risk components and eigenportfolio weights."""
        returns = self.df_returns
        assets = returns.columns.tolist()

        # Standardized returns
        norm_returns = (returns - returns.mean()) / returns.std()

        pca = PCA(n_components=n_components)
        pca.fit(norm_returns)

        explained_var = pca.explained_variance_ratio_
        eigenvectors = pca.components_

        # Eigenportfolio weights (normalized so sum of absolute weights = 1)
        eigenportfolios = {}
        for i in range(n_components):
            v = eigenvectors[i]
            w = v / np.sum(np.abs(v))
            eigenportfolios[f"PC{i+1}"] = pd.Series(w, index=assets)

        return {
            "explained_variance": explained_var,
            "cumulative_variance": np.cumsum(explained_var),
            "eigenportfolios": eigenportfolios
        }

    def run_kmeans_risk_clustering(self, n_clusters: int = 3, seed: int = 42) -> dict:
        """Cluster assets based on the non-linear correlation distance metric d = sqrt(0.5 * (1 - rho))."""
        corr = self.df_returns.corr()
        assets = corr.columns.tolist()

        # Correlation distance metric (López de Prado)
        dist_matrix = np.sqrt(0.5 * (1.0 - corr.clip(-1.0, 1.0)))

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(dist_matrix)

        clusters = {}
        for c in range(n_clusters):
            members = [assets[i] for i in range(len(assets)) if labels[i] == c]
            sub_corr = corr.loc[members, members]
            avg_intra_corr = float(sub_corr.values[np.triu_indices_from(sub_corr.values, k=1)].mean()) if len(members) > 1 else 1.0
            clusters[f"Cluster_{c+1}"] = {
                "members": members,
                "avg_intra_corr": avg_intra_corr
            }

        return clusters


def main():
    print("=" * 75)
    print("Statistical Machine Learning & Factor Investing Engine")
    print("=" * 75)

    data_file = Path(__file__).parent / "data" / "stock_returns.csv"
    engine = FactorLearningEngine(str(data_file))

    print(f"Dataset: 10 US Equities ({len(engine.df_returns)} trading days)")
    print(f"Assets : {list(engine.df_returns.columns)}")
    print("-" * 75)

    # 1. LASSO L1 Regularization
    print("1. LASSO (L1 Regularization) Sparse Factor Selection for AAPL:")
    lasso_res = engine.run_lasso_feature_selection(target_asset="AAPL", n_lags=5)
    print(f"  Candidate Features (5 Lags x 10 Assets) : {lasso_res['total_candidate_features']}")
    print(f"  Optimal Regularization Parameter Alpha  : {lasso_res['best_alpha']:.6f}")
    print(f"  Selected Non-Zero Predictors            : {lasso_res['selected_features_count']}")
    print(f"  Enforced Sparsity Ratio                 : {lasso_res['sparsity_pct']:.1f}%")
    print(f"  Test MSE - Standard OLS Baseline        : {lasso_res['ols_test_mse']:.6f}")
    print(f"  Test MSE - Sparse LASSO Model           : {lasso_res['lasso_test_mse']:.6f}")
    print("  Top Selected Predictive Drivers:")
    for feat, coef in lasso_res["active_features"].head(5).items():
        print(f"    {feat:<15} : {coef:+.5f}")
    print("-" * 75)

    # 2. PCA Eigenportfolios
    print("2. Principal Component Analysis (PCA) Eigenportfolio Decomposition:")
    pca_res = engine.run_pca_eigenportfolios(n_components=4)
    exp_var = pca_res["explained_variance"]
    cum_var = pca_res["cumulative_variance"]

    print("  Component Variance Decomposition:")
    for i in range(len(exp_var)):
        print(f"    PC{i+1} : {exp_var[i]*100:5.2f}% of Variance (Cumulative: {cum_var[i]*100:5.2f}%)")

    print("\n  PC1 (Market Factor) vs PC2 (Style/Sector Rotation) Portfolio Weights:")
    df_weights = pd.DataFrame({
        "PC1 (Market)": pca_res["eigenportfolios"]["PC1"],
        "PC2 (Sector Tilt)": pca_res["eigenportfolios"]["PC2"]
    })
    for asset, row in df_weights.iterrows():
        print(f"    {asset:<5} | PC1: {row['PC1 (Market)']:+6.3f} | PC2: {row['PC2 (Sector Tilt)']:+6.3f}")
    print("-" * 75)

    # 3. k-Means Correlation Distance Clustering
    print("3. Unsupervised k-Means Risk Clustering (Correlation Metric):")
    clusters = engine.run_kmeans_risk_clustering(n_clusters=3)
    for c_name, c_info in clusters.items():
        members_str = ", ".join(c_info["members"])
        print(f"  {c_name}: [{members_str}] (Avg Intra-Cluster Correlation: {c_info['avg_intra_corr']:.3f})")
    print("=" * 75)


if __name__ == "__main__":
    main()
