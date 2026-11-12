# Statistical Machine Learning in Finance: LASSO, PCA & k-Means Clustering

> **Program:** MSc in Financial Engineering (WorldQuant University) — MScFE 632 Machine Learning in Finance  
> **Student / Researcher:** Hitesh Nanwani  
> **Key Domains:** Factor Investing • Feature Selection • Dimensionality Reduction • Unsupervised Clustering

---

## Executive Summary

This project conducts an empirical computational investigation of core statistical machine learning algorithms applied to financial market data. It addresses the fundamental challenges of equity quantitative modeling: **high dimensionality, collinearity among multi-factor features, and non-linear risk groupings**.

The study implements and evaluates three primary methodologies:
1. **LASSO Regularization ($L_1$ Shrinkage):** Enforcing sparsity to isolate true predictive factors and eliminate redundant collinear predictors.
2. **Principal Component Analysis (PCA):** Extracting latent market and sector drivers (Eigenportfolios) from multi-asset return covariance matrices.
3. **k-Means Clustering:** Grouping equities based on empirical return distributions and volatility dynamics rather than static GICS sector classifications.

---

## Methodological Framework

### 1. LASSO Regression & Sparse Factor Selection
Given target returns $y \in \mathbb{R}^n$ and multi-factor matrix $X \in \mathbb{R}^{n \times p}$, the LASSO objective minimizes mean squared error subject to an $L_1$ penalty:

$$\min_{\beta_0, \boldsymbol{\beta}} \left\{ \frac{1}{2n} \sum_{i=1}^n \left( y_i - \beta_0 - \mathbf{x}_i^\top \boldsymbol{\beta} \right)^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}$$

- **Shrinkage & Sparsity:** Unlike Ridge ($L_2$) regression, the sharp corners of the $L_1$ simplex drive non-essential factor coefficients strictly to zero ($\beta_j = 0$).
- **Cross-Validation ($\lambda$ Selection):** Employs $K$-fold cross-validation to locate the optimal hyperparameter $\lambda^*$ balancing bias and estimation variance.

### 2. Principal Component Analysis (Eigenportfolios)
For asset returns $R \in \mathbb{R}^{T \times N}$ with empirical covariance matrix $\boldsymbol{\Sigma} = \frac{1}{T} R^\top R$, PCA performs eigendecomposition:

$$\boldsymbol{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i, \qquad i = 1, \dots, N$$

- **Eigenvalue Spectrum ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_N$):** The first principal component typically corresponds to the overall market factor ($\sim 50\%+$ of total variance explained), while subsequent components capture sector, size, and momentum rotation.
- **Dimensionality Reduction:** Projects returns into a reduced $k$-dimensional orthogonal factor space ($\mathbf{Z} = R \mathbf{V}_k$).

### 3. k-Means Asset Clustering
Partitions $N$ asset return trajectories into $K$ mutually exclusive clusters $S = \{S_1, \dots, S_K\}$ by minimizing within-cluster sum of squares (inertia):

$$\arg\min_{S} \sum_{k=1}^K \sum_{\mathbf{x} \in S_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$

- **Elbow Method & Silhouette Scores:** Used to identify the natural number of risk clusters present in empirical equities.

---

## Structure & Artifacts

```text
04_machine_learning_factor_models/
├── README.md                                # Methodological overview & theoretical foundations
├── lasso_pca_kmeans_factor_analysis.ipynb   # Complete implementation of LASSO, PCA & k-Means
├── ml_feature_selection_validation.ipynb    # Advanced model validation & feature selection issues
└── data/
    └── stock_returns.csv                    # Empirical equity return dataset
```

---

## Technologies Used

- **Language:** Python 3.12+ / Jupyter Notebook
- **Machine Learning:** `scikit-learn` (`LassoCV`, `PCA`, `KMeans`, `StandardScaler`)
- **Linear Algebra & Data:** `numpy`, `pandas`
- **Visualization:** `matplotlib`, `seaborn`
