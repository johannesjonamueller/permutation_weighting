import numpy as np
import pandas as pd
import warnings
from sklearn.neighbors import KernelDensity
from scipy import optimize
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
import time

warnings.filterwarnings('ignore')


def create_kang_schafer_continuous_data(n=1000, misspecified=False, treatment_noise_sd=1, logit=True):
    """
    Create the Kang-Schafer dataset with continuous treatment.

    Parameters
    ----------
    n : int, default=1000
        Number of samples
    misspecified : bool, default=False
        Whether to transform the covariates (misspecified case)
    treatment_noise_sd : float, default=1
        Standard deviation of the noise added to the treatment
    logit : bool, default=True
        Whether to transform the treatment using logistic function

    Returns
    -------
    tuple
        (A, X, Y) arrays
    """
    # Generate covariates
    X = np.random.normal(size=(n, 4))

    # Generate z-score (linear combination of covariates)
    coeffs = np.array([1, -0.5, 0.25, 0.1])
    z_score = X.dot(coeffs)

    # Generate treatment with noise
    noise = np.random.normal(0, treatment_noise_sd, size=n)
    A = z_score + noise

    # Treatment transformation function for outcome
    trt_fn = lambda z: 1 / (1 + np.exp(z)) if logit else z

    # Generate outcome
    base_outcome = 210 + 27.4 * X[:, 0] + 13.7 * X[:, 1] + 13.7 * X[:, 2] + 13.7 * X[:, 3]
    outcome_noise = np.random.normal(size=n)
    Y = base_outcome + trt_fn(A) + outcome_noise

    # Transform covariates if misspecified
    if misspecified:
        X_transformed = np.zeros_like(X)
        X_transformed[:, 0] = np.exp(X[:, 0] / 2)
        X_transformed[:, 1] = X[:, 1] / (1 + np.exp(X[:, 0])) + 10
        X_transformed[:, 2] = (X[:, 0] * X[:, 2] / 25 + 0.6) ** 3
        X_transformed[:, 3] = (X[:, 1] + X[:, 3] + 20) ** 2
        X = X_transformed

    return A, X, Y


def calculate_ipsw_linear(A, X):
    """
    Calculate inverse propensity score weights for continuous treatment
    using normal linear regression (PS in the paper).

    Parameters
    ----------
    A : array-like
        Continuous treatment variable
    X : array-like
        Covariate matrix

    Returns
    -------
    numpy.ndarray
        Computed weights
    """
    # Add constant term
    X_with_const = sm.add_constant(X)

    # Fit linear model
    model = sm.OLS(A, X_with_const)
    result = model.fit()

    # Predict treatment
    A_pred = result.predict(X_with_const)

    # Compute residuals
    residuals = A - A_pred

    # Estimate residual variance
    sigma2 = np.mean(residuals ** 2)

    # Compute density ratios (weights)
    # For continuous treatments, weights are based on PDF ratios
    # We need to compute p(A|X)/p(A) where p(A|X) is N(A_pred, sigma2)
    # and p(A) is the marginal distribution of A

    # Compute p(A|X)
    p_a_given_x = 1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-(A - A_pred) ** 2 / (2 * sigma2))

    # Estimate p(A) using kernel density estimation
    kde = KernelDensity(bandwidth='scott').fit(A.reshape(-1, 1))
    p_a = np.exp(kde.score_samples(A.reshape(-1, 1)))

    # Compute weights as stabilized IPW: p(A)/p(A|X)
    weights = p_a / p_a_given_x

    # Normalize weights to avoid extreme values
    weights = weights / np.mean(weights) * len(weights)

    # Handle extreme weights
    q1, q99 = np.percentile(weights, [1, 99])
    weights = np.clip(weights, q1, q99)

    return weights


def calculate_ipsw_gbm_continuous(A, X):
    """
    Calculate inverse propensity score weights for continuous treatment
    using gradient boosting regression (GBM in the paper).

    Parameters
    ----------
    A : array-like
        Continuous treatment variable
    X : array-like
        Covariate matrix

    Returns
    -------
    numpy.ndarray
        Computed weights
    """
    # Fit GBM model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=42)
    model.fit(X, A)

    # Predict treatment
    A_pred = model.predict(X)

    # Compute residuals
    residuals = A - A_pred

    # Estimate residual variance
    sigma2 = np.mean(residuals ** 2)

    # Compute density ratios (weights)
    # Similar approach as in the linear case
    p_a_given_x = 1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-(A - A_pred) ** 2 / (2 * sigma2))

    # Estimate p(A) using kernel density estimation
    kde = KernelDensity(bandwidth='scott').fit(A.reshape(-1, 1))
    p_a = np.exp(kde.score_samples(A.reshape(-1, 1)))

    # Compute weights
    weights = p_a / p_a_given_x

    # Normalize weights
    weights = weights / np.mean(weights) * len(weights)

    # Handle extreme weights
    q1, q99 = np.percentile(weights, [1, 99])
    weights = np.clip(weights, q1, q99)

    return weights


def calculate_npcbps(A, X):
    """
    Calculate non-parametric covariate balancing propensity score weights
    for continuous treatment (NPCBPS in the paper).

    Parameters
    ----------
    A : array-like
        Continuous treatment variable
    X : array-like
        Covariate matrix

    Returns
    -------
    numpy.ndarray
        Computed weights
    """
    n = len(A)
    t_start = time.time()
    max_time =  n /1000 *5 # seconds


    # Standardize treatment and covariates
    A_std = (A - np.mean(A)) / np.std(A)
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Define basis functions (including interactions)
    def create_basis(X_std, A_std):
        # Include linear terms
        basis = [X_std[:, j] for j in range(X_std.shape[1])]

        # Include interactions with treatment
        for j in range(X_std.shape[1]):
            basis.append(X_std[:, j] * A_std)

        # Convert to array
        return np.column_stack(basis)

    # Create basis functions
    B = create_basis(X_std, A_std)

    # Initial weights
    w_init = np.ones(n) / n

    # Function to compute balance violation (correlation between treatment and covariates)
    def balance_violation(w):
        # Normalize weights
        w = w / np.sum(w) * n

        # Compute weighted correlations
        w_mean_A = np.sum(w * A_std) / np.sum(w)
        w_mean_B = np.sum(w * B.T, axis=1) / np.sum(w)

        # Compute weighted covariance
        cov_terms = []
        for j in range(B.shape[1]):
            cov = np.sum(w * (B[:, j] - w_mean_B[j]) * (A_std - w_mean_A)) / np.sum(w)
            cov_terms.append(cov ** 2)

        return np.sum(cov_terms)

    # Function to optimize (entropy + lambda * balance)
    def objective(w, lambda_val=1.0):
        # Entropy term (minimize deviation from uniform weights)
        entropy = np.sum((w / np.mean(w) - 1) ** 2)

        # Balance term
        bal = balance_violation(w)

        return entropy + lambda_val * bal

    # Constraints (weights sum to n and are non-negative)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - n}
    ]

    bounds = [(1e-6, None) for _ in range(n)]

    # Optimize with reasonable computation limits
    try:
        result = optimize.minimize(
            objective,
            w_init,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-4},
            callback=lambda x: time.time() - t_start > max_time  # Stop if time exceeded
        )

        # Extract weights
        weights = result.x
    except:
        # Fallback to simpler method if optimization fails
        print("NPCBPS optimization failed, using linear IPSW as fallback")
        return calculate_ipsw_linear(A, X)

    # Normalize weights
    weights = weights / np.sum(weights) * n

    # Handle extreme weights
    q1, q99 = np.percentile(weights, [1, 99])
    weights = np.clip(weights, q1, q99)

    return weights


# def evaluate_dose_response(df, weights=None, n_points=19, bandwidth=None):
#     """
#     Evaluate the dose-response function at specified treatment levels.
#
#     Parameters
#     ----------
#     df : DataFrame
#         DataFrame with A (treatment) and Y (outcome) columns
#     weights : array-like, optional
#         Sample weights
#     n_points : int, default=19
#         Number of points to evaluate
#     bandwidth : float, optional
#         Kernel bandwidth
#
#     Returns
#     -------
#     tuple
#         (a_values, estimated_outcomes, true_outcomes)
#     """
#     if weights is None:
#         weights = np.ones(len(df))
#
#     # Normalize weights
#     weights = weights / np.sum(weights) * len(weights)
#
#     # Treatment values to evaluate
#     a_values = np.quantile(df['A'], np.linspace(0.05, 0.95, n_points))
#
#     # Set bandwidth if not provided
#     if bandwidth is None:
#         # Scott's rule
#         bandwidth = 1.06 * np.std(df['A']) * len(df['A']) ** (-1 / 5)
#
#     estimated_outcomes = []
#
#     for a in a_values:
#         # Calculate kernel weights
#         kernel_weights = np.exp(-0.5 * ((df['A'] - a) / bandwidth) ** 2)
#         kernel_weights = kernel_weights / np.sum(kernel_weights) * len(df)
#
#         # Combine with balancing weights
#         combined_weights = weights * kernel_weights
#         combined_weights = combined_weights / np.sum(combined_weights) * len(df)
#
#         # Estimate outcome
#         estimated_outcome = np.sum(combined_weights * df['Y']) / np.sum(combined_weights)
#         estimated_outcomes.append(estimated_outcome)
#
#     # Calculate true outcomes (assuming logistic dose-response for now)
#     # True dose response is 1/(1+exp(-a)) + base outcome
#     # But we're only interested in the effect relative to covariates
#     true_outcomes = 1 / (1 + np.exp(-a_values))
#
#     return a_values, np.array(estimated_outcomes), true_outcomes
#
#
# def evaluate_rmse(df, weights=None):
#     """
#     Evaluate the RMSE of dose-response estimation.
#
#     Parameters
#     ----------
#     df : DataFrame
#         DataFrame with A and Y columns
#     weights : array-like, optional
#         Sample weights
#
#     Returns
#     -------
#     float
#         RMSE between estimated and true dose-response
#     """
#     # Evaluate dose-response
#     a_values, estimated, true = evaluate_dose_response(df, weights)
#
#     # Center both estimated and true to compare treatment effects only
#     estimated_centered = estimated - np.mean(estimated)
#     true_centered = true - np.mean(true)
#
#     # Calculate RMSE
#     mse = np.mean((estimated - true - np.mean(estimated - true)) ** 2)
#     rmse = np.sqrt(mse)
#
#     # Calculate mean absolute bias
#     mean_abs_bias = np.mean(np.abs(estimated - true))
#
#     return {
#         'rmse': rmse,
#         'mean_abs_bias': mean_abs_bias,
#         'a_values': a_values,
#         'estimated': estimated,
#         'true': true
#     }

def evaluate_rmse(df, weights=None):
    """
    Evaluate the integrated RMSE of dose-response estimation.

    Parameters
    ----------
    df : DataFrame
        DataFrame with A and Y columns
    weights : array-like, optional
        Sample weights

    Returns
    -------
    dict
        Dictionary with RMSE and other metrics
    """
    if weights is None:
        weights = np.ones(len(df))

    # Normalize weights
    weights = weights / np.sum(weights) * len(weights)

    # Define treatment grid (use 19 points like in the paper)
    n_points = 19
    a_values = np.linspace(
        np.percentile(df['A'], 5),
        np.percentile(df['A'], 95),
        n_points
    )

    # Set kernel bandwidth
    bandwidth = 1.06 * np.std(df['A']) * len(df['A']) ** (-1 / 5)

    estimated_outcomes = []
    true_outcomes = []

    for a in a_values:
        # Kernel weights
        kernel_weights = np.exp(-0.5 * ((df['A'] - a) / bandwidth) ** 2)
        kernel_weights = kernel_weights / np.sum(kernel_weights) * len(df)

        # Combined with balancing weights
        combined_weights = weights * kernel_weights
        combined_weights = combined_weights / np.sum(combined_weights) * len(df)

        # Estimate outcome
        estimated_outcome = np.sum(combined_weights * df['Y']) / np.sum(combined_weights)
        estimated_outcomes.append(estimated_outcome)

        # True outcome is just logit(a)
        true_outcome = 1 / (1 + np.exp(-a))
        true_outcomes.append(true_outcome)

    # Extract just the treatment effect from both estimated and true outcomes
    baseline = 210 + np.mean(df[['X1', 'X2', 'X3', 'X4']].values.dot([27.4, 13.7, 13.7, 13.7]))
    estimated_centered = np.array(estimated_outcomes) - baseline
    true_centered = np.array(true_outcomes)

    # Calculate integrated metrics
    abs_bias = np.mean(np.abs(estimated_centered - true_centered))
    mse = np.mean((estimated_centered - true_centered) ** 2)
    rmse = np.sqrt(mse)

    return {
        'rmse': rmse,
        'mean_abs_bias': abs_bias,
        'a_values': a_values,
        'estimated': estimated_outcomes,
        'true': [t + baseline for t in true_outcomes]  # Add baseline for visualization
    }