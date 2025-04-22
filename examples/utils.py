import numpy as np
import warnings

warnings.filterwarnings('ignore')

# For traditional methods
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier




def create_kang_schafer_data(n=1000, misspecified=False):
    """
    Create the Kang-Schafer dataset.

    Parameters
    ----------
    n : int, default=1000
        Number of samples
    misspecified : bool, default=False
        Whether to transform the covariates (misspecified case)

    Returns
    -------
    tuple
        (A, X, Y) arrays
    """
    # Generate covariates
    X = np.random.normal(size=(n, 4))

    # Generate propensity scores
    propensity = 1 / (1 + np.exp(X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2] + 0.1 * X[:, 3]))

    # Generate treatment
    A = np.random.binomial(1, propensity, size=n)

    # Generate outcome
    Y = 210 + 27.4 * X[:, 0] + 13.7 * X[:, 1] + 13.7 * X[:, 2] + 13.7 * X[:, 3] + A + np.random.normal(size=n)

    # Transform covariates if misspecified
    if misspecified:
        X_transformed = np.zeros_like(X)
        X_transformed[:, 0] = np.exp(X[:, 0] / 2)
        X_transformed[:, 1] = X[:, 1] / (1 + np.exp(X[:, 0])) + 10
        X_transformed[:, 2] = (X[:, 0] * X[:, 2] / 25 + 0.6) ** 3
        X_transformed[:, 3] = (X[:, 1] + X[:, 3] + 20) ** 2
        X = X_transformed

    return A, X, Y


def compute_covariate_balance(A, X, weights=None):
    """
    Compute the covariate balance metrics.

    Parameters
    ----------
    A : array-like
        Treatment variable
    X : array-like
        Covariate matrix
    weights : array-like, optional
        Sample weights

    Returns
    -------
    dict
        Dictionary of balance metrics
    """
    if weights is None:
        weights = np.ones(len(A))

    # Normalize weights
    weights = weights / np.sum(weights) * len(weights)

    # Standardized mean differences
    std_mean_diffs = []

    # For each covariate
    for j in range(X.shape[1]):
        # Get covariate
        x_j = X[:, j]

        # Calculate weighted means
        treated_mean = np.sum(x_j[A == 1] * weights[A == 1]) / np.sum(weights[A == 1])
        control_mean = np.sum(x_j[A == 0] * weights[A == 0]) / np.sum(weights[A == 0])

        # Calculate pooled standard deviation
        treated_var = np.sum(weights[A == 1] * (x_j[A == 1] - treated_mean) ** 2) / np.sum(weights[A == 1])
        control_var = np.sum(weights[A == 0] * (x_j[A == 0] - control_mean) ** 2) / np.sum(weights[A == 0])
        pooled_std = np.sqrt((treated_var + control_var) / 2)

        # Calculate standardized mean difference
        std_mean_diff = abs(treated_mean - control_mean) / pooled_std if pooled_std > 0 else float('inf')
        std_mean_diffs.append(std_mean_diff)

    return {
        'std_mean_diffs': std_mean_diffs,
        'avg_std_mean_diff': np.mean(std_mean_diffs),
        'max_std_mean_diff': np.max(std_mean_diffs)
    }


# Traditional method implementations
def calculate_ipsw_glm(A, X):
    """Calculate inverse propensity score weights using logistic regression"""
    # Fit propensity score model
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X, A)

    # Calculate propensity scores
    ps = model.predict_proba(X)[:, 1]

    # Avoid numerical issues
    ps = np.clip(ps, 0.01, 0.99)

    # Calculate weights
    weights = np.where(A == 1, 1 / ps, 1 / (1 - ps))

    # Stabilize weights
    treated_indices = A == 1
    control_indices = A == 0

    # Calculate stabilizing factors
    stabilizer_treated = np.mean(A) / np.mean(weights[treated_indices]) if np.any(treated_indices) else 1.0
    stabilizer_control = (1 - np.mean(A)) / np.mean(weights[control_indices]) if np.any(control_indices) else 1.0

    # Apply stabilization
    stabilized_weights = np.copy(weights)
    if np.any(treated_indices):
        stabilized_weights[treated_indices] *= stabilizer_treated
    if np.any(control_indices):
        stabilized_weights[control_indices] *= stabilizer_control

    return stabilized_weights


def calculate_ipsw_gbm(A, X):
    """Calculate inverse propensity score weights using gradient boosting"""
    # Fit propensity score model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=42)
    model.fit(X, A)

    # Calculate propensity scores
    ps = model.predict_proba(X)[:, 1]

    # Avoid numerical issues
    ps = np.clip(ps, 0.01, 0.99)

    # Calculate weights
    weights = np.where(A == 1, 1 / ps, 1 / (1 - ps))

    # Stabilize weights
    treated_indices = A == 1
    control_indices = A == 0

    # Calculate stabilizing factors
    stabilizer_treated = np.mean(A) / np.mean(weights[treated_indices]) if np.any(treated_indices) else 1.0
    stabilizer_control = (1 - np.mean(A)) / np.mean(weights[control_indices]) if np.any(control_indices) else 1.0

    # Apply stabilization
    stabilized_weights = np.copy(weights)
    if np.any(treated_indices):
        stabilized_weights[treated_indices] *= stabilizer_treated
    if np.any(control_indices):
        stabilized_weights[control_indices] *= stabilizer_control

    return stabilized_weights


def calculate_cbps(A, X):
    """
    Calculate covariate balancing propensity score weights
    A simplified implementation of CBPS using statsmodels
    """
    # Add constant to X
    X_with_const = sm.add_constant(X)

    # Initial propensity score model
    logit_model = sm.Logit(A, X_with_const)
    result = logit_model.fit(disp=0)

    # Get initial coefficients
    beta = result.params

    # Calculate propensity scores
    ps = 1 / (1 + np.exp(-np.dot(X_with_const, beta)))
    ps = np.clip(ps, 0.01, 0.99)

    # Calculate weights
    weights = np.where(A == 1, 1 / ps, 1 / (1 - ps))

    # Iterative refinement to balance covariates (simplified)
    for _ in range(5):  # Usually 3-5 iterations are sufficient
        # Calculate weighted means of covariates
        w_treated = weights * A
        w_control = weights * (1 - A)

        mean_treated = np.average(X, axis=0, weights=w_treated)
        mean_control = np.average(X, axis=0, weights=w_control)

        # Calculate imbalance
        imbalance = mean_treated - mean_control

        # Adjust propensity scores to reduce imbalance (simplified)
        ps_adj = ps * (1 + 0.1 * np.dot(X, imbalance / np.linalg.norm(imbalance)))
        ps_adj = np.clip(ps_adj, 0.01, 0.99)

        # Recalculate weights
        weights_new = np.where(A == 1, 1 / ps_adj, 1 / (1 - ps_adj))

        # Stabilize weights
        treated_indices = A == 1
        control_indices = A == 0

        # Calculate stabilizing factors
        stabilizer_treated = np.mean(A) / np.mean(weights_new[treated_indices]) if np.any(treated_indices) else 1.0
        stabilizer_control = (1 - np.mean(A)) / np.mean(weights_new[control_indices]) if np.any(
            control_indices) else 1.0

        # Apply stabilization
        weights = np.copy(weights_new)
        if np.any(treated_indices):
            weights[treated_indices] *= stabilizer_treated
        if np.any(control_indices):
            weights[control_indices] *= stabilizer_control
        else:
            break

        ps = ps_adj

    return weights


