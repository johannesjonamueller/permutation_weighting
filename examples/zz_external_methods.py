"""
External methods for comparison with permutation weighting.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import norm


def get_features(X, misspecified=False):
    """
    Extract appropriate features based on misspecification flag

    Parameters:
    -----------
    X : array-like or DataFrame
        Covariates
    misspecified : bool
        Whether to use misspecified covariates

    Returns:
    --------
    array-like
        Feature matrix
    """
    if isinstance(X, pd.DataFrame):
        if misspecified and 'X1_mis' in X.columns:
            X_mat = X[['X1_mis', 'X2_mis', 'X3_mis', 'X4_mis']].values
        else:
            X_mat = X[['X1', 'X2', 'X3', 'X4']].values
    else:
        X_mat = X

    return X_mat


# BINARY TREATMENT METHODS

def compute_unweighted(A, X, misspecified=False):
    """
    Compute uniform weights (no adjustment)

    Parameters:
    -----------
    A : array-like
        Treatment variable
    X : array-like or DataFrame
        Covariates
    misspecified : bool
        Whether to use misspecified covariates (unused, included for API consistency)

    Returns:
    --------
    numpy.ndarray
        Unit weights
    """
    return np.ones(len(A))


def compute_ipsw_logit(A, X, misspecified=False):
    """
    Compute Stabilized Inverse Propensity Score Weights using logistic regression

    Parameters:
    -----------
    A : array-like
        Binary treatment indicator
    X : array-like or DataFrame
        Covariates
    misspecified : bool
        Whether to use misspecified covariates

    Returns:
    --------
    numpy.ndarray
        Stabilized IPW weights
    """
    X_mat = get_features(X, misspecified)

    try:
        # Fit propensity score model
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X_mat, A)

        # Compute propensity scores
        ps = ps_model.predict_proba(X_mat)[:, 1]

        # Clip propensity scores to prevent extreme weights
        ps = np.clip(ps, 0.01, 0.99)

        # Marginal treatment probability
        p_A = np.mean(A)

        # Compute stabilized weights
        weights = np.where(A == 1, p_A / ps, (1 - p_A) / (1 - ps))

        return weights
    except Exception as e:
        print(f"Warning: Error in compute_ipsw_logit: {e}")
        return np.ones(len(A))


def compute_ipsw_gbm(A, X, misspecified=False):
    """
    Compute IPSW weights using gradient boosting

    Parameters:
    -----------
    A : array-like
        Treatment indicator
    X : array-like or DataFrame
        Covariates
    misspecified : bool
        Whether to use misspecified covariates

    Returns:
    --------
    numpy.ndarray
        IPSW weights
    """
    X_mat = get_features(X, misspecified)

    try:
        # Fit propensity score model with GBM
        gbm = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        gbm.fit(X_mat, A)

        # Compute propensity scores
        ps = gbm.predict_proba(X_mat)[:, 1]

        # Clip propensity scores to avoid extreme weights
        ps = np.clip(ps, 0.01, 0.99)

        # Marginal treatment probability
        p_A = np.mean(A)

        # Compute stabilized weights
        weights = np.where(A == 1, p_A / ps, (1 - p_A) / (1 - ps))

        return weights
    except Exception as e:
        print(f"Warning: Error in compute_ipsw_gbm: {e}")
        return np.ones(len(A))


def compute_cbps_binary(A, X, misspecified=False):
    """
    Simplified version of Covariate Balancing Propensity Score for binary treatment

    Parameters:
    -----------
    A : array-like
        Binary treatment indicator
    X : array-like or DataFrame
        Covariates
    misspecified : bool
        Whether to use misspecified covariates

    Returns:
    --------
    numpy.ndarray
        CBPS weights
    """
    X_mat = get_features(X, misspecified)

    try:
        # This is a simplification - true CBPS adds balance constraints
        # For demonstration, we use logistic regression with L2 penalty
        ps_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        ps_model.fit(X_mat, A)

        # Compute propensity scores
        ps = ps_model.predict_proba(X_mat)[:, 1]

        # Clip propensity scores to avoid extreme weights
        ps = np.clip(ps, 0.01, 0.99)

        # Marginal treatment probability
        p_A = np.mean(A)

        # Compute weights
        weights = np.where(A == 1, p_A / ps, (1 - p_A) / (1 - ps))

        return weights
    except Exception as e:
        print(f"Warning: Error in compute_cbps_binary: {e}")
        return np.ones(len(A))


def compute_sbw_binary(A, X, misspecified=False):
    """
    Simplified version of Stabilized Balancing Weights for binary treatment

    Parameters:
    -----------
    A : array-like
        Binary treatment indicator
    X : array-like or DataFrame
        Covariates
    misspecified : bool
        Whether to use misspecified covariates

    Returns:
    --------
    numpy.ndarray
        SBW weights
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("Warning: cvxpy not installed. Falling back to unweighted.")
        return np.ones(len(A))

    X_mat = get_features(X, misspecified)

    try:
        n = len(A)
        n_treated = np.sum(A)
        n_control = n - n_treated

        if n_treated == 0 or n_control == 0:
            print("Warning: No treated or control units found. Falling back to unweighted.")
            return np.ones(n)

        # Separate covariates for treated and control
        X_treated = X_mat[A == 1]
        X_control = X_mat[A == 0]

        # Calculate means
        treated_mean = np.mean(X_treated, axis=0)

        # Initialize weights for control units
        w = cp.Variable(n_control, nonneg=True)

        # Balance constraint: weighted mean of control features equals mean of treated features
        balance_constraint = []
        for j in range(X_mat.shape[1]):
            # Allow small imbalance (delta)
            delta = 0.1 * np.std(X_mat[:, j])
            balance_constraint.append(
                cp.abs(cp.sum(cp.multiply(w, X_control[:, j])) - treated_mean[j] * cp.sum(w)) <= delta
            )

        # Sum constraint
        balance_constraint.append(cp.sum(w) == 1)

        # Objective: minimize variance
        objective = cp.Minimize(cp.sum_squares(w - 1 / n_control))

        # Solve optimization problem
        prob = cp.Problem(objective, balance_constraint)
        try:
            prob.solve(solver=cp.OSQP)
        except:
            try:
                prob.solve(solver=cp.ECOS)
            except:
                prob.solve(solver=cp.SCS)

        # Create final weights vector
        weights = np.ones(n)
        weights[A == 0] = w.value * n_control

        # Check for NaN or inf values
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            print("Warning: NaN or inf values in SBW weights. Falling back to unweighted.")
            return np.ones(n)

        return weights
    except Exception as e:
        print(f"Warning: Error in compute_sbw_binary: {e}")
        return np.ones(len(A))


# CONTINUOUS TREATMENT METHODS

def compute_normal_linear_weights(A, X, misspecified=False):
    """
    Compute weights for continuous treatments using a normal-linear model

    Parameters:
    -----------
    A : array-like
        Continuous treatment variable
    X : array-like or DataFrame
        Covariates
    misspecified : bool
        Whether to use misspecified covariates

    Returns:
    --------
    numpy.ndarray
        Stabilized weights
    """
    X_mat = get_features(X, misspecified)

    try:
        # Fit linear regression for treatment given covariates
        model = LinearRegression()
        model.fit(X_mat, A)

        # Predict treatment and calculate residuals
        A_pred = model.predict(X_mat)
        resid = A - A_pred

        # Estimate residual variance
        sigma = np.std(resid)

        # Compute likelihood of observed treatment under model
        pdf_cond = norm.pdf(A, loc=A_pred, scale=sigma)

        # Compute likelihood under marginal distribution
        pdf_marg = norm.pdf(A, loc=np.mean(A), scale=np.std(A))

        # Compute stabilized weights
        weights = pdf_marg / pdf_cond

        # Clip extreme weights
        weights = np.clip(weights, 0.01, 100)

        return weights
    except Exception as e:
        print(f"Warning: Error in compute_normal_linear_weights: {e}")
        return np.ones(len(A))


def compute_np_cbps(A, X, misspecified=False):
    """
    Compute non-parametric Covariate Balancing Propensity Score weights for continuous treatments

    Parameters:
    -----------
    A : array-like
        Continuous treatment variable
    X : array-like or DataFrame
        Covariates
    misspecified : bool
        Whether to use misspecified covariates

    Returns:
    --------
    numpy.ndarray
        CBPS weights
    """
    X_mat = get_features(X, misspecified)

    try:
        # Ensure A is 1-dimensional
        A = np.asarray(A).flatten()

        # Create polynomial features to approximate non-parametric model
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(X_mat)

        # Fit Lasso model with cross-validation for regularization
        model = LassoCV(cv=5, max_iter=2000, random_state=42)
        model.fit(X_poly, A)

        # Compute residuals
        A_pred = model.predict(X_poly)
        resid = A - A_pred

        # Compute covariance between residuals and covariate functions
        cov_mat = np.zeros((X_poly.shape[1], 1))
        for j in range(X_poly.shape[1]):
            cov_mat[j, 0] = np.cov(X_poly[:, j], resid)[0, 1]

        # Use ridge regression to solve for lambda parameters (more stable)
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=0.01)
        ridge.fit(X_poly, resid)
        lambda_param = ridge.coef_.reshape(-1, 1)

        # Compute weights using exponential tilting
        scores = X_poly @ lambda_param

        # Apply exponential tilting and normalize
        weights = np.exp(scores.flatten() - np.max(scores))  # Subtract max for numerical stability
        weights = weights / np.mean(weights) * len(A)

        # Clip extreme weights
        weights = np.clip(weights, 0.01, 100)

        return weights
    except Exception as e:
        print(f"Warning: Error in compute_np_cbps: {e}")
        return np.ones(len(A))