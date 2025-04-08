"""
Data generation and evaluation utilities for permutation weighting simulations.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression

def create_output_dir(prefix="results"):
    """
    Create timestamped output directory

    Parameters:
    -----------
    prefix : str
        Prefix for the output directory

    Returns:
    --------
    str
        Path to the created directory
    """
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{prefix}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_kang_schafer_data(n=1000, seed=42, misspecified=False, continuous_treatment=False):
    """
    Generate data according to the Kang-Schafer setup with either binary or continuous treatment

    Parameters:
    -----------
    n : int
        Number of observations
    seed : int
        Random seed
    misspecified : bool
        Whether to return the misspecified transformations of covariates
    continuous_treatment : bool
        Whether to generate continuous (True) or binary (False) treatment

    Returns:
    --------
    df : pd.DataFrame
        Data frame with covariates, treatment, and outcome
    """
    np.random.seed(seed)

    # Generate covariates
    X = np.random.normal(0, 1, size=(n, 4))

    # Treatment assignment
    ps_linear = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2] + 0.1 * X[:, 3]

    if continuous_treatment:
        # Continuous treatment with noise
        A = ps_linear + np.random.normal(0, 1, size=n)

        # Generate outcome with non-linear treatment effect
        Y = 210 + 1 / (1 + np.exp(A)) + 27.4 * X[:, 0] + 13.7 * X[:, 1] + 13.7 * X[:, 2] + 13.7 * X[:,
                                                                                                  3] + np.random.normal(
            0, 1, size=n)

        # Create DataFrame
        df = pd.DataFrame({
            'X1': X[:, 0],
            'X2': X[:, 1],
            'X3': X[:, 2],
            'X4': X[:, 3],
            'A': A,
            'Y': Y
        })

        # True dose-response function (for evaluation)
        def true_dose_response(a):
            return 210 + 1 / (1 + np.exp(a))

        df['true_dr'] = [true_dose_response(a) for a in A]

    else:
        # Binary treatment
        ps = expit(ps_linear)
        A = np.random.binomial(1, ps, size=n)

        # Generate potential outcomes
        Y1 = 210 + 1 + 27.4 * X[:, 0] + 13.7 * X[:, 1] + 13.7 * X[:, 2] + 13.7 * X[:, 3] + np.random.normal(0, 1,
                                                                                                            size=n)
        Y0 = 210 + 0 + 27.4 * X[:, 0] + 13.7 * X[:, 1] + 13.7 * X[:, 2] + 13.7 * X[:, 3] + np.random.normal(0, 1,
                                                                                                            size=n)

        # Observed outcome
        Y = A * Y1 + (1 - A) * Y0

        # Create DataFrame
        df = pd.DataFrame({
            'X1': X[:, 0],
            'X2': X[:, 1],
            'X3': X[:, 2],
            'X4': X[:, 3],
            'A': A,
            'Y': Y,
            'Y1': Y1,
            'Y0': Y0
        })

    # Add misspecified covariates if requested
    if misspecified:
        df['X1_mis'] = np.exp(X[:, 0] / 2)
        df['X2_mis'] = X[:, 1] / (1 + np.exp(X[:, 0])) + 10
        df['X3_mis'] = (X[:, 0] * X[:, 2] / 25 + 0.6) ** 3
        df['X4_mis'] = (X[:, 1] + X[:, 3] + 20) ** 2

    return df


def evaluate_ate_binary(df, weights, true_ate=1.0):
    """
    Evaluate ATE estimation for binary treatment

    Parameters:
    -----------
    df : pd.DataFrame
        Data with 'A', 'Y', 'Y1', 'Y0' columns
    weights : array-like
        Weights for each observation
    true_ate : float
        True average treatment effect

    Returns:
    --------
    float
        ATE estimation error
    """
    # Check for valid weights
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        print("Warning: NaN or inf values in weights. Using unweighted estimation.")
        weights = np.ones(len(df))

    # Calculate weighted ATE
    treated_idx = df['A'] == 1
    control_idx = df['A'] == 0

    # Prevent division by zero
    treated_weight_sum = np.sum(weights[treated_idx])
    control_weight_sum = np.sum(weights[control_idx])

    if treated_weight_sum == 0 or control_weight_sum == 0:
        print("Warning: Zero sum of weights for treated or control group. Using unweighted estimation.")
        return np.abs(np.mean(df.loc[treated_idx, 'Y']) - np.mean(df.loc[control_idx, 'Y']) - true_ate)

    treated_mean = np.sum(df.loc[treated_idx, 'Y'] * weights[treated_idx]) / treated_weight_sum
    control_mean = np.sum(df.loc[control_idx, 'Y'] * weights[control_idx]) / control_weight_sum

    estimated_ate = treated_mean - control_mean

    error = estimated_ate - true_ate

    return error


def evaluate_dose_response_continuous(df, weights, treatment_grid=None):
    """
    Evaluate dose-response estimation for continuous treatment

    Parameters:
    -----------
    df : pd.DataFrame
        Data with 'A', 'Y', 'true_dr' columns
    weights : array-like
        Weights for each observation
    treatment_grid : array-like, optional
        Grid of treatment values for evaluation

    Returns:
    --------
    dict
        Dictionary with integrated bias and rmse
    """
    # Check for valid weights
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        print("Warning: NaN or inf values in weights. Using unweighted estimation.")
        weights = np.ones(len(df))

    # Define treatment grid if not provided
    if treatment_grid is None:
        min_a, max_a = np.percentile(df['A'], [5, 95])
        treatment_grid = np.linspace(min_a, max_a, 50)

    # True dose-response function
    true_dr = [210 + 1 / (1 + np.exp(a)) for a in treatment_grid]

    # Estimate dose-response function using weighted local linear regression
    est_dr = []
    for a in treatment_grid:
        # Calculate kernel weights
        bandwidth = (np.percentile(df['A'], 75) - np.percentile(df['A'], 25)) / 1.34
        kernel_weights = np.exp(-0.5 * ((df['A'] - a) / bandwidth) ** 2) * weights

        # Check for all zero weights
        if np.sum(kernel_weights) == 0:
            print(f"Warning: All zero kernel weights for treatment value {a}. Using unweighted estimation.")
            kernel_weights = np.exp(-0.5 * ((df['A'] - a) / bandwidth) ** 2)

        # Fit weighted linear regression
        try:
            model = LinearRegression()
            model.fit(
                df[['A']],
                df['Y'],
                sample_weight=kernel_weights
            )

            # Predict at treatment value a
            est_dr.append(model.predict([[a]])[0])
        except Exception as e:
            print(f"Warning: Error in local linear regression for a={a}: {e}")
            # Fallback to weighted mean of nearest neighbors
            idx = np.argsort(np.abs(df['A'] - a))[:5]  # 5 nearest neighbors
            if len(idx) > 0:
                nearest_weights = weights[idx]
                if np.sum(nearest_weights) > 0:
                    est_dr.append(np.sum(df.loc[idx, 'Y'] * nearest_weights) / np.sum(nearest_weights))
                else:
                    est_dr.append(np.mean(df.loc[idx, 'Y']))
            else:
                est_dr.append(np.mean(df['Y']))

    # Calculate integrated bias and RMSE
    bias = np.mean(np.abs(np.array(est_dr) - np.array(true_dr)))
    rmse = np.sqrt(np.mean((np.array(est_dr) - np.array(true_dr)) ** 2))

    return {
        'integrated_bias': bias,
        'integrated_rmse': rmse,
        'treatment_grid': treatment_grid,
        'estimated_dr': est_dr,
        'true_dr': true_dr
    }