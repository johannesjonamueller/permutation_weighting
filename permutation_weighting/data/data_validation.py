"""
Data validation utilities for permutation weighting.
"""

import numpy as np

def check_data(A, X):
    """
    Validates the input data for permutation weighting.

    Parameters
    ----------
    A : array-like
        Treatment variable (binary or continuous)
    X : array-like
        Covariate matrix

    Raises
    ------
    ValueError
        If the data does not meet the requirements
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A)

    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if A.ndim != 1:
        raise ValueError("A must be a 1-dimensional array")

    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array (matrix)")

    if not np.issubdtype(A.dtype, np.number):
        raise ValueError("A must contain numeric values")

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must contain numeric values")

    N_A = len(A)
    N_X = X.shape[0]

    if N_A != N_X:
        raise ValueError(f"A and X must have the same number of observations, got {N_A} and {N_X}")

    return A, X

def check_data_for_att(A, X):
    """
    Additional checks for ATT estimand

    Parameters
    ----------
    A : array-like
        Treatment variable (must be binary for ATT)
    X : array-like
        Covariate matrix

    Raises
    ------
    ValueError
        If the data does not meet ATT requirements
    """
    A, X = check_data(A, X)

    # First check if there are any treated units #TODO: Discuss with Drew as a Fix for Tests
    if not np.any(A == 1):
        raise ValueError('A must take the value of one at least once for the ATT.')

    # Then check the number of unique values
    n_unq = len(np.unique(A))
    if n_unq != 2:
        raise ValueError('A can only have two unique values for the ATT estimand.')

    return A, X

def check_eval_data(eval_data):
    """
    Validates evaluation data

    Parameters
    ----------
    eval_data : dict
        Dictionary containing 'A' and 'X' keys

    Raises
    ------
    ValueError
        If eval_data is not properly formatted
    """
    if not isinstance(eval_data, dict):
        raise ValueError("eval_data must be a dictionary")

    if 'A' not in eval_data or 'X' not in eval_data:
        raise ValueError("eval_data must contain 'A' and 'X' keys")

    check_data(eval_data['A'], eval_data['X'])

    return eval_data

def is_data_binary(A):
    """
    Checks if treatment data is binary

    Parameters
    ----------
    A : array-like
        Treatment variable

    Returns
    -------
    bool
        True if A has exactly 2 unique values
    """
    n_unq = len(np.unique(A))
    return n_unq == 2