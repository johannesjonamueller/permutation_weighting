"""
Data generation factories for permutation weighting.
"""

import numpy as np


def get_data_factory(A, X, estimand, params=None):
    """
    Factory function for creating data generation functions

    Parameters
    ----------
    A : array-like
        Treatment variable
    X : array-like
        Covariate matrix
    estimand : str
        Estimand type ('ATE' or 'ATT')
    params : dict, optional
        Additional parameters

    Returns
    -------
    function
        A function that generates balanced datasets
    """
    from .data_validation import check_data_for_att, is_data_binary

    if params is None:
        params = {}

    estimand = estimand.upper()

    # Always use regular ate_factory for continuous treatments
    if not is_data_binary(A):
        return ate_factory(A, X, params)

    if estimand == 'ATE':
        return ate_factory(A, X, params)
    elif estimand == 'ATT':
        check_data_for_att(A, X)
        return att_factory(A, X, params)
    else:
        raise ValueError(f'Unknown estimand: {estimand}')


def get_binary_data_factory(A, X, estimand, params=None):
    """
    Factory function for creating data generation functions for binary treatments

    Parameters
    ----------
    A : array-like
        Binary treatment variable
    X : array-like
        Covariate matrix
    estimand : str
        Estimand type ('ATE' or 'ATT')
    params : dict, optional
        Additional parameters

    Returns
    -------
    function
        A function that generates balanced datasets
    """
    from .data_validation import check_data_for_att

    if params is None:
        params = {}

    estimand = estimand.upper()

    if estimand == 'ATE':
        return binary_ate_factory(A, X, params)
    elif estimand == 'ATT':
        check_data_for_att(A, X)
        return binary_att_factory(A, X, params)
    else:
        raise ValueError(f'Unknown estimand: {estimand}')


# def ate_factory(A, X, params=None):
#     """
#     Factory for ATE estimand
#
#     Parameters
#     ----------
#     A : array-like
#         Treatment variable
#     X : array-like
#         Covariate matrix
#     params : dict, optional
#         Additional parameters
#
#     Returns
#     -------
#     function
#         A function that generates balanced datasets for ATE
#     """
#     if params is None:
#         params = {}
#
#     bootstrap = params.get('bootstrap', True)
#     N = len(A)
#
#     def factory():
#         if bootstrap:
#             # # Generate bootstrap indices
#             # idx = np.random.choice(N, N, replace=True)
#             #
#             # # For permuted data, independently sample treatment and covariates
#             # # IMPORTANT FIX FOR CONTINUOUS TREATMENTS: use random permutation for pA
#             # perm_idx = np.random.permutation(N)
#             # pA = A[perm_idx]
#             # pX = X[idx]
#             #
#             # # For observed data, use same bootstrap indices
#             # oA = A[idx]
#             # oX = X[idx]
#             #if bootstrap:
#         # Create different indices for different parts
#             p_idx = np.random.choice(N, N, replace=True)  # For permuted X
#             a_idx = np.random.choice(N, N, replace=True)  # For permuted A
#             o_idx = np.random.choice(N, N, replace=True)  # For observed data
#
#             # Use different random permutations
#             perm_idx = np.random.permutation(N)
#             pA = A[perm_idx][a_idx]  # Permute then bootstrap
#             pX = X[p_idx]
#             oA = A[o_idx]
#             oX = X[o_idx]
#         else:
#             # Non-bootstrap version
#             perm_idx = np.random.permutation(N)
#             idx = np.random.permutation(N)
#             pA = A[perm_idx]
#             pX = X[idx]
#             oA = A[idx]
#             oX = X[idx]
#
#         return {
#             'permuted': {'C': 1, 'A': pA, 'X': pX},
#             'observed': {'C': 0, 'A': oA, 'X': oX}
#         }
#
#     return factory
def ate_factory(A, X, params=None):
    if params is None:
        params = {}

    bootstrap = params.get('bootstrap', True)
    N = len(A)

    def factory():
        if bootstrap:
            # Create indices for initial permutation
            sampled_ids = np.random.permutation(N)

            # Create bootstrap indices for permuted A
            a_perm_idx = np.random.choice(N, N, replace=True)
            pA = A[sampled_ids[a_perm_idx]]  # Sample from permuted treatments

            # Create bootstrap indices for X
            x_perm_idx = np.random.choice(N, N, replace=True)
            pX = X[x_perm_idx]  # Sample features

            # Create bootstrap indices for observed data
            o_idx = np.random.choice(N, N, replace=True)
            oA = A[o_idx]
            oX = X[o_idx]
        else:
            # Non-bootstrap version with proper permutation
            perm_idx = np.random.permutation(N)
            pA = A[perm_idx]

            # Sample X independently
            x_idx = np.random.permutation(N)
            pX = X[x_idx]

            # Keep original data intact
            oA = A
            oX = X

        return {
            'permuted': {'C': 1, 'A': pA, 'X': pX},
            'observed': {'C': 0, 'A': oA, 'X': oX}
        }

    return factory

def att_factory(A, X, params=None):
    """
    Factory for ATT estimand

    Parameters
    ----------
    A : array-like
        Treatment variable
    X : array-like
        Covariate matrix
    params : dict, optional
        Additional parameters

    Returns
    -------
    function
        A function that generates balanced datasets for ATT
    """
    if params is None:
        params = {}

    bootstrap = params.get('bootstrap', True)
    N = len(A)
    A1_idx = np.where(A == 1)[0]

    def factory():
        # Sample treatment with replacement
        pA = np.random.choice(A, N, replace=bootstrap)

        # Sample covariates from treated units
        if bootstrap:
            p_idx = np.random.choice(A1_idx, N, replace=True)
        else:
            p_idx = np.random.choice(A1_idx, N, replace=False)

        pX = X[p_idx]

        # Sample observed data indices
        idx = np.random.choice(N, N, replace=bootstrap)

        return {
            'permuted': {'C': 1, 'A': pA, 'X': pX},
            'observed': {'C': 0, 'A': A[idx], 'X': X[idx]}
        }

    return factory


def binary_ate_factory(A, X, params=None):
    """
    Factory for binary ATE estimand

    Parameters
    ----------
    A : array-like
        Binary treatment variable
    X : array-like
        Covariate matrix
    params : dict, optional
        Additional parameters

    Returns
    -------
    function
        A function that generates balanced datasets for binary ATE
    """
    if params is None:
        params = {}

    N = len(A)
    unique_A = np.unique(A)

    def factory():
        # Create cross-product of unique A values with X
        return {
            'permuted': {
                'C': 1,
                'A': np.repeat(unique_A, N),
                'X': np.vstack([X, X])
            },
            'observed': {
                'C': 0,
                'A': A,
                'X': X
            }
        }

    return factory


def binary_att_factory(A, X, params=None):
    """
    Factory for binary ATT estimand

    Parameters
    ----------
    A : array-like
        Binary treatment variable
    X : array-like
        Covariate matrix
    params : dict, optional
        Additional parameters

    Returns
    -------
    function
        A function that generates balanced datasets for binary ATT
    """
    if params is None:
        params = {}

    N = len(A)
    A1_idx = np.where(A == 1)[0]
    N1 = len(A1_idx)

    def factory():
        return {
            'permuted': {
                'C': 1,
                'A': np.ones(N1),
                'X': X[A1_idx]
            },
            'observed': {
                'C': 0,
                'A': A,
                'X': X
            }
        }

    return factory