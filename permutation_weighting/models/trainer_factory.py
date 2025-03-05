"""
Trainer factory for permutation weighting.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def get_trainer_factory(classifier, params=None):
    """
    Factory function for creating trainers

    Parameters
    ----------
    classifier : str
        Type of classifier ('logit' or 'boosting')
    params : dict, optional
        Classifier parameters

    Returns
    -------
    function
        A function that trains a classifier
    """
    if params is None:
        params = {}

    if classifier == 'logit':
        return logit_factory(params)
    elif classifier == 'boosting':
        return boosting_factory(params)
    else:
        raise ValueError(f'Unknown classifier: {classifier}')


def construct_df(data):
    """
    Constructs a DataFrame from permutation weighting data

    Parameters
    ----------
    data : dict
        Dictionary containing permuted and observed data

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame for classification
    """
    # Extract dimensions
    n_permuted = len(data['permuted']['A'])
    n_observed = len(data['observed']['A'])

    # Get feature dimension
    if isinstance(data['permuted']['X'], np.ndarray):
        n_features = data['permuted']['X'].shape[1]
    else:
        n_features = data['permuted']['X'].shape[1]

    # Combine data
    df = pd.DataFrame({
        'C': np.concatenate([
            np.repeat(data['permuted']['C'], n_permuted),
            np.repeat(data['observed']['C'], n_observed)
        ]),
        'A': np.concatenate([data['permuted']['A'], data['observed']['A']])
    })

    # Add X features
    X_combined = np.vstack([data['permuted']['X'], data['observed']['X']])
    for i in range(n_features):
        df[f'X{i}'] = X_combined[:, i]

    # Add interactions between A and X
    for i in range(n_features):
        df[f'A_X{i}'] = df['A'] * df[f'X{i}']

    return df


def construct_eval_df(A, X):
    """
    Constructs a DataFrame for evaluation

    Parameters
    ----------
    A : array-like
        Treatment variable
    X : array-like
        Covariate matrix

    Returns
    -------
    pandas.DataFrame
        DataFrame for evaluation
    """
    n_features = X.shape[1]

    df = pd.DataFrame({'A': A})

    # Add X features
    for i in range(n_features):
        df[f'X{i}'] = X[:, i]

    # Add interactions between A and X
    for i in range(n_features):
        df[f'A_X{i}'] = df['A'] * df[f'X{i}']

    return df


def logit_factory(params=None):
    """
    Factory for logistic regression trainer

    Parameters
    ----------
    params : dict, optional
        Logistic regression parameters

    Returns
    -------
    function
        A function that trains a logistic regression
    """
    if params is None:
        params = {}

    # Set default parameters
    default_params = {
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42
    }

    # Override defaults with provided params
    model_params = {**default_params, **params}

    def trainer(data):
        """
        Trains a logistic regression model

        Parameters
        ----------
        data : dict
            Dictionary containing permuted and observed data

        Returns
        -------
        function
            A function that computes weights
        """
        df = construct_df(data)

        # Separate features and target
        X_cols = [col for col in df.columns if col != 'C']
        X_train = df[X_cols]
        y_train = df['C']

        # Train model
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)

        def weight_function(A, X):
            """
            Computes weights from the trained model

            Parameters
            ----------
            A : array-like
                Treatment variable
            X : array-like
                Covariate matrix

            Returns
            -------
            numpy.ndarray
                Computed weights
            """
            eval_df = construct_eval_df(A, X)

            # Predict probabilities
            probs = model.predict_proba(eval_df)[:, 1]

            # Compute weights
            weights = probs / (1 - probs)

            return weights

        return weight_function

    return trainer


def boosting_factory(params=None):
    """
    Factory for gradient boosting trainer

    Parameters
    ----------
    params : dict, optional
        Gradient boosting parameters

    Returns
    -------
    function
        A function that trains a gradient boosting classifier
    """
    if params is None:
        params = {}

    # Set default parameters
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }

    # Override defaults with provided params
    model_params = {**default_params, **params}

    def trainer(data):
        """
        Trains a gradient boosting model

        Parameters
        ----------
        data : dict
            Dictionary containing permuted and observed data

        Returns
        -------
        function
            A function that computes weights
        """
        df = construct_df(data)

        # Separate features and target
        X_cols = [col for col in df.columns if col != 'C']
        X_train = df[X_cols]
        y_train = df['C']

        # Train model
        model = GradientBoostingClassifier(**model_params)
        model.fit(X_train, y_train)

        def weight_function(A, X):
            """
            Computes weights from the trained model

            Parameters
            ----------
            A : array-like
                Treatment variable
            X : array-like
                Covariate matrix

            Returns
            -------
            numpy.ndarray
                Computed weights
            """
            eval_df = construct_eval_df(A, X)

            # Predict probabilities
            probs = model.predict_proba(eval_df)[:, 1]

            # Compute weights
            weights = probs / (1 - probs)

            return weights

        return weight_function

    return trainer