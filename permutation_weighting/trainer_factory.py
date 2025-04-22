"""
Trainer factory for permutation weighting.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .data_validation import is_data_binary


def get_trainer_factory(classifier, params=None, A=None):
    """
    Factory function for creating trainers based on treatment type

    Parameters
    ----------
    classifier : str
        Type of model ('logit', 'boosting', 'sgd', 'mlp')
    params : dict, optional
        Model parameters
    A : array-like, optional
        Treatment variable to determine if binary or continuous

    Returns
    -------
    function
        A function that trains a model
    """
    if params is None:
        params = {}

    # Determine if treatment is binary
    is_binary = True
    if A is not None:
        is_binary = is_data_binary(A)

    if is_binary:
        # Use classifiers for binary treatments
        if classifier == 'logit':
            return logit_factory(params)
        elif classifier == 'boosting':
            return boosting_factory(params)
        elif classifier == 'sgd':
            return sgd_factory(params)
        elif classifier == 'mlp':
            return mlp_factory(params)
        else:
            raise ValueError(f'Unknown classifier: {classifier}')
    else:
        #  continuous treatments
        if classifier == 'logit':
            return linear_factory(params)
        elif classifier == 'boosting':
            return boosting_cont_factory(params)
        elif classifier == 'sgd':
            return sgd_cont_factory(params)
        elif classifier == 'mlp':
            return mlp_cont_factory(params)
        else:
            raise ValueError(f'Unknown classifier: {classifier}')


def construct_df(data):
    """
    Constructs a DataFrame from permuted and observed data

    Parameters
    ----------
    data : dict
        Dictionary containing permuted and observed data

    Returns
    -------
    pandas.DataFrame
        DataFrame for training
    """
    # Extract dimensions
    n_permuted = len(data['permuted']['A'])
    n_observed = len(data['observed']['A'])
    n_features = data['permuted']['X'].shape[1]

    # Create base features
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

    # Add interactions between A and X - critical for performance
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


# ================= BINARY TREATMENT MODELS =================

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

            probs = np.clip(probs, 0.00001, 0.99999)

            # Compute weights
            weights = probs / (1 - probs)

            return weights

        return weight_function

    return trainer


def boosting_factory(params=None):
    """
    Factory for gradient boosting classifier

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

            probs = np.clip(probs, 0.00001, 0.99999)

            # Compute weights
            weights = probs / (1 - probs)

            return weights

        return weight_function

    return trainer


def sgd_factory(params=None):
    """
    Factory for SGD-based logistic regression trainer

    Parameters
    ----------
    params : dict, optional
        SGD parameters

    Returns
    -------
    function
        A function that trains an SGD-based logistic regression
    """
    if params is None:
        params = {}

    # Set default parameters with better values
    default_params = {
        'loss': 'log_loss',
        'penalty': 'l2',
        'alpha': 0.001,
        'max_iter': 1000,
        'tol': 1e-4,
        'learning_rate': 'adaptive',
        'eta0': 0.01,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1
    }
    # Override defaults with provided params
    model_params = {**default_params, **params}

    def trainer(data):
        """
        Trains an SGD-based logistic regression model

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

        # Identify columns to scale: everything except 'A'
        to_scale_cols = [col for col in X_cols if col != 'A']

        # Apply scaling to features and interactions, but not A
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_train_scaled[to_scale_cols] = scaler.fit_transform(X_train[to_scale_cols])

        model = SGDClassifier(**model_params)
        model.fit(X_train_scaled, y_train)

        # Store convergence information
        converged = model.n_iter_ < model.max_iter
        iterations = model.n_iter_

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

            eval_to_scale_cols = [col for col in eval_df.columns if col != 'A']
            eval_df_scaled = eval_df.copy()
            eval_df_scaled[eval_to_scale_cols] = scaler.transform(eval_df[eval_to_scale_cols])

            # Predict probabilities
            probs = model.predict_proba(eval_df_scaled)[:, 1]

            # Clip probabilities to avoid extreme weights
            probs = np.clip(probs, 0.00001, 0.99999)
            # Compute weights
            weights = probs / (1 - probs)

            # Normalize weights
            weights = weights / np.sum(weights) * len(weights)

            # Attach convergence info
            weight_function.convergence_info = {
                'converged': converged,
                'iterations': iterations
            }

            return weights

        return weight_function

    return trainer


def mlp_factory(params=None):
    """
    Factory for neural network classifier

    Parameters
    ----------
    params : dict, optional
        Neural network parameters

    Returns
    -------
    function
        A function that trains a neural network
    """
    if params is None:
        params = {}

    # Set default parameters with better architecture
    default_params = {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,
        'batch_size': 'auto',
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.005,
        'max_iter': 500,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'random_state': 42
    }

    # Override defaults with provided params
    model_params = {**default_params, **params}

    def trainer(data):
        """
        Trains a neural network model

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

        # Identify columns to scale: everything except 'A'
        to_scale_cols = [col for col in X_cols if col != 'A']

        # Apply scaling to features and interactions, but not A
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_train_scaled[to_scale_cols] = scaler.fit_transform(X_train[to_scale_cols])

        model = MLPClassifier(**model_params)
        model.fit(X_train_scaled, y_train)

        converged = model.n_iter_ < model.max_iter
        iterations = model.n_iter_

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

            # Scale features and interactions, but not A
            eval_to_scale_cols = [col for col in eval_df.columns if col != 'A']
            eval_df_scaled = eval_df.copy()
            eval_df_scaled[eval_to_scale_cols] = scaler.transform(eval_df[eval_to_scale_cols])

            # Predict probabilities
            probs = model.predict_proba(eval_df_scaled)[:, 1]

            # Clip probabilities to avoid extreme weights
            probs = np.clip(probs, 0.00001, 0.99999)
            # Compute weights
            weights = probs / (1 - probs)

            # Normalize weights for stability
            weights = weights / np.sum(weights) * len(weights)

            # Attach convergence info
            weight_function.convergence_info = {
                'converged': converged,
                'iterations': iterations
            }
            return weights

        return weight_function

    return trainer



# ================= CONTINUOUS TREATMENT MODELS =================

def linear_factory(params=None):
    """
    Factory for linear regression trainer for continuous treatments

    Parameters
    ----------
    params : dict, optional
        Linear regression parameters

    Returns
    -------
    function
        A function that trains a linear regression
    """
    if params is None:
        params = {}

    # Set default parameters
    default_params = {
        'fit_intercept': True,
        'n_jobs': -1
    }

    # Override defaults with provided params
    model_params = {**default_params, **params}

    def trainer(data):
        """
        Trains a linear regression model for continuous treatment

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
        model = LogisticRegression(max_iter=2000)
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

            # Predict probabilities using logistic regression
            probs = model.predict_proba(eval_df)[:, 1]

            # Clip probabilities to avoid extreme weights
            probs = np.clip(probs, 0.00001, 0.99999)

            # Compute weights
            weights = probs / (1 - probs)

            # Normalize weights
            # Additional trimming of extreme weights
            ###weights = np.clip(weights, np.percentile(weights, 0.5), np.percentile(weights, 97.5))
            weights = weights / np.sum(weights) * len(weights)
            return weights

        return weight_function

    return trainer


def boosting_cont_factory(params=None):
    """
    Factory for gradient boosting  for continuous treatments

    Parameters
    ----------
    params : dict, optional
        Gradient boosting parameters

    Returns
    -------
    function
        A function that trains a gradient boosting
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
        Trains a gradient boosting regression model

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

            # Clip probabilities to avoid extreme weights
            probs = np.clip(probs, 0.00001, 0.99999)

            # Compute weights
            weights = probs / (1 - probs)

            # Normalize weights
            # Additional trimming of extreme weights
            ###weights = np.clip(weights, np.percentile(weights, 0.5), np.percentile(weights, 97.5))
            weights = weights / np.sum(weights) * len(weights)
            return weights

        return weight_function

    return trainer


def sgd_cont_factory(params=None):
    """
    Factory for SGD-based linear regression trainer for continuous treatments

    Parameters
    ----------
    params : dict, optional
        SGD  parameters

    Returns
    -------
    function
        A function that trains an SGD-based linear regression
    """
    if params is None:
        params = {}

    # Set default parameters
    default_params = {
        'loss': 'log_loss',  # Changed to log_loss
        'penalty': 'l2',
        'alpha': 0.001,
        'max_iter': 1000,
        'tol': 1e-4,
        'learning_rate': 'adaptive',
        'eta0': 0.01,
        'random_state': 42,
        'early_stopping': True
    }

    # Override defaults with provided params
    model_params = {**default_params, **params}

    def trainer(data):
        """
        Trains an SGD-based linear regression model

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

        # Identify columns to scale: everything except 'A'
        to_scale_cols = [col for col in X_cols if col != 'A']

        # Apply scaling to features and interactions, but not A
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_train_scaled[to_scale_cols] = scaler.fit_transform(X_train[to_scale_cols])

        model = SGDClassifier(**model_params)
        model.fit(X_train_scaled, y_train)

        # Store convergence information
        converged = model.n_iter_ < model.max_iter
        iterations = model.n_iter_

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

            eval_to_scale_cols = [col for col in eval_df.columns if col != 'A']
            eval_df_scaled = eval_df.copy()
            eval_df_scaled[eval_to_scale_cols] = scaler.transform(eval_df[eval_to_scale_cols])

            # Predict probabilities
            probs = model.predict_proba(eval_df_scaled)[:, 1]

            # Clip probabilities to avoid extreme weights
            probs = np.clip(probs, 0.00001, 0.99999)

            # Compute weights
            weights = probs / (1 - probs)

            # Normalize weights for stability
            # Additional trimming of extreme weights
            ###weights = np.clip(weights, np.percentile(weights, 0.5), np.percentile(weights, 97.5))
            weights = weights / np.sum(weights) * len(weights)
            # Attach convergence info
            weight_function.convergence_info = {
                'converged': converged,
                'iterations': iterations
            }

            return weights

        return weight_function

    return trainer


def mlp_cont_factory(params=None):
    """
    Factory for neural network  for continuous treatments

    Parameters
    ----------
    params : dict, optional
        Neural network parameters

    Returns
    -------
    function
        A function that trains a neural network
    """
    if params is None:
        params = {}

    # Set default parameters
    default_params = {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,
        'batch_size': 'auto',
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.005,
        'max_iter': 500,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'random_state': 42
    }

    # Override defaults with provided params
    model_params = {**default_params, **params}

    def trainer(data):
        """
        Trains a neural network regression model

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

        # Identify columns to scale: everything except 'A'
        to_scale_cols = [col for col in X_cols if col != 'A']

        # Apply scaling to features and interactions, but not A
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_train_scaled[to_scale_cols] = scaler.fit_transform(X_train[to_scale_cols])

        model = MLPClassifier(**model_params)
        model.fit(X_train_scaled, y_train)

        converged = model.n_iter_ < model.max_iter
        iterations = model.n_iter_

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

            # Scale features and interactions, but not A
            eval_to_scale_cols = [col for col in eval_df.columns if col != 'A']
            eval_df_scaled = eval_df.copy()
            eval_df_scaled[eval_to_scale_cols] = scaler.transform(eval_df[eval_to_scale_cols])

            # Predict probabilities
            probs = model.predict_proba(eval_df_scaled)[:, 1]

            # Clip probabilities to avoid extreme weights
            probs = np.clip(probs, 0.00001, 0.99999)

            # Compute weights
            weights = probs / (1 - probs)

            # Normalize weights for stability
            # Additional trimming of extreme weights
            ###weights = np.clip(weights, np.percentile(weights, 0.5), np.percentile(weights, 97.5))
            weights = weights / np.sum(weights) * len(weights)
            # Attach convergence info
            weight_function.convergence_info = {
                'converged': converged,
                'iterations': iterations
            }

            return weights

        return weight_function

    return trainer