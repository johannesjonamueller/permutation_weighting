"""
Trainer factory for permutation weighting with batch-then-permute support.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .data_validation import is_data_binary


def get_trainer_factory(classifier, params=None):
    """
    Factory function for creating trainers based on treatment type

    Parameters
    ----------
    classifier : str
        Type of model ('logit', 'boosting', 'sgd', 'mlp')
    params : dict, optional
        Model parameters
    Returns
    -------
    function
        A function that trains a model
    """
    if params is None:
        params = {}

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

def create_batches(data, batch_size):
    """
    Create mini-batches from data while preserving distribution characteristics

    Parameters
    ----------
    data : dict
        Dictionary containing permuted and observed data
    batch_size : int
        Size of each mini-batch

    Returns
    -------
    list
        List of data batches with the same structure as the input data
    """
    # Calculate total sizes
    n_permuted = len(data['permuted']['A'])
    n_observed = len(data['observed']['A'])

    # If batch_size is larger than dataset or not specified, return original data
    if batch_size is None or batch_size >= n_permuted + n_observed:
        return [data]

    # Calculate number of samples per batch while maintaining the ratio
    n_total = n_permuted + n_observed
    n_batches = max(1, n_total // batch_size)

    # Calculate proportions for permuted and observed
    perm_ratio = n_permuted / n_total
    obs_ratio = n_observed / n_total

    perm_per_batch = max(1, int(batch_size * perm_ratio))
    obs_per_batch = max(1, batch_size - perm_per_batch)

    # Create random indices for shuffling
    perm_indices = np.random.permutation(n_permuted)
    obs_indices = np.random.permutation(n_observed)

    batches = []

    # Create batches with proper distribution
    for i in range(n_batches):
        perm_start = (i * perm_per_batch) % n_permuted
        perm_end = min(perm_start + perm_per_batch, n_permuted)
        perm_idx = perm_indices[perm_start:perm_end]

        obs_start = (i * obs_per_batch) % n_observed
        obs_end = min(obs_start + obs_per_batch, n_observed)
        obs_idx = obs_indices[obs_start:obs_end]

        batch = {
            'permuted': {
                'C': data['permuted']['C'],
                'A': data['permuted']['A'][perm_idx],
                'X': data['permuted']['X'][perm_idx]
            },
            'observed': {
                'C': data['observed']['C'],
                'A': data['observed']['A'][obs_idx],
                'X': data['observed']['X'][obs_idx]
            }
        }

        batches.append(batch)

    return batches

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
        'C': 10.0,
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
        'max_depth': 2,
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

    # Extract batch_size but don't include it in model_params
    batch_size = params.pop('permute_batch_size', None) if params else None

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
        # Create batches if batch_size is specified
        batches = create_batches(data, batch_size)

        # Initialize model
        model = SGDClassifier(**model_params, warm_start=True)

        # Initialize scaler - we'll fit it on the full dataset first
        full_df = construct_df(data)
        X_cols = [col for col in full_df.columns if col != 'C']
        to_scale_cols = [col for col in X_cols if col != 'A']

        scaler = StandardScaler()
        full_df[to_scale_cols] = scaler.fit_transform(full_df[to_scale_cols])

        # Train on each batch sequentially
        converged = True
        iterations = 0

        for batch in batches:
            df = construct_df(batch)

            # Separate features and target
            X_cols = [col for col in df.columns if col != 'C']
            X_train = df[X_cols]
            y_train = df['C']

            # Apply scaling
            to_scale_cols = [col for col in X_cols if col != 'A']
            X_train_scaled = X_train.copy()
            X_train_scaled[to_scale_cols] = scaler.transform(X_train[to_scale_cols])

            # Train on this batch
            model.fit(X_train_scaled, y_train)

            # Update convergence info
            converged = converged and model.n_iter_ < model.max_iter
            iterations = max(iterations, model.n_iter_)

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
        'batch_size': 'auto',  # This is sklearn's internal batch size
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.005,
        'max_iter': 500,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'random_state': 42
    }

    # Extract our custom batch_size parameter
    custom_batch_size = params.pop('permute_batch_size', None) if params else None

    # Override defaults with provided params
    model_params = {**default_params, **params}

    def trainer(data):
        """
        Trains a neural network model with sequential batch training

        Parameters
        ----------
        data : dict
            Dictionary containing permuted and observed data

        Returns
        -------
        function
            A function that computes weights
        """
        # Create batches if batch_size is specified
        batches = create_batches(data, custom_batch_size)

        # Initialize scaler - we'll fit it on the full dataset first
        full_df = construct_df(data)
        X_cols = [col for col in full_df.columns if col != 'C']
        to_scale_cols = [col for col in X_cols if col != 'A']

        scaler = StandardScaler()
        full_df[to_scale_cols] = scaler.fit_transform(full_df[to_scale_cols])

        # Initialize model
        model = MLPClassifier(**model_params, warm_start=True)

        # Train on each batch sequentially
        converged = True
        iterations = 0

        for batch in batches:
            df = construct_df(batch)

            # Separate features and target
            X_cols = [col for col in df.columns if col != 'C']
            X_train = df[X_cols]
            y_train = df['C']

            # Apply scaling
            to_scale_cols = [col for col in X_cols if col != 'A']
            X_train_scaled = X_train.copy()
            X_train_scaled[to_scale_cols] = scaler.transform(X_train[to_scale_cols])

            # Train on this batch
            model.fit(X_train_scaled, y_train)

            # Update convergence info
            converged = converged and model.n_iter_ < model.max_iter
            iterations = max(iterations, model.n_iter_)

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

´