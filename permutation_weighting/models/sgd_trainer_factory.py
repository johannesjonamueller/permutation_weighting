"""
SGD-based trainer factory for permutation weighting.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .trainer_factory import construct_df, construct_eval_df


def sgd_logit_factory(params=None):
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

    # Set default parameters
    default_params = {
        'loss': 'log_loss',  # Log loss for logistic regression
        'penalty': 'l2',
        'alpha': 0.0001,
        'max_iter': 1000,
        'tol': 1e-3,
        'learning_rate': 'optimal',
        'eta0': 0.01,
        'random_state': 42
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

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        model = SGDClassifier(**model_params)
        model.fit(X_train_scaled, y_train)

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

            # Standardize features
            X_eval_scaled = scaler.transform(eval_df)

            # Predict probabilities
            probs = model.predict_proba(X_eval_scaled)[:, 1]

            # Prevent division by zero by clipping probabilities
            probs = np.clip(probs, 0.001, 0.999)  # Clip to avoid 0 or 1 #TODO: Exlain to Drew

            # Compute weights
            weights = probs / (1 - probs)

            return weights

        return weight_function

    return trainer

    # In sgd_trainer_factory.py, update neural_net_factory:


def neural_net_factory(params=None):
    """
    Factory for neural network trainer using SGD

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
        'hidden_layer_sizes': (100,),
        'activation': 'relu',
        'solver': 'adam',  # Adam is a variant of SGD
        'alpha': 0.0001,
        'batch_size': 'auto',
        'learning_rate': 'constant',
        'learning_rate_init': 0.001,
        'max_iter': 500,
        'shuffle': True,
        'random_state': 42,
        'early_stopping': False,  # Disabled for small datasets
        'validation_fraction': 0.1,
        'n_iter_no_change': 10
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

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        model = MLPClassifier(**model_params)

        # Track convergence information
        converged = True
        iterations = 0

        try:
            model.fit(X_train_scaled, y_train)
            # Check if model converged
            iterations = model.n_iter_
            converged = iterations < model_params['max_iter']
        except Exception as e:
            print(f"Warning: Neural network training failed: {e}")
            # Fallback to a simpler model if training fails
            model = LogisticRegression()
            model.fit(X_train_scaled, y_train)
            converged = False

        # Store convergence information to be accessed later
        convergence_info = {
            'converged': converged,
            'iterations': iterations
        }

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

            # Standardize features
            X_eval_scaled = scaler.transform(eval_df)

            # Predict probabilities
            probs = model.predict_proba(X_eval_scaled)[:, 1]

            # Prevent division by zero by clipping probabilities
            probs = np.clip(probs, 0.001, 0.999)

            # Compute weights
            weights = probs / (1 - probs)

            # Attach convergence info to the weight function
            weight_function.convergence_info = convergence_info

            return weights

        return weight_function

    return trainer


def minibatch_trainer_factory(classifier='logit', params=None, batch_size=128):
    """
    Factory for minibatch SGD trainer

    Parameters
    ----------
    classifier : str, default='logit'
        Type of classifier ('logit' or 'neural_net')
    params : dict, optional
        Classifier parameters
    batch_size : int, default=128
        Size of minibatches

    Returns
    -------
    function
        A function that trains using minibatch SGD
    """
    if params is None:
        params = {}

    if classifier == 'logit':
        base_trainer = sgd_logit_factory(params)
    elif classifier == 'neural_net':
        base_trainer = neural_net_factory(params)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    def trainer(data):
        """
        Trains a model using minibatch SGD

        Parameters
        ----------
        data : dict
            Dictionary containing permuted and observed data

        Returns
        -------
        function
            A function that computes weights
        """
        # Extract data
        permuted_A = data['permuted']['A']
        permuted_X = data['permuted']['X']
        observed_A = data['observed']['A']
        observed_X = data['observed']['X']

        # Determine number of samples
        n_permuted = len(permuted_A)
        n_observed = len(observed_A)

        # Create indices
        permuted_indices = np.arange(n_permuted)
        observed_indices = np.arange(n_observed)

        # Shuffle indices
        np.random.shuffle(permuted_indices)
        np.random.shuffle(observed_indices)

        # Create minibatches
        n_batches = max(n_permuted, n_observed) // (batch_size // 2)

        # Process in minibatches
        batched_data = []
        for i in range(n_batches):
            # Get batch indices
            p_start = (i * batch_size // 2) % n_permuted
            p_end = min(p_start + batch_size // 2, n_permuted)
            p_idx = permuted_indices[p_start:p_end]

            o_start = (i * batch_size // 2) % n_observed
            o_end = min(o_start + batch_size // 2, n_observed)
            o_idx = observed_indices[o_start:o_end]

            # Create batch data
            batch_data = {
                'permuted': {
                    'C': 1,
                    'A': permuted_A[p_idx],
                    'X': permuted_X[p_idx] if isinstance(permuted_X, np.ndarray) else permuted_X.iloc[p_idx]
                },
                'observed': {
                    'C': 0,
                    'A': observed_A[o_idx],
                    'X': observed_X[o_idx] if isinstance(observed_X, np.ndarray) else observed_X.iloc[o_idx]
                }
            }

            batched_data.append(batch_data)

        # Train on each batch
        trained_models = []
        for batch_data in batched_data:
            trained_models.append(base_trainer(batch_data))

        def weight_function(A, X):
            """
            Computes weights as average of weights from all trained models

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
            # Compute weights for each model
            all_weights = []
            for model in trained_models:
                all_weights.append(model(A, X))

            # Average weights
            avg_weights = np.mean(np.column_stack(all_weights), axis=1)

            return avg_weights

        return weight_function

    return trainer


