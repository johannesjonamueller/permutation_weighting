"""
Improved SGD-based trainer factory for permutation weighting.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

class BaseTrainerFactory:
    """Base class for all SGD trainer factories"""

    def __init__(self, params=None):
        """
        Initialize the trainer factory with parameters

        Parameters
        ----------
        params : dict, optional
            Model parameters
        """
        self.params = params or {}
        self.scaler = StandardScaler()

    def _construct_arrays(self, data):
        """
        Constructs numpy arrays from permutation weighting data

        Parameters
        ----------
        data : dict
            Dictionary containing permuted and observed data

        Returns
        -------
        tuple
            (X_train, y_train) tuple of numpy arrays for training
        """
        # Extract data components
        permuted_A = data['permuted']['A']
        permuted_X = data['permuted']['X']
        observed_A = data['observed']['A']
        observed_X = data['observed']['X']

        # Convert to numpy arrays if needed
        if isinstance(permuted_X, pd.DataFrame):
            permuted_X = permuted_X.values
        if isinstance(observed_X, pd.DataFrame):
            observed_X = observed_X.values

        # Create labels
        permuted_y = np.ones(len(permuted_A))
        observed_y = np.zeros(len(observed_A))

        # Combine features: [A, X1, X2, ...]
        permuted_features = np.column_stack([permuted_A, permuted_X])
        observed_features = np.column_stack([observed_A, observed_X])

        # Combine into training data
        X_train = np.vstack([permuted_features, observed_features])
        y_train = np.concatenate([permuted_y, observed_y])

        return X_train, y_train

    def _create_weight_function(self, model, scaler):
        """
        Creates a weight function from a trained model

        Parameters
        ----------
        model : object
            Trained model with predict_proba method
        scaler : StandardScaler
            Fitted scaler for feature standardization

        Returns
        -------
        function
            A function that computes weights
        """
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
            # Convert to numpy arrays if needed
            if isinstance(X, pd.DataFrame):
                X = X.values

            # Create features in the same format as training: [A, X1, X2, ...]
            X_eval = np.column_stack([A, X])

            # Apply scaler if used during training
            if scaler is not None and hasattr(scaler, 'n_features_in_'):
                # Only scale the X part, not the A part
                X_part = X_eval[:, 1:]
                if X_part.shape[1] == scaler.n_features_in_:
                    X_scaled = scaler.transform(X_part)
                    X_eval = np.column_stack([A, X_scaled])

            # Predict probabilities
            probs = model.predict_proba(X_eval)[:, 1]

            # Prevent division by zero by clipping probabilities
            probs = np.clip(probs, 0.001, 0.999)

            # Compute weights
            weights = probs / (1 - probs)

            # Attach convergence info if available
            if hasattr(model, 'n_iter_'):
                weight_function.convergence_info = {
                    'converged': getattr(model, 'n_iter_', 0) < getattr(model, 'max_iter', float('inf')),
                    'iterations': getattr(model, 'n_iter_', 0)
                }

            return weights

        return weight_function

    def __call__(self, data):
        """
        Abstract method to train a model on data

        Parameters
        ----------
        data : dict
            Dictionary containing permuted and observed data

        Returns
        -------
        function
            A function that computes weights
        """
        raise NotImplementedError("Subclasses must implement __call__")


class SGDLogitFactory(BaseTrainerFactory):
    """Factory for SGD-based logistic regression trainer"""

    def __init__(self, params=None):
        """
        Initialize the SGD logistic regression factory

        Parameters
        ----------
        params : dict, optional
            SGD parameters
        """
        super().__init__(params)

        # Set default parameters
        default_params = {
            'loss': 'log_loss',  # Log loss for logistic regression
            'penalty': 'l2',
            'alpha': 0.0001,
            'max_iter': 1000,
            'tol': 1e-3,
            'learning_rate': 'adaptive',
            'eta0': 0.001,
            'random_state': 42,
            'verbose': False,
            'early_stopping': True,
            'validation_fraction': 0.1
        }

        # Override defaults with provided params
        self.model_params = {**default_params, **self.params}

    def __call__(self, data):
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
        # Extract training data
        X_train, y_train = self._construct_arrays(data)

        # Extract and scale just the feature part (not the treatment part)
        X_features = X_train[:, 1:]  # Skip the first column (A)
        self.scaler.fit(X_features)

        # Apply scaler
        X_scaled = np.hstack([
            X_train[:, 0:1],  # Keep A unchanged
            self.scaler.transform(X_features)  # Scale features
        ])

        # Validate model parameters
        if self.model_params.get('max_iter', 1000) <= 0:
            warnings.warn("max_iter must be positive; using default of 1000")
            self.model_params['max_iter'] = 1000

        if self.model_params.get('eta0', 0.001) <= 0:
            warnings.warn("eta0 must be positive; using default of 0.001")
            self.model_params['eta0'] = 0.001

        # Train model with ConvergenceWarning suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = SGDClassifier(**self.model_params)
            model.fit(X_scaled, y_train)

        return self._create_weight_function(model, self.scaler)


class NeuralNetFactory(BaseTrainerFactory):
    """Factory for neural network trainer using SGD"""

    def __init__(self, params=None):
        """
        Initialize the neural network factory

        Parameters
        ----------
        params : dict, optional
            Neural network parameters
        """
        super().__init__(params)

        # Set default parameters
        default_params = {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'shuffle': True,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'verbose': False
        }

        # Override defaults with provided params
        self.model_params = {**default_params, **self.params}

    def __call__(self, data):
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
        # Extract training data
        X_train, y_train = self._construct_arrays(data)

        # Extract and scale just the feature part (not the treatment part)
        X_features = X_train[:, 1:]  # Skip the first column (A)
        self.scaler.fit(X_features)

        # Apply scaler
        X_scaled = np.hstack([
            X_train[:, 0:1],  # Keep A unchanged
            self.scaler.transform(X_features)  # Scale features
        ])

        # Validate parameters
        n_samples = len(X_train)
        if self.model_params.get('validation_fraction', 0.1) >= 1.0:
            warnings.warn("validation_fraction must be < 1.0; using default of 0.1")
            self.model_params['validation_fraction'] = 0.1

        validation_size = int(n_samples * self.model_params.get('validation_fraction', 0.1))
        if validation_size < 10 and self.model_params.get('early_stopping', True):
            warnings.warn(f"Small dataset ({n_samples} samples) with early_stopping=True may not work well; disabling early stopping")
            self.model_params['early_stopping'] = False

        # Train neural network with ConvergenceWarning suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            try:
                model = MLPClassifier(**self.model_params)
                model.fit(X_scaled, y_train)

                # Check if training converged
                converged = model.n_iter_ < model.max_iter
                if not converged:
                    warnings.warn("Neural network did not converge; results may be suboptimal")

            except Exception as e:
                warnings.warn(f"Error training neural network: {e}. Falling back to SGD logistic regression.")
                sgd_params = {
                    'loss': 'log_loss',
                    'max_iter': self.model_params.get('max_iter', 500),
                    'alpha': self.model_params.get('alpha', 0.0001),
                    'random_state': self.model_params.get('random_state', 42)
                }
                model = SGDClassifier(**sgd_params)
                model.fit(X_scaled, y_train)

        return self._create_weight_function(model, self.scaler)


class MinibatchPermuteFactory(BaseTrainerFactory):
    """Factory for minibatch SGD trainer with in-batch permutation"""

    def __init__(self, classifier='logit', params=None, batch_size=128):
        """
        Initialize the minibatch permutation factory

        Parameters
        ----------
        classifier : str, default='logit'
            Type of classifier ('logit' or 'neural_net')
        params : dict, optional
            Classifier parameters
        batch_size : int, default=128
            Size of minibatches
        """
        super().__init__(params)

        # Validate classifier type
        if classifier not in ['logit', 'neural_net']:
            raise ValueError(f"Unknown classifier: {classifier}. Choose from 'logit' or 'neural_net'.")

        self.classifier = classifier

        # Validate batch size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size

        # Configure the classifier based on type
        if classifier == 'logit':
            # Set default parameters for logistic regression
            default_params = {
                'loss': 'log_loss',
                'penalty': 'l2',
                'alpha': 0.0001,
                'max_iter': 1000,
                'tol': 1e-3,
                'learning_rate': 'adaptive',
                'eta0': 0.001,
                'random_state': 42,
                'verbose': False
            }

            # Create model
            self.model_params = {**default_params, **(params or {})}

        elif classifier == 'neural_net':
            # Set default parameters for neural network
            default_params = {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'batch_size': 'auto',
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 500,
                'shuffle': True,
                'random_state': 42,
                'verbose': False
            }

            # Create model
            self.model_params = {**default_params, **(params or {})}

    def __call__(self, data):
        """
        Trains a model using minibatch SGD with in-batch permutation

        Parameters
        ----------
        data : dict
            Dictionary containing observed data

        Returns
        -------
        function
            A function that computes weights
        """
        # Extract observed data
        A = data['observed']['A']
        X = data['observed']['X']
        n = len(A)

        # Convert to numpy if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Fit scaler on input features
        self.scaler.fit(X)

        # Apply scaling
        X_scaled = self.scaler.transform(X)

        # Validate batch size against dataset size
        actual_batch_size = min(self.batch_size, n // 2)
        if actual_batch_size != self.batch_size:
            warnings.warn(f"Batch size {self.batch_size} is too large for dataset size {n}; "
                         f"using batch_size={actual_batch_size} instead")
            self.batch_size = actual_batch_size

        # Ensure batch size is at least 2 for meaningful permutation
        if self.batch_size < 2:
            raise ValueError(f"Dataset size {n} is too small for minibatch permutation. "
                            "Need at least 4 samples.")

        # Enhanced batch processing with better balancing guarantees
        X_train_combined = []
        y_train = []

        # Create stratified sampling indices for treatment groups
        # This ensures better balance in the permutation process
        treatment_values = np.unique(A)
        treatment_indices = {t: np.where(A == t)[0] for t in treatment_values}

        # Calculate target number of samples from each treatment group per batch
        treatment_counts = {t: len(indices) for t, indices in treatment_indices.items()}
        total_samples = sum(treatment_counts.values())
        treatment_proportions = {t: count / total_samples for t, count in treatment_counts.items()}

        # Number of iterations to ensure sufficient training data
        n_iterations = max(50, int(np.ceil(5000 / self.batch_size)))

        # Priority queue for tracking undersampled units
        # We track a priority value for each sample; higher priority = more likely to be selected
        sampling_priority = np.ones(n)

        for iteration in range(n_iterations):
            # Sample batch with stratified treatment representation
            batch_indices = []

            # Target counts for this batch
            target_counts = {t: max(1, int(np.round(self.batch_size * prop)))
                           for t, prop in treatment_proportions.items()}

            # Adjust to ensure we hit the batch size exactly
            total_target = sum(target_counts.values())
            if total_target != self.batch_size:
                # Find the treatment with the largest proportion and adjust its count
                max_t = max(treatment_proportions.items(), key=lambda x: x[1])[0]
                target_counts[max_t] += (self.batch_size - total_target)

            # Sample from each treatment group based on priority
            for t, count in target_counts.items():
                t_indices = treatment_indices[t]
                if len(t_indices) <= count:
                    # If we need all or more samples than available, use all of them
                    selected = t_indices
                else:
                    # Sample based on priority
                    t_priorities = sampling_priority[t_indices]
                    probabilities = t_priorities / np.sum(t_priorities)
                    selected = np.random.choice(t_indices, size=count, replace=False, p=probabilities)

                batch_indices.extend(selected)

                # Decrease priority for selected units (less likely to be sampled in future batches)
                sampling_priority[selected] *= 0.8

            # Shuffle the batch indices
            np.random.shuffle(batch_indices)
            batch_indices = np.array(batch_indices)

            # Get the original batch data
            batch_A = A[batch_indices]
            batch_X = X_scaled[batch_indices]

            # Add original data (label 0)
            features_original = np.column_stack([batch_A, batch_X])
            X_train_combined.append(features_original)
            y_train.extend([0] * len(batch_indices))

            # Create permuted version with sophisticated sampling
            # Always permute within the same batch to maintain feature distributions
            perm_indices = np.random.permutation(len(batch_indices))
            perm_A = batch_A[perm_indices]

            # Add permuted data (label 1)
            features_permuted = np.column_stack([perm_A, batch_X])
            X_train_combined.append(features_permuted)
            y_train.extend([1] * len(batch_indices))

            # Increase priority for all units (more likely to be sampled in future if not selected recently)
            sampling_priority *= 1.05

            # Ensure priorities stay in a reasonable range
            if np.max(sampling_priority) > 100:
                sampling_priority = sampling_priority / np.max(sampling_priority) * 100

        # Combine all batches
        X_train = np.vstack(X_train_combined)
        y_train = np.array(y_train)

        # Train the model with ConvergenceWarning suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            # Initialize the appropriate model
            if self.classifier == 'logit':
                model = SGDClassifier(**self.model_params)
            else:  # neural_net
                model = MLPClassifier(**self.model_params)

            try:
                model.fit(X_train, y_train)
                converged = True
                iterations = getattr(model, 'n_iter_', None)
                if iterations is not None and hasattr(model, 'max_iter'):
                    converged = iterations < model.max_iter
            except Exception as e:
                warnings.warn(f"Error training model: {e}. Falling back to SGD logistic regression.")
                model = SGDClassifier(loss='log_loss', random_state=42)
                model.fit(X_train, y_train)
                converged = True
                iterations = getattr(model, 'n_iter_', 1)

        # Create weight function
        weight_func = self._create_weight_function(model, self.scaler)

        # Add convergence info
        weight_func.convergence_info = {
            'converged': converged,
            'iterations': iterations if iterations is not None else 0
        }

        return weight_func


# Factory function for SGD logistic regression
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
    return SGDLogitFactory(params)


# Factory function for neural network
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
    return NeuralNetFactory(params)


# Factory function for minibatch permutation
def minibatch_permute_trainer_factory(classifier='logit', params=None, batch_size=128):
    """
    Factory for minibatch SGD trainer with in-batch permutation

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
        A function that trains using minibatch SGD with in-batch permutation
    """
    return MinibatchPermuteFactory(classifier, params, batch_size)