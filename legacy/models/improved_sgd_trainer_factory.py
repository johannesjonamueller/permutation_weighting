"""
Improved SGD-based trainer factory for permutation weighting.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, roc_auc_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('sgd_trainer_factory')


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
        self.convergence_info = {'converged': False, 'iterations': 0, 'train_loss': [], 'val_loss': []}
        self.debug_mode = self.params.get('debug', False)

        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

    def _construct_arrays(self, data):
        """
        Constructs numpy arrays from permutation weighting data with proper interactions

        Parameters
        ----------
        data : dict
            Dictionary containing permuted and observed data

        Returns
        -------
        tuple
            (X_train, y_train) tuple of numpy arrays for training
        """
        logger.debug("Constructing training arrays")

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

        # Number of features
        n_features = permuted_X.shape[1]

        # Log feature dimensions for debugging
        logger.debug(f"Feature dimensions: {n_features}")
        logger.debug(f"Permuted data: A={permuted_A.shape}, X={permuted_X.shape}")
        logger.debug(f"Observed data: A={observed_A.shape}, X={observed_X.shape}")

        # Create feature arrays with interactions
        permuted_features = np.zeros((len(permuted_A), 1 + n_features + n_features))
        observed_features = np.zeros((len(observed_A), 1 + n_features + n_features))

        # Fill in features: first column is A, then X features, then A*X interactions
        permuted_features[:, 0] = permuted_A
        permuted_features[:, 1:1 + n_features] = permuted_X
        # Create A*X interactions explicitly
        for i in range(n_features):
            permuted_features[:, 1 + n_features + i] = permuted_A * permuted_X[:, i]

        observed_features[:, 0] = observed_A
        observed_features[:, 1:1 + n_features] = observed_X
        # Create A*X interactions explicitly
        for i in range(n_features):
            observed_features[:, 1 + n_features + i] = observed_A * observed_X[:, i]

        # Combine into training data
        X_train = np.vstack([permuted_features, observed_features])
        y_train = np.concatenate([permuted_y, observed_y])

        # Check for NaN or inf values
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            logger.warning("NaN or inf values detected in training data")
            # Replace with zeros to prevent training failures
            X_train = np.nan_to_num(X_train)

        # Log some statistics about the data for debugging
        if self.debug_mode:
            logger.debug(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
            logger.debug(f"Class balance: {np.mean(y_train):.2f}")
            logger.debug(f"Feature stats: min={X_train.min():.4f}, max={X_train.max():.4f}, mean={X_train.mean():.4f}")

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

            # Number of features
            n_features = X.shape[1]

            # Create features in the same format as training: [A, X1, X2, ..., A*X1, A*X2, ...]
            X_eval = np.zeros((len(A), 1 + n_features + n_features))
            X_eval[:, 0] = A
            X_eval[:, 1:1 + n_features] = X
            # Create A*X interactions explicitly
            for i in range(n_features):
                X_eval[:, 1 + n_features + i] = A * X[:, i]

            # Apply scaling if trained with scaling
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ > 0:
                # Only scale the X part and the interaction part, not the A part
                X_scaled = X_eval.copy()
                X_part = X_eval[:, 1:]  # Skip the A column
                if X_part.shape[1] == scaler.n_features_in_:
                    X_scaled[:, 1:] = scaler.transform(X_part)
                    X_eval = X_scaled
                else:
                    logger.warning(
                        f"Feature dimension mismatch: X_part={X_part.shape[1]}, scaler={scaler.n_features_in_}")

            # Predict probabilities
            # Try to handle potential errors in prediction
            try:
                probs = model.predict_proba(X_eval)[:, 1]
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                # Fall back to a safe default
                probs = np.ones(len(A)) * 0.5

            # Prevent division by zero by clipping probabilities
            # Use a slightly narrower range than before for numerical stability
            probs = np.clip(probs, 0.01, 0.99)

            # Check if probabilities are reasonable
            if self.debug_mode:
                logger.debug(
                    f"Probability stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
                # Check if most probabilities are near 0.5 (which would produce weights near 1)
                near_half = np.mean((probs > 0.45) & (probs < 0.55))
                if near_half > 0.9:  # If more than 90% of probabilities are near 0.5
                    logger.warning(f"Most probabilities ({near_half:.1%}) are near 0.5, suggesting poor classification")

            # Compute weights
            weights = probs / (1 - probs)

            # Normalize weights to sum to n (standard practice)
            weights = weights / np.sum(weights) * len(weights)

            # Check for extreme weights
            if np.max(weights) > 10 * np.median(weights):
                logger.warning("Extreme weights detected")

            # Attach convergence info if available
            weight_function.convergence_info = self.convergence_info

            # Log weight statistics
            if self.debug_mode:
                logger.debug(
                    f"Weight stats: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
                logger.debug(f"Proportion of weights near 1.0: {np.mean(np.abs(weights - 1) < 0.1):.2%}")

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

        # Set default parameters with improved defaults
        default_params = {
            'loss': 'log_loss',  # Log loss for logistic regression
            'penalty': 'l2',  # L2 regularization
            'alpha': 0.0001,  # Regularization strength
            'max_iter': 1000,  # Increased from default
            'tol': 1e-4,  # Slightly tighter tolerance
            'learning_rate': 'adaptive',  # Important: must be a string
            'eta0': 0.01,  # Higher initial learning rate
            'random_state': 42,
            'verbose': 0,
            'early_stopping': True,
            'validation_fraction': 0.2,  # Increased validation fraction
            'n_iter_no_change': 10,
            'warm_start': True  # Enable warm start for continued training
        }

        # Override defaults with provided params
        self.model_params = {**default_params, **self.params}

        # Log the final parameters used
        logger.debug(f"SGD parameters: {self.model_params}")

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
        logger.info("Training SGD logistic regression model")

        # Extract training data with interactions
        X_train, y_train = self._construct_arrays(data)

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.model_params.get('validation_fraction', 0.2),
            random_state=self.model_params.get('random_state', 42),
            stratify=y_train  # Ensure balanced classes in both sets
        )

        # Standardize features (excluding the first column which is A)
        # Scale both X features and A*X interaction terms
        self.scaler.fit(X_train[:, 1:])
        X_train_scaled = X_train.copy()
        X_train_scaled[:, 1:] = self.scaler.transform(X_train[:, 1:])

        X_val_scaled = X_val.copy()
        X_val_scaled[:, 1:] = self.scaler.transform(X_val[:, 1:])

        # Check that parameters are valid
        if not isinstance(self.model_params.get('learning_rate', 'adaptive'), str):
            logger.warning("learning_rate must be a string; using 'adaptive' instead")
            self.model_params['learning_rate'] = 'adaptive'

        # Initialize model with more robust parameters
        model = SGDClassifier(**self.model_params)

        # Initialize convergence monitoring
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.model_params.get('n_iter_no_change', 10)
        best_model = None

        # Track loss history
        train_losses = []
        val_losses = []

        # Train with explicit epochs and early stopping
        n_epochs = self.model_params.get('max_iter', 1000)
        batch_size = min(self.model_params.get('batch_size', 32), len(X_train))

        # Suppress ConvergenceWarning during training
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            try:
                for epoch in range(n_epochs):
                    # Train for one epoch
                    model.partial_fit(X_train_scaled, y_train, classes=np.array([0, 1]))

                    # Calculate training and validation loss
                    train_proba = model.predict_proba(X_train_scaled)
                    train_loss = log_loss(y_train, train_proba)
                    train_losses.append(train_loss)

                    val_proba = model.predict_proba(X_val_scaled)
                    val_loss = log_loss(y_val, val_proba)
                    val_losses.append(val_loss)

                    # Calculate AUC for monitoring
                    val_auc = roc_auc_score(y_val, val_proba[:, 1])

                    # Log progress
                    if epoch % 100 == 0 or epoch == n_epochs - 1:
                        logger.debug(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, "
                                     f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

                    # Check if validation loss improved
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save the best model
                        best_model = clone_sgd_classifier(model)
                    else:
                        patience_counter += 1

                    # Early stopping
                    if self.model_params.get('early_stopping', True) and patience_counter >= max_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

                # Use the best model if available
                if best_model is not None:
                    model = best_model

                # Check final model performance
                final_val_proba = model.predict_proba(X_val_scaled)
                final_val_auc = roc_auc_score(y_val, final_val_proba[:, 1])

                if final_val_auc < 0.55:  # AUC barely better than random
                    logger.warning(f"Model performance is poor: AUC = {final_val_auc:.4f}")

                # Update convergence info
                self.convergence_info = {
                    'converged': patience_counter < max_patience,
                    'iterations': epoch + 1,
                    'train_loss': train_losses,
                    'val_loss': val_losses,
                    'final_val_auc': final_val_auc
                }

            except Exception as e:
                logger.error(f"Error during SGD training: {e}")
                # Return a simple model that just predicts the base rate
                model = create_fallback_classifier(y_train)

        logger.info(f"SGD training completed. Iterations: {self.convergence_info['iterations']}")
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

        # Set default parameters with improved defaults
        default_params = {
            'hidden_layer_sizes': (64, 32),  # Larger network
            'activation': 'relu',
            'solver': 'adam',  # Adam optimizer for better convergence
            'alpha': 0.0001,  # L2 regularization
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 1000,  # Increased iterations
            'shuffle': True,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 10,
            'verbose': False
        }

        # Override defaults with provided params
        self.model_params = {**default_params, **self.params}

        # Log the final parameters used
        logger.debug(f"Neural network parameters: {self.model_params}")

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
        logger.info("Training neural network model")

        # Extract training data with interactions
        X_train, y_train = self._construct_arrays(data)

        # Standardize features (excluding the first column which is A)
        self.scaler.fit(X_train[:, 1:])
        X_train_scaled = X_train.copy()
        X_train_scaled[:, 1:] = self.scaler.transform(X_train[:, 1:])

        # Validate parameters
        n_samples = len(X_train)
        if self.model_params.get('validation_fraction', 0.2) >= 1.0:
            logger.warning("validation_fraction must be < 1.0; using default of 0.2")
            self.model_params['validation_fraction'] = 0.2

        validation_size = int(n_samples * self.model_params.get('validation_fraction', 0.2))
        if validation_size < 10 and self.model_params.get('early_stopping', True):
            logger.warning(
                f"Small dataset ({n_samples} samples) with early_stopping=True may not work well; increasing patience")
            self.model_params['n_iter_no_change'] = max(20, self.model_params.get('n_iter_no_change', 10))

        # Train neural network with ConvergenceWarning suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            try:
                logger.debug("Initializing neural network")
                model = MLPClassifier(**self.model_params)

                # Track training progress
                train_losses = []
                val_losses = []
                best_val_loss = float('inf')
                best_model = None
                patience_counter = 0
                max_patience = self.model_params.get('n_iter_no_change', 10)

                # Split data for manual validation if early stopping is enabled
                if self.model_params.get('early_stopping', True):
                    X_train_split, X_val, y_train_split, y_val = train_test_split(
                        X_train_scaled, y_train,
                        test_size=self.model_params.get('validation_fraction', 0.2),
                        random_state=self.model_params.get('random_state', 42),
                        stratify=y_train
                    )
                else:
                    X_train_split, X_val, y_train_split, y_val = X_train_scaled, None, y_train, None

                # Train with partial_fit to monitor convergence
                batch_size = self.model_params.get('batch_size', 'auto')
                if batch_size == 'auto':
                    batch_size = min(200, len(X_train_split))
                else:
                    batch_size = min(batch_size, len(X_train_split))

                n_epochs = self.model_params.get('max_iter', 1000)

                # Initial fit to initialize the model
                logger.debug("Starting neural network training")
                model.fit(X_train_split, y_train_split)

                # Check if training converged
                converged = model.n_iter_ < model.max_iter
                if not converged:
                    logger.warning("Neural network did not converge within max_iter; results may be suboptimal")

                # Evaluate final model
                if X_val is not None:
                    val_proba = model.predict_proba(X_val)
                    final_val_auc = roc_auc_score(y_val, val_proba[:, 1])

                    if final_val_auc < 0.55:  # AUC barely better than random
                        logger.warning(f"Model performance is poor: AUC = {final_val_auc:.4f}")

                    # Store the validation AUC
                    self.convergence_info['final_val_auc'] = final_val_auc

                # Update convergence info
                self.convergence_info = {
                    'converged': converged,
                    'iterations': model.n_iter_,
                    'train_loss': model.loss_curve_ if hasattr(model, 'loss_curve_') else [],
                    'val_loss': []  # MLPClassifier doesn't store validation loss
                }

            except Exception as e:
                logger.error(f"Error training neural network: {e}")
                # Fall back to a simple logistic regression
                logger.warning("Falling back to SGD logistic regression")
                sgd_params = {
                    'loss': 'log_loss',
                    'max_iter': self.model_params.get('max_iter', 1000),
                    'alpha': self.model_params.get('alpha', 0.0001),
                    'random_state': self.model_params.get('random_state', 42),
                    'learning_rate': 'adaptive'  # Must be a string
                }
                model = SGDClassifier(**sgd_params)
                model.fit(X_train_scaled, y_train)

        logger.info(f"Neural network training completed. Iterations: {self.convergence_info.get('iterations', 0)}")
        return self._create_weight_function(model, self.scaler)


# Helper functions

def clone_sgd_classifier(model):
    """Clone an SGDClassifier to save the best model during training"""
    clone = SGDClassifier()
    clone.set_params(**model.get_params())
    clone.coef_ = model.coef_.copy()
    clone.intercept_ = model.intercept_.copy()
    if hasattr(model, 't_'):
        clone.t_ = model.t_
    if hasattr(model, 'loss_function_'):
        clone.loss_function_ = model.loss_function_
    return clone


def create_fallback_classifier(y_train):
    """Create a simple classifier that predicts the base rate"""

    class FallbackClassifier:
        def __init__(self, y):
            self.base_rate = np.mean(y)

        def predict_proba(self, X):
            n_samples = X.shape[0]
            proba = np.zeros((n_samples, 2))
            proba[:, 0] = 1 - self.base_rate
            proba[:, 1] = self.base_rate
            return proba

    return FallbackClassifier(y_train)


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