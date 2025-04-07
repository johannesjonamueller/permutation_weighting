import os
import numpy as np
import pandas as pd
import pytest
from permutation_weighting.models.sgd_trainer_factory import (
    sgd_logit_factory, neural_net_factory, minibatch_permute_trainer_factory,
    SGDLogitFactory, NeuralNetFactory, MinibatchPermuteFactory
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from permutation_weighting.estimator import PW


# Test data fixture
@pytest.fixture
def test_data():
    """Create test data for trainer factory tests"""
    # Create test data with consistent dimensions
    n_samples = 10
    np.random.seed(42)  # For reproducibility
    X = np.random.normal(size=(n_samples, 2))
    data = {
        'permuted': {
            'C': 1,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': X.copy()
        },
        'observed': {
            'C': 0,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': X.copy()
        }
    }
    return data


# Test data with DataFrame X
@pytest.fixture
def test_data_df():
    """Create test data with DataFrame X for trainer factory tests"""
    # Create test data with consistent dimensions
    n_samples = 10
    np.random.seed(42)  # For reproducibility
    X = np.random.normal(size=(n_samples, 2))

    # Create X as DataFrame
    X_permuted = pd.DataFrame(X.copy(), columns=['feature1', 'feature2'])
    X_observed = pd.DataFrame(X.copy(), columns=['feature1', 'feature2'])

    data = {
        'permuted': {
            'C': 1,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': X_permuted
        },
        'observed': {
            'C': 0,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': X_observed
        }
    }
    return data


def test_sgd_logit_factory(test_data):
    """Test SGD logistic regression factory"""
    # Test with default parameters
    trainer = sgd_logit_factory()
    model = trainer(test_data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_sgd_logit_factory_with_df(test_data_df):
    """Test SGD logistic regression factory with DataFrame input"""
    # Test with default parameters
    trainer = sgd_logit_factory()
    model = trainer(test_data_df)

    # Test weight computation with DataFrame
    A = np.array([0, 1])
    X = pd.DataFrame([[1, 2], [3, 4]], columns=['feature1', 'feature2'])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_sgd_logit_factory_with_params(test_data):
    """Test SGD logistic regression factory with custom parameters"""
    # Test with custom parameters
    custom_params = {
        'alpha': 0.01,
        'learning_rate': 'constant',
        'eta0': 0.1,
        'max_iter': 100
    }

    trainer = sgd_logit_factory(custom_params)
    model = trainer(test_data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_neural_net_factory(test_data):
    """Test neural network factory"""
    # Test with default parameters
    trainer = neural_net_factory()
    model = trainer(test_data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_neural_net_factory_with_params(test_data):
    """Test neural network factory with custom parameters"""
    # Test with custom parameters
    custom_params = {
        'hidden_layer_sizes': (5, 3),
        'alpha': 0.01,
        'learning_rate_init': 0.01,
        'max_iter': 100
    }

    trainer = neural_net_factory(custom_params)
    model = trainer(test_data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_minibatch_permute_factory(test_data):
    """Test minibatch permutation factory"""
    # Test with logit classifier
    trainer = minibatch_permute_trainer_factory('logit', batch_size=4)
    model = trainer(test_data)

    # Test weight computation using the same feature dimensions as in training
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_minibatch_permute_neural_net(test_data):
    """Test minibatch permutation factory with neural network"""
    # Test with neural_net classifier
    trainer = minibatch_permute_trainer_factory('neural_net', batch_size=4)
    model = trainer(test_data)

    # Test weight computation using the same feature dimensions as in training
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_minibatch_permute_factory_unknown_classifier():
    """Test minibatch permutation factory with unknown classifier"""
    with pytest.raises(ValueError, match="Unknown classifier"):
        minibatch_permute_trainer_factory('unknown', batch_size=4)


def test_parameter_validation():
    """Test parameter validation in factories"""
    # Create test data with invalid batch size
    with pytest.raises(ValueError, match="batch_size must be positive"):
        MinibatchPermuteFactory('logit', batch_size=-1)

    # Test with very small dataset
    small_data = {
        'observed': {
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        }
    }

    # Should raise error for dataset that's too small
    with pytest.raises(ValueError, match="Dataset size .* is too small"):
        trainer = minibatch_permute_trainer_factory('logit', batch_size=10)
        model = trainer(small_data)


def test_class_based_interface():
    """Test the class-based interface"""
    # Create test data
    n_samples = 10
    np.random.seed(42)
    X = np.random.normal(size=(n_samples, 2))
    data = {
        'permuted': {
            'C': 1,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': X.copy()
        },
        'observed': {
            'C': 0,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': X.copy()
        }
    }

    # Test SGDLogitFactory directly
    trainer = SGDLogitFactory(params={'alpha': 0.01})
    model = trainer(data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)
    assert np.all(np.isfinite(weights))

    # Test NeuralNetFactory directly
    trainer = NeuralNetFactory(params={'hidden_layer_sizes': (5,)})
    model = trainer(data)

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)
    assert np.all(np.isfinite(weights))


def test_convergence_tracking():
    """Test that convergence information is properly tracked"""
    # Create test data
    n_samples = 20
    np.random.seed(42)
    X = np.random.normal(size=(n_samples, 2))
    data = {
        'permuted': {
            'C': 1,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': X.copy()
        },
        'observed': {
            'C': 0,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': X.copy()
        }
    }

    # Test with very small max_iter to force non-convergence
    trainer = sgd_logit_factory({'max_iter': 1})
    model = trainer(data)

    # Test weight computation with the same features as in training
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check convergence info is attached to the weight function
    assert hasattr(model, 'convergence_info')

    # Test minibatch convergence info
    trainer = minibatch_permute_trainer_factory('logit', batch_size=5)
    model = trainer(data)

    weights = model(A, X)

    # Check convergence info
    assert hasattr(model, 'convergence_info')
    assert 'converged' in model.convergence_info
    assert 'iterations' in model.convergence_info


def test_with_estimator():
    """Test integration with PW estimator"""
    # Generate sample data with consistent dimensions
    np.random.seed(42)
    n = 50
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)
    Y = X[:, 0] + A * 2 + np.random.normal(size=n)

    # Run PW with SGD logit
    result_sgd = PW(
        A=A,
        X=X,
        classifier='logit',
        use_sgd=True,
        num_replicates=1,
        classifier_params={'alpha': 0.01}
    )

    # Run PW with neural network
    result_nn = PW(
        A=A,
        X=X,
        classifier='neural_net',
        use_sgd=True,
        num_replicates=1,
        classifier_params={'hidden_layer_sizes': (5,)}
    )

    # Run PW with minibatch permutation
    result_mini = PW(
        A=A,
        X=X,
        classifier='logit',
        use_sgd=True,
        batch_size=10,
        num_replicates=1
    )

    # Check that all approaches produce reasonable weights
    assert 'weights' in result_sgd
    assert 'weights' in result_nn
    assert 'weights' in result_mini

    assert len(result_sgd['weights']) == n
    assert len(result_nn['weights']) == n
    assert len(result_mini['weights']) == n

    assert np.all(result_sgd['weights'] >= 0)
    assert np.all(result_nn['weights'] >= 0)
    assert np.all(result_mini['weights'] >= 0)

    # Check convergence info
    assert 'convergence_info' in result_sgd
    assert 'convergence_info' in result_nn
    assert 'convergence_info' in result_mini


@pytest.mark.skipif("CI" in os.environ, reason="Skipping large-scale test in CI")
def test_larger_scale():
    """Test SGD trainers with larger datasets to ensure they scale"""
    # Create larger test data with consistent dimensions
    n_samples = 500  # Reduced for faster testing
    n_features = 5

    # Generate random data with consistent dimensions
    np.random.seed(42)
    X = np.random.normal(size=(n_samples, n_features))
    propensity = 1 / (1 + np.exp(X[:, 0] - 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n_samples)

    # Create permutation data structure
    data = {
        'permuted': {
            'C': 1,
            'A': np.random.permutation(A),
            'X': X.copy()
        },
        'observed': {
            'C': 0,
            'A': A,
            'X': X.copy()
        }
    }

    # Test SGD logistic regression
    sgd_trainer = sgd_logit_factory()
    sgd_model = sgd_trainer(data)

    # Create evaluation data with the same dimensions as training
    eval_A = np.random.binomial(1, 0.5, 10)
    eval_X = np.random.normal(size=(10, n_features))

    sgd_weights = sgd_model(eval_A, eval_X)

    # Test minibatch training
    mb_trainer = minibatch_permute_trainer_factory('logit', batch_size=50)
    mb_model = mb_trainer(data)
    mb_weights = mb_model(eval_A, eval_X)

    # Check that weights are properly formed
    assert len(sgd_weights) == 10
    assert len(mb_weights) == 10
    assert np.all(np.isfinite(sgd_weights))
    assert np.all(np.isfinite(mb_weights))


