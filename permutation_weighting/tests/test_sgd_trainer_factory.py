import os
import numpy as np
import pandas as pd
import pytest
from permutation_weighting.models.sgd_trainer_factory import (
    sgd_logit_factory, neural_net_factory, minibatch_trainer_factory
)


def test_sgd_logit_factory():
    # Create test data
    data = {
        'permuted': {
            'C': 1,
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        },
        'observed': {
            'C': 0,
            'A': np.array([0, 1]),
            'X': np.array([[5, 6], [7, 8]])
        }
    }

    # Test with default parameters
    trainer = sgd_logit_factory()
    model = trainer(data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_sgd_logit_factory_with_params():
    # Create test data
    data = {
        'permuted': {
            'C': 1,
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        },
        'observed': {
            'C': 0,
            'A': np.array([0, 1]),
            'X': np.array([[5, 6], [7, 8]])
        }
    }

    # Test with custom parameters
    custom_params = {
        'alpha': 0.01,
        'learning_rate': 'constant',
        'eta0': 0.1,
        ''
        'max_iter': 100
    }

    trainer = sgd_logit_factory(custom_params)
    model = trainer(data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_neural_net_factory():
    # Create test data
    data = {
        'permuted': {
            'C': 1,
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        },
        'observed': {
            'C': 0,
            'A': np.array([0, 1]),
            'X': np.array([[5, 6], [7, 8]])
        }
    }

    # Test with default parameters
    trainer = neural_net_factory()
    model = trainer(data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_neural_net_factory_with_params():
    # Create test data
    data = {
        'permuted': {
            'C': 1,
            'A': np.array([0, 1]),
            'X': np.array([[1, 2], [3, 4]])
        },
        'observed': {
            'C': 0,
            'A': np.array([0, 1]),
            'X': np.array([[5, 6], [7, 8]])
        }
    }

    # Test with custom parameters
    custom_params = {
        'hidden_layer_sizes': (50, 25),
        'alpha': 0.01,
        'learning_rate_init': 0.01,
        'max_iter': 100
    }

    trainer = neural_net_factory(custom_params)
    model = trainer(data)

    # Test weight computation
    A = np.array([0, 1])
    X = np.array([[1, 2], [3, 4]])

    weights = model(A, X)

    # Check weights
    assert len(weights) == 2
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_minibatch_trainer_factory_logit():
    # Create test data with more samples
    n_samples = 10
    data = {
        'permuted': {
            'C': 1,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': np.random.normal(size=(n_samples, 2))
        },
        'observed': {
            'C': 0,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': np.random.normal(size=(n_samples, 2))
        }
    }

    # Test with logit classifier
    trainer = minibatch_trainer_factory('logit', batch_size=4)
    model = trainer(data)

    # Test weight computation
    A = np.random.binomial(1, 0.5, 5)
    X = np.random.normal(size=(5, 2))

    weights = model(A, X)

    # Check weights
    assert len(weights) == 5
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_minibatch_trainer_factory_neural_net():
    # Create test data with more samples
    n_samples = 10
    data = {
        'permuted': {
            'C': 1,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': np.random.normal(size=(n_samples, 2))
        },
        'observed': {
            'C': 0,
            'A': np.random.binomial(1, 0.5, n_samples),
            'X': np.random.normal(size=(n_samples, 2))
        }
    }

    # Test with neural_net classifier
    trainer = minibatch_trainer_factory('neural_net', batch_size=4)
    model = trainer(data)

    # Test weight computation
    A = np.random.binomial(1, 0.5, 5)
    X = np.random.normal(size=(5, 2))

    weights = model(A, X)

    # Check weights
    assert len(weights) == 5
    assert np.all(weights >= 0)  # Weights should be non-negative
    assert np.all(np.isfinite(weights))  # Weights should be finite


def test_minibatch_trainer_factory_unknown_classifier():
    with pytest.raises(ValueError, match="Unknown classifier"):
        minibatch_trainer_factory('unknown', batch_size=4)


def test_larger_scale():
    """Test SGD trainers with larger datasets to ensure they scale"""
    # Skip this test when running in CI environments
    if "CI" in os.environ:
        pytest.skip("Skipping large-scale test in CI")

    # Create larger test data
    n_samples = 1000
    n_features = 5

    # Generate random data
    np.random.seed(42)
    X = np.random.normal(size=(n_samples, n_features))
    propensity = 1 / (1 + np.exp(X[:, 0] - 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n_samples)

    # Create permutation data structure
    data = {
        'permuted': {
            'C': 1,
            'A': np.random.permutation(A),
            'X': X
        },
        'observed': {
            'C': 0,
            'A': A,
            'X': X
        }
    }

    # Test SGD logistic regression
    sgd_trainer = sgd_logit_factory()
    sgd_model = sgd_trainer(data)
    sgd_weights = sgd_model(A, X)

    # Test minibatch training
    mb_trainer = minibatch_trainer_factory('logit', batch_size=100)
    mb_model = mb_trainer(data)
    mb_weights = mb_model(A, X)

    # Check that weights are properly formed
    assert len(sgd_weights) == n_samples
    assert len(mb_weights) == n_samples
    assert np.all(np.isfinite(sgd_weights))
    assert np.all(np.isfinite(mb_weights))