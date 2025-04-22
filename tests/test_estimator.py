import numpy as np
import pytest
from permutation_weighting.estimator import PW


def test_pw_init():
    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Run PW with bootstrap=True to override default behavior
    result = PW(A, X, num_replicates=2, estimand_params={'bootstrap': True})

    # Check result structure
    assert 'weights' in result
    assert 'train' in result
    assert 'call' in result

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)

    # Check training metrics
    assert 'MSEEvaluator' in result['train']
    assert 'LogLossEvaluator' in result['train']

    # Check call info
    assert result['call']['classifier'] == 'logit'
    assert result['call']['estimand'] == 'ATE'
    assert result['call']['num_replicates'] == 2


def test_pw_eval_data():
    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Create eval data
    eval_data = {
        'A': A.copy(),
        'X': X.copy()
    }

    # Run PW with eval data
    result = PW(A, X, eval_data=eval_data, num_replicates=2)

    # Check result structure
    assert 'eval' in result

    # Check eval metrics
    assert 'MSEEvaluator' in result['eval']
    assert 'LogLossEvaluator' in result['eval']


def test_pw_binary_crossproduct():
    # Create simple binary data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    A = np.array([0, 1] * (n // 2))

    # Run PW with binary data (should use crossproduct)
    result = PW(A, X)

    # Since using crossproduct, should only have 1 replicate regardless of num_replicates
    assert result['call']['num_replicates'] == 1

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)


def test_pw_bootstrap():
    # Create simple binary data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    A = np.array([0, 1] * (n // 2))

    # Run PW with bootstrap=True to override crossproduct
    result = PW(A, X, estimand_params={'bootstrap': True}, num_replicates=2)

    # Should use specified num_replicates
    assert result['call']['num_replicates'] == 2

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)


def test_pw_boosting():
    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Run PW with boosting
    result = PW(A, X, classifier='boosting', num_replicates=2)

    # Check classifier in call info
    assert result['call']['classifier'] == 'boosting'

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)


def test_pw_att():
    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Ensure at least one treated unit
    A[0] = 1

    # Run PW with ATT estimand
    result = PW(A, X, estimand='ATT', num_replicates=2)

    # Check estimand in call info
    assert result['call']['estimand'] == 'ATT'

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)


def test_pw_sgd():
    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Run PW with SGD
    result = PW(A, X, classifier='sgd', num_replicates=2)

    # Check classifier in call info
    assert result['call']['classifier'] == 'sgd'

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)


def test_pw_mlp():
    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Custom MLP parameters
    params = {'hidden_layer_sizes': (8, 4), 'max_iter': 100}

    # Run PW with MLP
    result = PW(A, X, classifier='mlp', classifier_params=params, num_replicates=2)

    # Check classifier in call info
    assert result['call']['classifier'] == 'mlp'
    assert result['call']['classifier_params'] == params

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)


def test_pw_continuous_treatment():
    # Create simple data with continuous treatment
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    A = np.random.normal(size=n)  # Continuous treatment

    # Run PW with continuous treatment
    result = PW(A, X, num_replicates=2)

    # Check weights
    assert len(result['weights']) == n
    assert np.all(result['weights'] >= 0)