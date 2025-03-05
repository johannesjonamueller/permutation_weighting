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
    # Create simple comparison_with_r_package.ipynb data
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
    # Create simple binary comparison_with_r_package.ipynb data
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
    # Create simple binary comparison_with_r_package.ipynb data
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
    # Create simple comparison_with_r_package.ipynb data
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
    # Create simple comparison_with_r_package.ipynb data
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


#New test for torch #TODO: Better understand the tests


def test_pw_with_torch_logistic():
    """Test PW with PyTorch logistic regression"""
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        pytest.skip("PyTorch not available, skipping test")

    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Run PW with PyTorch logistic regression
    result = PW(A, X, classifier='logistic', use_torch=True,
                estimand_params={'bootstrap': True}, num_replicates=2)

    # Check result structure
    assert 'weights' in result
    assert 'train' in result
    assert 'call' in result

    # Check weights
    assert len(result['weights']) == n
    assert np.all(np.isfinite(result['weights']))  # No NaN or Inf
    assert np.all(result['weights'] >= 0)

    # Check call info
    assert result['call']['classifier'] == 'logistic'
    assert result['call']['use_torch'] is True
    assert result['call']['num_replicates'] == 2


def test_pw_with_torch_mlp():
    """Test PW with PyTorch MLP"""
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        pytest.skip("PyTorch not available, skipping test")

    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Custom MLP parameters
    params = {'hidden_dims': [8, 4], 'epochs': 5, 'batch_size': 4}

    # Run PW with PyTorch MLP
    result = PW(A, X, classifier='mlp', use_torch=True,
                classifier_params=params,
                estimand_params={'bootstrap': True}, num_replicates=2)

    # Check result
    assert 'weights' in result
    assert len(result['weights']) == n
    assert np.all(np.isfinite(result['weights']))
    assert np.all(result['weights'] >= 0)

    # Check call info
    assert result['call']['classifier'] == 'mlp'
    assert result['call']['use_torch'] is True
    assert result['call']['classifier_params'] == params


def test_pw_with_torch_custom():
    """Test PW with custom PyTorch configuration"""
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        pytest.skip("PyTorch not available, skipping test")

    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    propensity = 1 / (1 + np.exp(-X[:, 0] + 0.5 * X[:, 1]))
    A = np.random.binomial(1, propensity, size=n)

    # Custom parameters with model type specification
    params = {
        'model_type': 'resnet',
        'hidden_dim': 8,
        'num_blocks': 1,
        'epochs': 5,
        'batch_size': 4
    }

    # Run PW with custom PyTorch configuration
    result = PW(A, X, classifier='torch_custom', use_torch=True,
                classifier_params=params,
                estimand_params={'bootstrap': True}, num_replicates=2)

    # Check result
    assert 'weights' in result
    assert len(result['weights']) == n
    assert np.all(np.isfinite(result['weights']))
    assert np.all(result['weights'] >= 0)

    # Check call info
    assert result['call']['classifier'] == 'torch_custom'
    assert result['call']['use_torch'] is True
    assert result['call']['classifier_params'] == params


def test_pw_torch_not_available():
    """Test that PW fails gracefully when PyTorch is not available"""
    # Create simple data
    np.random.seed(42)
    n = 20
    X = np.random.normal(size=(n, 2))
    A = np.random.binomial(1, 0.5, size=n)

    # Monkeypatch TORCH_AVAILABLE to False
    import permutation_weighting.estimator as estimator
    original_value = getattr(estimator, 'TORCH_AVAILABLE', True)
    setattr(estimator, 'TORCH_AVAILABLE', False)

    try:
        # Should raise ImportError when PyTorch not available
        with pytest.raises(ImportError, match="PyTorch is not available"):
            PW(A, X, classifier='logistic', use_torch=True,
               estimand_params={'bootstrap': True}, num_replicates=2)
    finally:
        # Restore original value
      setattr(estimator, 'TORCH_AVAILABLE', original_value)